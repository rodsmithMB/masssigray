"""Detector noise model and photon-transfer calibration.

Variance of the signal recorded in one energy point (ROI sum), in effective
counts squared:

    var(S) = n_frames * read_var + dark_rate * t + S + (instability * S)^2

* read_var: read noise variance per frame, summed over the ROI
* dark_rate: dark-current variance per second, summed over the ROI
* S: shot noise. In effective counts this term has unit slope by definition;
  the camera gain and any excess-noise factor are absorbed in the conversion
  from detector units (ADU) to effective counts.
* instability: fractional point-to-point flux fluctuation of the source that
  does not average out with counting (after removing whole-scan drift).

Calibration fits the same model in ADU to repeated I0 scans recorded at
several dwell times, plus optional dark scans:

    var_ADU = R + D * t + g * m + eps^2 * m^2

g is ADU per effective count. Divide measured I0 (ADU/s) by g to get the
effective count rate used everywhere else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls


@dataclass
class DetectorModel:
    read_var: float = 0.0
    dark_rate: float = 0.0
    instability: float = 0.0
    frame_time: float | None = None  # s; None means one frame per point
    max_rate: float | None = None  # effective counts/s before saturation
    label: str = "ideal Poisson"

    def n_frames(self, t):
        t = np.asarray(t, float)
        if not self.frame_time:
            return np.ones_like(t)
        return np.maximum(1.0, np.ceil(t / self.frame_time - 1e-9))

    def variance(self, counts, t):
        counts = np.asarray(counts, float)
        return (self.n_frames(t) * self.read_var + self.dark_rate * np.asarray(t, float)
                + counts + (self.instability * counts) ** 2)


def sigma_mut(rate, mut, t_sample, t_i0, det: DetectorModel):
    """Standard deviation of mu*t = ln(I0/I) for one energy point.

    rate: I0 count rate (effective counts/s); I0 and I are recorded in separate
    scans with dwell times t_i0 and t_sample.
    """
    rate = np.asarray(rate, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        n0 = rate * t_i0
        n = n0 * (t_sample / t_i0) * np.exp(-np.asarray(mut, float))
        rel = det.variance(n, t_sample) / n**2 + det.variance(n0, t_i0) / n0**2
        return np.where((n > 0) & (n0 > 0), np.sqrt(rel), np.inf)


# ---------------------------------------------------------------- calibration

@dataclass
class Scan:
    dwell: float  # s per point
    energy: np.ndarray
    values: np.ndarray  # ADU per point (ROI sum)
    dark: bool = False


@dataclass
class CalibrationResult:
    gain: float  # ADU per effective count
    read_var_adu: float
    dark_var_rate_adu: float
    instability: float
    drift_rms: float  # rms whole-scan intensity change between repeats
    dark_level_adu_per_s: float
    n_points: int
    r2: float  # mean observed/predicted variance
    points: dict = field(repr=False, default_factory=dict)  # t, mean, var for plots

    def to_detector(self, frame_time=None, max_rate=None) -> DetectorModel:
        g2 = self.gain**2
        return DetectorModel(read_var=self.read_var_adu / g2,
                             dark_rate=self.dark_var_rate_adu / g2,
                             instability=self.instability, frame_time=frame_time,
                             max_rate=max_rate, label="calibrated")

    def predicted_var(self, mean_adu, t):
        mean_adu = np.asarray(mean_adu, float)
        return (self.read_var_adu + self.dark_var_rate_adu * np.asarray(t, float)
                + self.gain * mean_adu + (self.instability * mean_adu) ** 2)

    def summary(self) -> dict:
        return {
            "gain (ADU per effective count)": self.gain,
            "read-noise variance (ADU^2/point)": self.read_var_adu,
            "dark variance rate (ADU^2/s)": self.dark_var_rate_adu,
            "point-to-point flux instability (fraction)": self.instability,
            "scan-to-scan drift rms (fraction)": self.drift_rms,
            "dark level (ADU/s)": self.dark_level_adu_per_s,
            "points used": self.n_points,
            "observed / predicted variance (should be ~1)": self.r2,
        }


def _group(scans):
    groups: dict[float, list[Scan]] = {}
    for s in scans:
        groups.setdefault(round(float(s.dwell), 9), []).append(s)
    return groups


def _common(scans):
    """Values of repeated scans on the energy grid of the first one."""
    E = np.asarray(scans[0].energy, float)
    return np.array([np.interp(E, s.energy, s.values) for s in scans])


def calibrate(scans: list[Scan], min_counts: float = 0.0) -> CalibrationResult:
    """Fit the noise model to repeated I0 scans (and optional dark scans).

    Needs at least two repeats at one or more dwell times. Use several dwell
    times (a factor of 10 or more apart) to separate read noise from dark
    current and shot noise.
    """
    light = _group([s for s in scans if not s.dark])
    darks = _group([s for s in scans if s.dark])

    # Dark level per second, for offset subtraction of the light scans
    dark_level, dark_rows = 0.0, []
    if darks:
        levels = []
        for t, grp in darks.items():
            V = _common(grp)
            levels.append(V.mean() / t)
            if len(grp) >= 2:
                dark_rows.append((t, V.var(axis=0, ddof=1).mean()))
            elif V.shape[1] > 3:  # single dark scan: point-to-point variance
                dark_rows.append((t, V[0].var(ddof=1)))
        dark_level = float(np.mean(levels))

    rows, rhs, pts, drifts = [], [], {"t": [], "mean": [], "var": []}, []
    for t, grp in light.items():
        if len(grp) < 2:
            continue
        V = _common(grp) - dark_level * t
        ref = V.mean(axis=0)
        good = ref > max(min_counts, 0) + 1e-12
        scale = np.array([np.median(v[good] / ref[good]) for v in V])
        drifts.extend(scale / scale.mean() - 1)
        Vc = V / scale[:, None]
        m = Vc.mean(axis=0)
        v = Vc.var(axis=0, ddof=1)
        ok = good & (v > 0)
        for mi, vi in zip(m[ok], v[ok]):
            rows.append([1.0, t, mi, mi * mi])
            rhs.append(vi)
            pts["t"].append(t)
            pts["mean"].append(mi)
            pts["var"].append(vi)
    if len(rows) < 4:
        raise ValueError("need at least two repeated I0 scans at one dwell time")
    for t, v in dark_rows:
        rows.append([1.0, t, 0.0, 0.0])
        rhs.append(v)

    X, y = np.array(rows), np.array(rhs)
    if len(set(pts["t"]) | {t for t, _ in dark_rows}) < 2:
        X[:, 1] = 0.0  # one dwell time: dark rate is not separable from read noise
    # Iteratively reweighted least squares. Weighting by the observed variance
    # would bias the fit low (few repeats per point), so weight by the predicted one.
    pred = np.full_like(y, y.mean())
    for _ in range(6):
        Aw = X / pred[:, None]
        norm = np.linalg.norm(Aw, axis=0)
        norm[norm == 0] = 1.0
        coef, _ = nnls(Aw / norm, y / pred)
        coef = coef / norm
        pred = np.maximum(X @ coef, 1e-12)
    R, D, g, eps2 = coef
    if g <= 0:
        raise ValueError("fit gave no shot-noise term; check that scans are dark-subtracted "
                         "and span a useful range of signal levels")
    res = CalibrationResult(gain=float(g), read_var_adu=float(R), dark_var_rate_adu=float(D),
                            instability=float(math.sqrt(eps2)),
                            drift_rms=float(np.sqrt(np.mean(np.square(drifts)))) if drifts else 0.0,
                            dark_level_adu_per_s=dark_level, n_points=len(pts["t"]), r2=0.0,
                            points={k: np.array(v) for k, v in pts.items()})
    res.r2 = float(np.mean(res.points["var"] / res.predicted_var(res.points["mean"], res.points["t"])))
    return res


def simulate_scans(rate_fn, energy, dwells=(0.5, 1, 2, 5, 10), repeats=3, gain=2.5,
                   read_sd_adu=30.0, dark_adu_per_s=50.0, instability=0.002, drift=0.01,
                   n_dark=2, rng=None) -> list[Scan]:
    """Synthetic calibration data, for tests and demonstrations."""
    rng = np.random.default_rng(rng)
    energy = np.asarray(energy, float)
    scans = []
    for t in dwells:
        for _ in range(repeats):
            n = rate_fn(energy) * t * (1 + drift * rng.standard_normal())
            n = n * (1 + instability * rng.standard_normal(n.size))
            counts = rng.poisson(np.clip(n, 0, None))
            dark = rng.poisson(dark_adu_per_s * t, size=n.size)
            adu = gain * counts + dark + read_sd_adu * rng.standard_normal(n.size)
            scans.append(Scan(t, energy, adu))
        for _ in range(n_dark):
            dark = rng.poisson(dark_adu_per_s * t, size=energy.size)
            scans.append(Scan(t, energy, dark + read_sd_adu * rng.standard_normal(energy.size), dark=True))
    return scans
