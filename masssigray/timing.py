"""Counting time needed to reach a target rms noise of k^w chi(k).

The planners report the noise of one sample scan plus one I0 scan at the
planned dwell times. Two ways to spend more time are compared:

* longer dwells: every dwell (sample and I0) is multiplied by a factor f and
  the full detector model is re-evaluated. Shot noise, dark current and
  per-frame read noise fall as 1/sqrt(f); source instability does not, so the
  noise approaches a floor and a low target may be unreachable.
* repeated scans: N independent scan pairs are averaged. Every noise term is
  independent between scans, so the noise falls exactly as 1/sqrt(N).

Times are counting time plus an optional fixed overhead per point (motor move,
readout), which is paid once per point per scan.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

_F_MIN, _F_MAX = 1e-4, 1e6


@dataclass
class TimeEstimate:
    target: float
    sigma_one: float  # rms noise of one scan pair at the planned dwells
    floor: float  # limit of the noise for very long dwells (source instability)
    time_one_s: float  # one sample scan + one I0 scan, including overhead
    dwell_factor: float | None  # multiply every dwell by this; None if unreachable
    time_dwell_s: float | None
    n_scans: int  # scan pairs to average at the planned dwells
    time_scans_s: float


def _chi_rms(model, m_s, m_m, factor):
    scaled = copy.copy(model)
    scaled.t = model.t * factor
    return float(scaled.evaluate([m_s], [m_m])["chi"][0])


def estimate(model, scan, m_s, m_m, target, overhead_s=0.0) -> TimeEstimate:
    """Time to reach rms sigma(k^w chi) = target for a planner model at masses m_s, m_m (mg)."""
    if target <= 0:
        raise ValueError("target must be positive")
    sigma_one = _chi_rms(model, m_s, m_m, 1.0)
    if not np.isfinite(sigma_one):
        raise ValueError("no noise estimate: the k range is outside the scan or there is no signal")
    t_sample, t_i0 = scan.total_time()
    n_pts = len(model.E)
    per_scan_overhead = 2 * n_pts * overhead_s
    time_one = t_sample + t_i0 + per_scan_overhead

    n_scans = max(1, math.ceil((sigma_one / target) ** 2 - 1e-9))
    # sigma^2(f) ~ a/f + floor^2 at long dwells; extrapolate from two points
    f1, f2 = _F_MAX / 100, _F_MAX
    s1, s2 = _chi_rms(model, m_s, m_m, f1), _chi_rms(model, m_s, m_m, f2)
    floor = math.sqrt(max(0.0, (f2 * s2**2 - f1 * s1**2) / (f2 - f1)))

    if s2 > target:
        f = None
    elif _chi_rms(model, m_s, m_m, _F_MIN) <= target:
        f = _F_MIN
    else:
        g = lambda lf: _chi_rms(model, m_s, m_m, 10**lf) - target
        f = 10 ** brentq(g, math.log10(_F_MIN), math.log10(_F_MAX), xtol=1e-4)
    return TimeEstimate(
        target=target, sigma_one=sigma_one, floor=floor, time_one_s=time_one,
        dwell_factor=f,
        time_dwell_s=None if f is None else f * (t_sample + t_i0) + per_scan_overhead,
        n_scans=n_scans, time_scans_s=n_scans * time_one)
