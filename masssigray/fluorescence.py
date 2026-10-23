"""Fluorescence planner for pellets: signal, self-absorption and noise.

For a uniform pellet with optical depths tau(E) (total, incident energy),
tau_f (total, at the fluorescence energy) and a(E) (absorber edge part), with
incidence/exit angles theta_in/theta_out measured from the surface:

    X     = tau/sin(theta_in) + tau_f/sin(theta_out)
    If/I0 = (Omega/4pi) * eta * Y * (a/sin(theta_in)) * (1 - exp(-X)) / X

Self-absorption reduces the EXAFS amplitude by the retention factor

    R = d ln(If) / d ln(a) = 1 - (a/sin(theta_in)) * (1/X - exp(-X)/(1 - exp(-X)))

which tends to 1 for dilute or thin samples and to (mu_other + g mu_f)/(mu_total + g mu_f)
for thick ones. Unlike transmission, extra matrix can help (it lowers a/X), so
the planner searches over both sample and matrix mass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xraydb

from .detector import DetectorModel
from .flux import FluxSpectrum
from .materials import Material
from .scan import ScanPlan
from .transmission import PelletConstraint, dilution_recipe, matrix_mass, pellet_thickness_mm
from .xray import edge_energy, edge_step_mu_rho, k_of_E, mu_rho

EDGE_LINE = {"K": "Ka", "L3": "La", "L2": "Lb", "L1": "Lb", "M5": "Ma", "M4": "Mb"}


@dataclass
class FluorescenceSetup:
    sample: Material
    matrix: Material | None
    area_cm2: float
    absorber: str
    edge: str
    flux: FluxSpectrum
    scan: ScanPlan | None = None
    theta_in: float = 45.0
    theta_out: float = 45.0
    solid_angle_frac: float = 0.01  # Omega / 4pi
    det_efficiency: float = 1.0
    flux_scale: float = 1.0  # incident rate on the sample / I0 rate in the flux file
    background_per_photon: float = 0.0  # scatter counts in the ROI per incident photon
    i0_detector: DetectorModel = field(default_factory=DetectorModel)
    objective: str = "exafs"
    E_ref_offset: float = 50.0
    k_range: tuple[float, float] = (3.0, 10.0)
    k_weight: float = 2.0
    min_retention: float = 0.9
    max_total_mass_mg: float = 300.0

    def __post_init__(self):
        self.E0 = edge_energy(self.absorber, self.edge)
        if self.scan is None:
            self.scan = ScanPlan(E0=self.E0)

    def line(self):
        """(yield*probability, mean line energy) for the main fluorescence line."""
        fam = EDGE_LINE.get(self.edge, "Ka")
        y, e, p = xraydb.fluor_yield(self.absorber, self.edge, fam, self.E0 + self.E_ref_offset)
        if not e:
            raise ValueError(f"no {fam} fluorescence line for {self.absorber} {self.edge}")
        return y * p, e


@dataclass
class FluorescencePlan:
    m_sample_mg: float
    m_matrix_mg: float
    thickness_mm: float | None
    retention_ref: float
    retention_min_k: float
    fluo_rate_ref: float  # counts/s in the fluorescence ROI at E_ref
    snr_edge: float
    chi_noise_rms: float
    line_energy: float
    binding: str
    warnings: list[str]
    curve: dict
    spectra: dict
    dilution: dict | None

    @property
    def total_mg(self):
        return self.m_sample_mg + self.m_matrix_mg

    @property
    def sample_wt_pct(self):
        return 100 * self.m_sample_mg / self.total_mg if self.total_mg else 0.0


class _FModel:
    def __init__(self, s: FluorescenceSetup):
        self.s = s
        E, dwell = s.scan.grid()
        self.E, self.dwell = E, dwell
        self.E_ref = s.E0 + s.E_ref_offset
        pts = np.concatenate([E, [self.E_ref]])
        self.Y, self.E_f = s.line()
        self.mu_s = mu_rho(s.sample, pts)
        self.mu_m = mu_rho(s.matrix, pts) if s.matrix else np.zeros_like(pts)
        self.muf_s = float(mu_rho(s.sample, self.E_f)[0])
        self.muf_m = float(mu_rho(s.matrix, self.E_f)[0]) if s.matrix else 0.0
        step = edge_step_mu_rho(s.absorber, s.edge, pts)
        self.dmu_s = s.sample.mass_fraction(s.absorber) * step
        self.dmu_m = (s.matrix.mass_fraction(s.absorber) if s.matrix else 0.0) * step
        self.rate = s.flux(pts) * s.flux_scale
        self.t = np.concatenate([dwell, [np.interp(self.E_ref, E, dwell)]])
        self.k = k_of_E(pts, s.E0)
        km = (pts >= s.E0) & (self.k >= s.k_range[0]) & (self.k <= s.k_range[1])
        km[-1] = False
        self.kmask = km
        self.sin_in = np.sin(np.radians(s.theta_in))
        self.sin_out = np.sin(np.radians(s.theta_out))

    def evaluate(self, m_s, m_m):
        s = self.s
        ms = np.atleast_1d(np.asarray(m_s, float))[:, None] * 1e-3
        mm = np.atleast_1d(np.asarray(m_m, float))[:, None] * 1e-3
        A = s.area_cm2
        tau = (ms * self.mu_s + mm * self.mu_m) / A
        tau_f = (ms * self.muf_s + mm * self.muf_m) / A
        a = (ms * self.dmu_s + mm * self.dmu_m) / A
        X = tau / self.sin_in + tau_f / self.sin_out
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            esc = np.where(X > 1e-8, -np.expm1(-X) / X, 1.0 - X / 2)
            yld = s.solid_angle_frac * s.det_efficiency * self.Y * (a / self.sin_in) * esc
            # 1/X - e^-X/(1-e^-X), stable for small X
            corr = np.where(X > 1e-6, 1.0 / X - 1.0 / np.expm1(X), 0.5 - X / 12)
            R = 1.0 - (a / self.sin_in) * corr
            n_f = self.rate * self.t * yld
            n_bg = self.rate * self.t * s.background_per_photon
            n0 = self.rate * self.t * s.scan.i0_dwell_factor
            t0 = self.t * s.scan.i0_dwell_factor
            rel = (n_f + n_bg) / n_f**2 + s.i0_detector.variance(n0, t0) / n0**2
            sig_rel = np.where(n_f > 0, np.sqrt(rel), np.inf)
            sig_chi = sig_rel / R
            snr = 1.0 / sig_rel[:, -1]
            if self.kmask.any():
                kw = self.k[self.kmask] ** s.k_weight
                chi = np.sqrt(np.mean((sig_chi[:, self.kmask] * kw) ** 2, axis=1))
                rmin = R[:, self.kmask].min(axis=1)
            else:
                chi = np.full(ms.shape[0], np.nan)
                rmin = R[:, -1]
        n = len(self.E)
        return dict(yld=yld[:, :n], R=R[:, :n], n_f=n_f[:, :n], sig_chi=sig_chi[:, :n],
                    R_ref=R[:, -1], rate_ref=(self.rate[-1] * yld[:, -1]), snr=snr,
                    chi=chi, R_min=rmin)


def optimize(setup: FluorescenceSetup, c: PelletConstraint | None = None,
             n_sample: int = 300, n_matrix: int = 40) -> FluorescencePlan:
    c = c or PelletConstraint()
    model = _FModel(setup)
    s = setup
    Mmax = max(s.max_total_mass_mg, c.total_mass_mg)
    hi = Mmax if s.matrix else 50.0 * s.area_cm2 * 1e3 / float(np.max(model.mu_s))
    ms = np.geomspace(hi * 1e-6, hi, n_sample)
    mm0 = matrix_mass(ms, setup, c)
    if s.matrix is None or c.mode in ("fixed_total_mass", "none"):
        lam = np.array([0.0])
    else:
        lam = np.linspace(0, 1, n_matrix)
    MS = np.repeat(ms, lam.size)
    MM0 = np.repeat(mm0, lam.size)
    LAM = np.tile(lam, ms.size)
    room = np.clip(Mmax - MS - np.nan_to_num(MM0), 0, None)
    MM = np.nan_to_num(MM0) + LAM * room
    ev = model.evaluate(MS, MM)
    score = ev["snr"] if s.objective == "edge" else 1.0 / ev["chi"]
    ok = np.isfinite(MM0) & np.isfinite(score) & (MS + MM <= Mmax + 1e-9)
    ok_ret = ev["R_min"] >= s.min_retention
    feas = ok & ok_ret
    warnings = []
    if not feas.any():
        warnings.append(f"No loading reaches the {s.min_retention:g} retention target; "
                        "showing the best retention achievable.")
        pick = np.where(ok, ev["R_min"], -np.inf)
    else:
        pick = np.where(feas, score, -np.inf)
    b = int(np.argmax(pick))
    m_s, m_m = float(MS[b]), float(MM[b])

    # best score per sample mass, for the plot
    sc = np.where(feas, score, np.nan).reshape(ms.size, lam.size)
    with np.errstate(all="ignore"):
        best_per = np.nanmax(np.where(np.isnan(sc), -np.inf, sc), axis=1)
    best_per[~np.isfinite(best_per)] = np.nan
    ret = ev["R_min"].reshape(ms.size, lam.size)[:, 0]

    binding = "SNR optimum"
    if not feas.any():
        binding = "retention target not reachable"
    elif ev["R_min"][b] < s.min_retention + 0.01:
        binding = f"retention target ({s.min_retention:g})"
    elif MS[b] + MM[b] >= Mmax * 0.999:
        binding = f"maximum pellet mass ({Mmax:g} mg)"
    e1 = model.evaluate([m_s], [m_m])
    spectra = dict(E=model.E, k=k_of_E(model.E, s.E0), dwell=model.dwell,
                   R=e1["R"][0], counts_f=e1["n_f"][0], sigma_chi=e1["sig_chi"][0],
                   rate=model.rate[:-1])
    pres = [el for el in (set(s.sample.elements) | (set(s.matrix.elements) if s.matrix else set()))
            if el != s.absorber]
    if s.background_per_photon == 0:
        warnings.append("Scatter background is set to zero; real SNR will be lower.")
    dil = dilution_recipe(m_s, m_m, c.min_weighable_mg) if s.matrix else None
    if dil:
        warnings.append(f"Sample mass {m_s:.2f} mg is below the {c.min_weighable_mg:g} mg "
                        "weighing limit; use the dilution recipe.")
    for el in pres:
        for name, e in xraydb.xray_edges(el).items():
            if model.E_f < e.energy < model.E[-1] and e.energy > s.E0 - 200 and name in ("K", "L3"):
                warnings.append(f"{el} {name} edge ({e.energy:.0f} eV) may add fluorescence "
                                "or scatter background.")
    return FluorescencePlan(
        m_sample_mg=m_s, m_matrix_mg=m_m,
        thickness_mm=pellet_thickness_mm(m_s, m_m, setup, c.packing_fraction) if s.matrix else None,
        retention_ref=float(e1["R_ref"][0]), retention_min_k=float(e1["R_min"][0]),
        fluo_rate_ref=float(e1["rate_ref"][0]), snr_edge=float(e1["snr"][0]),
        chi_noise_rms=float(e1["chi"][0]), line_energy=float(model.E_f), binding=binding,
        warnings=warnings, spectra=spectra, dilution=dil,
        curve=dict(m_sample=ms, best_score=best_per, retention=ret, objective=s.objective))
