"""Transmission planner: optimal sample and matrix masses for a pressed pellet,
or optimal thickness for a foil or film.

For sample mass m_s and matrix mass m_m pressed over area A:

    mu*t(E)   = [m_s (mu/rho)_s(E) + m_m (mu/rho)_m(E)] / A
    dmu*t(E)  = [m_s w_s + m_m w_m] * dmu/rho_abs(E) / A      (edge step)
    sigma(mu*t) from counting statistics of the I0 and I scans (detector.py)

The matrix only adds absorption, so for a given sample mass the planner uses
the least matrix the pellet constraint allows. That leaves one free variable,
m_s, which is scanned on a fine log grid to maximize the chosen objective:

    "edge":  edge-step SNR at E0 + E_ref_offset
    "exafs": 1 / rms noise of k^w chi(k) over the k range
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .detector import DetectorModel, sigma_mut
from .flux import FluxSpectrum
from .materials import Material
from .scan import ScanPlan
from .timing import TimeEstimate, estimate
from .xray import edge_energy, edge_step_mu_rho, interfering_edges, k_of_E, mu_rho

CONSTRAINT_MODES = ("min_total_mass", "fixed_total_mass", "min_thickness", "min_matrix_mass", "none")


@dataclass
class PelletConstraint:
    mode: str = "min_total_mass"
    total_mass_mg: float = 100.0  # minimum or fixed total, by mode
    min_thickness_mm: float = 0.5
    packing_fraction: float = 0.7  # pellet density / ideal mixture density
    min_matrix_fraction: float = 0.0  # matrix wt fraction of pellet, lower bound
    max_edge_step: float | None = 1.5
    max_mut: float | None = 4.0  # total mu*t at any scan energy
    min_weighable_mg: float = 2.0
    min_matrix_mg: float = 50.0  # matrix needed to bind the pellet, for "min_matrix_mass"

    def __post_init__(self):
        if self.mode not in CONSTRAINT_MODES:
            raise ValueError(f"mode must be one of {CONSTRAINT_MODES}")


@dataclass
class TransmissionSetup:
    sample: Material
    matrix: Material | None
    area_cm2: float
    absorber: str
    edge: str
    flux: FluxSpectrum
    detector: DetectorModel = field(default_factory=DetectorModel)
    scan: ScanPlan | None = None
    objective: str = "edge"  # "edge" | "exafs"
    E_ref_offset: float = 50.0
    k_range: tuple[float, float] = (3.0, 10.0)
    k_weight: float = 2.0

    def __post_init__(self):
        self.E0 = edge_energy(self.absorber, self.edge)
        if self.scan is None:
            self.scan = ScanPlan(E0=self.E0)
        if self.objective not in ("edge", "exafs"):
            raise ValueError("objective must be 'edge' or 'exafs'")


@dataclass
class TransmissionPlan:
    m_sample_mg: float
    m_matrix_mg: float
    thickness_mm: float | None
    edge_step: float
    mut_below: float
    mut_above: float
    mut_max: float
    snr_edge: float
    chi_noise_rms: float
    binding: str
    feasible: bool
    warnings: list[str]
    curve: dict
    spectra: dict
    dilution: dict | None
    area_cm2: float
    absorber_mass_mg: float

    @property
    def total_mg(self):
        return self.m_sample_mg + self.m_matrix_mg

    @property
    def sample_wt_pct(self):
        return 100 * self.m_sample_mg / self.total_mg if self.total_mg else 0.0


class _Model:
    """Precomputed cross sections on the scan grid."""

    def __init__(self, setup: TransmissionSetup):
        s = setup
        self.s = s
        E, dwell = s.scan.grid()
        self.E, self.dwell = E, dwell
        self.E_ref = s.E0 + s.E_ref_offset
        self.E_below = s.E0 - 20.0
        pts = np.concatenate([E, [self.E_ref, self.E_below]])
        mat = s.matrix
        self.mu_s = mu_rho(s.sample, pts)
        self.mu_m = mu_rho(mat, pts) if mat else np.zeros_like(pts)
        step = edge_step_mu_rho(s.absorber, s.edge, pts)
        self.dmu_s = s.sample.mass_fraction(s.absorber) * step
        self.dmu_m = (mat.mass_fraction(s.absorber) if mat else 0.0) * step
        self.rate = s.flux(pts)
        self.t = np.concatenate([dwell, [np.interp(self.E_ref, E, dwell)] * 2])
        self.k = k_of_E(pts, s.E0)
        kmask = (pts >= s.E0) & (self.k >= s.k_range[0]) & (self.k <= s.k_range[1])
        kmask[-2:] = False
        self.kmask = kmask

    def evaluate(self, m_s, m_m):
        """Vectorized over arrays of masses (mg). Returns dict of arrays."""
        s = self.s
        ms = np.atleast_1d(np.asarray(m_s, float))[:, None] * 1e-3
        mm = np.atleast_1d(np.asarray(m_m, float))[:, None] * 1e-3
        mut = (ms * self.mu_s + mm * self.mu_m) / s.area_cm2
        step = (ms * self.dmu_s + mm * self.dmu_m) / s.area_cm2
        t_i0 = self.t * s.scan.i0_dwell_factor
        sig = sigma_mut(self.rate, mut, self.t, t_i0, s.detector)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = step[:, -2] / sig[:, -2]
            if self.kmask.any():
                kw = self.k[self.kmask] ** s.k_weight
                chi = np.sqrt(np.mean((sig[:, self.kmask] / step[:, self.kmask] * kw) ** 2, axis=1))
            else:
                chi = np.full(ms.shape[0], np.nan)
        n = len(self.E)
        return dict(mut=mut[:, :n], step=step[:, :n], sigma=sig[:, :n],
                    step_ref=step[:, -2], mut_above=mut[:, -2], mut_below=mut[:, -1],
                    mut_max=mut[:, :n].max(axis=1), snr=snr, chi=chi)


def matrix_mass(m_s, setup: TransmissionSetup, c: PelletConstraint):
    """Least matrix mass (mg) satisfying the pellet constraint, for sample mass m_s (mg).

    Returns NaN where the constraint cannot be met (fixed total exceeded).
    """
    m_s = np.asarray(m_s, float)
    if setup.matrix is None or c.mode == "none":
        return np.zeros_like(m_s)
    f = min(max(c.min_matrix_fraction, 0.0), 0.999)
    by_fraction = f / (1 - f) * m_s
    if c.mode == "min_total_mass":
        mm = np.maximum(c.total_mass_mg - m_s, by_fraction)
    elif c.mode == "fixed_total_mass":
        mm = c.total_mass_mg - m_s
        mm = np.where(mm >= by_fraction - 1e-12, mm, np.nan)
    elif c.mode == "min_matrix_mass":
        mm = np.maximum(c.min_matrix_mg, by_fraction)
    else:  # min_thickness
        rho_s, rho_m = setup.sample.density, setup.matrix.density
        if not (rho_s and rho_m):
            raise ValueError("min_thickness mode needs sample and matrix densities")
        vol_needed = c.packing_fraction * setup.area_cm2 * c.min_thickness_mm / 10.0  # cm3
        mm = rho_m * 1e3 * (vol_needed - m_s * 1e-3 / rho_s)
        mm = np.maximum(mm, by_fraction)
    return mm


def pellet_thickness_mm(m_s, m_m, setup: TransmissionSetup, packing: float):
    rho_s = setup.sample.density
    rho_m = setup.matrix.density if setup.matrix else None
    if not rho_s or (m_m > 0 and not rho_m):
        return None
    vol = m_s * 1e-3 / rho_s + (m_m * 1e-3 / rho_m if m_m > 0 else 0.0)
    return 10.0 * vol / (setup.area_cm2 * packing)


def _nice_ceiling(x):
    exp = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * exp >= x - 1e-12:
            return m * exp
    return 10 * exp


def dilution_recipe(m_s, m_m, min_weigh):
    """Two-step recipe when the sample mass is too small to weigh directly."""
    if m_s <= 0 or m_s >= min_weigh:
        return None
    need = min_weigh / m_s
    D = _nice_ceiling(need)
    d_max = 1 + m_m / m_s
    if D > d_max:
        D = d_max
    if D < need * 0.999:
        return dict(possible=False, message="Total pellet mass is below the weighable "
                    "limit; increase the pellet mass or the area.")
    stock_sample = max(min_weigh, 10.0)
    return dict(possible=True, factor=D,
                stock_sample_mg=stock_sample, stock_matrix_mg=stock_sample * (D - 1),
                stock_for_pellet_mg=m_s * D, extra_matrix_mg=m_m - m_s * (D - 1))


def _describe(setup, c, model, m_s, m_m, ev, binding, feasible, extra_warnings=()):
    s = setup
    i = 0
    warnings = list(extra_warnings)
    spectra = dict(E=model.E, dwell=model.dwell, k=k_of_E(model.E, s.E0),
                   mut=ev["mut"][i], step=ev["step"][i], sigma=ev["sigma"][i],
                   rate=model.rate[:len(model.E)])
    spectra["counts_I0"] = spectra["rate"] * model.dwell * s.scan.i0_dwell_factor
    spectra["counts_I"] = spectra["rate"] * model.dwell * np.exp(-spectra["mut"])
    with np.errstate(divide="ignore", invalid="ignore"):
        spectra["sigma_chi"] = np.where(spectra["step"] > 0, spectra["sigma"] / spectra["step"], np.nan)

    if not s.flux.covers(model.E[0], model.E[-1]):
        warnings.append(f"I0 spectrum ({s.flux.energy[0]:.0f}-{s.flux.energy[-1]:.0f} eV) does not "
                        f"cover the scan ({model.E[0]:.0f}-{model.E[-1]:.0f} eV); points outside "
                        "have no flux.")
    elems = set(s.sample.elements) | (set(s.matrix.elements) if s.matrix else set())
    others = interfering_edges(elems, model.E[0], model.E[-1], exclude=(s.absorber, s.edge))
    if others:
        txt = ", ".join(f"{el} {ed} ({e:.0f} eV)" for el, ed, e in others)
        warnings.append(f"Other absorption edges inside the scan range: {txt}.")
    if s.matrix and s.matrix.mass_fraction(s.absorber) > 0:
        warnings.append(f"The matrix contains {s.absorber}; it contributes to the edge step.")
    if s.sample.mass_fraction(s.absorber) == 0:
        warnings.append(f"The sample contains no {s.absorber}.")
    step = float(ev["step_ref"][i])
    if step < 0.1:
        warnings.append(f"Edge step {step:.3f} is small: transmission will be noise-limited; "
                        "consider fluorescence detection.")
    if s.detector.max_rate and np.nanmax(spectra["rate"]) > s.detector.max_rate:
        warnings.append("I0 count rate exceeds the detector's maximum rate somewhere in the scan.")
    dil = dilution_recipe(m_s, m_m, c.min_weighable_mg) if s.matrix else None
    if dil:
        warnings.append(f"Sample mass {m_s:.2f} mg is below the {c.min_weighable_mg:g} mg weighing "
                        "limit; use the dilution recipe.")
    w_abs = s.sample.mass_fraction(s.absorber) * m_s + (
        s.matrix.mass_fraction(s.absorber) * m_m if s.matrix else 0.0)
    return TransmissionPlan(
        m_sample_mg=float(m_s), m_matrix_mg=float(m_m),
        thickness_mm=pellet_thickness_mm(m_s, m_m, s, c.packing_fraction if s.matrix else 1.0),
        edge_step=step, mut_below=float(ev["mut_below"][i]), mut_above=float(ev["mut_above"][i]),
        mut_max=float(ev["mut_max"][i]), snr_edge=float(ev["snr"][i]),
        chi_noise_rms=float(ev["chi"][i]), binding=binding, feasible=feasible,
        warnings=warnings, curve={}, spectra=spectra, dilution=dil,
        area_cm2=s.area_cm2, absorber_mass_mg=float(w_abs))


def optimize(setup: TransmissionSetup, c: PelletConstraint | None = None,
             n_grid: int = 1500) -> TransmissionPlan:
    """Find the sample mass (and matrix mass) that maximizes the objective."""
    c = c or PelletConstraint()
    model = _Model(setup)
    mu_max_s = float(np.max(model.mu_s[:-2]))
    cap = c.max_mut or 8.0
    hi = 1.5 * cap * setup.area_cm2 / mu_max_s * 1e3  # mg; sample alone at 1.5x the cap
    if setup.matrix is not None and c.mode == "fixed_total_mass":
        f = min(max(c.min_matrix_fraction, 0.0), 0.999)
        hi = min(hi, c.total_mass_mg * (1 - f))
    lo = hi * 1e-6
    ms = np.geomspace(lo, hi, n_grid)
    mm = matrix_mass(ms, setup, c)
    ev = model.evaluate(ms, np.nan_to_num(mm))
    score = ev["snr"] if setup.objective == "edge" else 1.0 / ev["chi"]

    ok_mass = np.isfinite(mm)
    ok_step = ev["step_ref"] <= c.max_edge_step if c.max_edge_step else np.ones_like(ok_mass)
    ok_mut = ev["mut_max"] <= c.max_mut if c.max_mut else np.ones_like(ok_mass)
    feasible = ok_mass & ok_step & ok_mut & np.isfinite(score)
    warnings = []
    is_feasible = bool(feasible.any())
    if not is_feasible:
        relaxed = ok_mass & ok_step & np.isfinite(score)
        if relaxed.any():
            warnings.append(f"No loading keeps mu*t below {c.max_mut:g} everywhere in the scan "
                            "(the matrix alone is too absorbing); the mu*t cap was ignored.")
            feasible = relaxed
        else:
            warnings.append("No feasible loading found; check the constraints.")
            feasible = ok_mass & np.isfinite(score)
    best = int(np.argmax(np.where(feasible, score, -np.inf)))

    binding = "SNR optimum"
    if best + 1 < n_grid and not feasible[best + 1]:
        if not ok_step[best + 1]:
            binding = f"edge-step cap ({c.max_edge_step:g})"
        elif not ok_mut[best + 1]:
            binding = f"mu*t cap ({c.max_mut:g})"
        elif not ok_mass[best + 1]:
            binding = "fixed total mass"
    elif best == n_grid - 1:
        binding = "search limit"

    e1 = model.evaluate([ms[best]], [np.nan_to_num(mm[best])])
    plan = _describe(setup, c, model, ms[best], float(np.nan_to_num(mm[best])), e1, binding,
                     is_feasible, warnings)
    plan.curve = dict(m_sample=ms, m_matrix=mm, score=score, snr=ev["snr"], chi=ev["chi"],
                      step=ev["step_ref"], mut_max=ev["mut_max"], feasible=feasible,
                      objective=setup.objective)
    return plan


def evaluate_masses(setup: TransmissionSetup, m_sample_mg: float, m_matrix_mg: float,
                    c: PelletConstraint | None = None) -> TransmissionPlan:
    """Predictions for masses the user chose (or actually weighed)."""
    c = c or PelletConstraint()
    model = _Model(setup)
    ev = model.evaluate([m_sample_mg], [m_matrix_mg])
    notes = []
    if c.max_edge_step and ev["step_ref"][0] > c.max_edge_step:
        notes.append(f"Edge step exceeds the cap of {c.max_edge_step:g}.")
    if c.max_mut and ev["mut_max"][0] > c.max_mut:
        notes.append(f"mu*t exceeds the cap of {c.max_mut:g} in part of the scan.")
    return _describe(setup, c, model, m_sample_mg, m_matrix_mg, ev, "user masses",
                     not notes, notes)


def time_for_target(setup: TransmissionSetup, m_sample_mg: float, m_matrix_mg: float,
                    target: float, overhead_s: float = 0.0) -> TimeEstimate:
    """Counting time to reach rms sigma(k^w chi) = target at the given masses (see timing.py)."""
    return estimate(_Model(setup), setup.scan, m_sample_mg, m_matrix_mg, target, overhead_s)


def foil_setup(material: Material, absorber: str, edge: str, flux: FluxSpectrum,
               detector: DetectorModel | None = None, scan: ScanPlan | None = None,
               objective: str = "edge", **kw) -> TransmissionSetup:
    """A matrix-free setup over 1 cm2: sample 'mass' in mg is the areal density in mg/cm2."""
    return TransmissionSetup(sample=material, matrix=None, area_cm2=1.0, absorber=absorber,
                             edge=edge, flux=flux, detector=detector or DetectorModel(),
                             scan=scan, objective=objective, **kw)


def areal_to_thickness_um(areal_mg_cm2: float, density: float) -> float:
    return areal_mg_cm2 * 1e-3 / density * 1e4
