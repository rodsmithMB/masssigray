"""X-ray cross sections and absorption edges (thin wrapper around xraydb).

Mass attenuation coefficients are from the Elam tables in xraydb, in cm2/g.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import xraydb

from .materials import Material

HC_K = 0.262468  # 2m/hbar^2 in eV^-1 A^-2, k = sqrt(HC_K*(E-E0))
EDGE_NAMES = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")


def k_of_E(E, E0):
    E = np.asarray(E, dtype=float)
    return np.sqrt(np.clip(HC_K * (E - E0), 0, None))


def E_of_k(k, E0):
    return E0 + np.asarray(k, dtype=float) ** 2 / HC_K


def edge_energy(element: str, edge: str) -> float:
    return float(xraydb.xray_edge(element, edge).energy)


def element_edges(element: str, emin: float = 0.0, emax: float = np.inf) -> dict[str, float]:
    """Edges of an element in [emin, emax], {edge: energy}."""
    out = {}
    for name, e in xraydb.xray_edges(element).items():
        if name in EDGE_NAMES and emin <= e.energy <= emax:
            out[name] = float(e.energy)
    return out


def mu_rho_element(element: str, E, kind: str = "total") -> np.ndarray:
    """Mass attenuation coefficient of an element, cm2/g."""
    return np.asarray(xraydb.mu_elam(element, np.atleast_1d(np.asarray(E, float)), kind=kind))


def mu_rho(material: Material, E, kind: str = "total") -> np.ndarray:
    """Mass attenuation coefficient of a material, cm2/g."""
    E = np.atleast_1d(np.asarray(E, float))
    out = np.zeros_like(E)
    for el, w in material.mass_fractions.items():
        out += w * mu_rho_element(el, E, kind)
    return out


@lru_cache(maxsize=256)
def _pre_edge_fit(element: str, edge: str) -> tuple[float, float]:
    """Log-log line through the element's attenuation just below the edge."""
    E0 = edge_energy(element, edge)
    lower = [e for e in element_edges(element, 0, E0 - 0.5).values()]
    floor = max(lower) + 2.0 if lower else 0.0
    hi = E0 - 2.0
    lo = max(E0 - 80.0, floor)
    if lo >= hi - 1.0:
        lo = hi - 1.0
    e = np.array([lo, hi])
    mu = mu_rho_element(element, e, kind="photo")
    slope = np.log(mu[1] / mu[0]) / np.log(e[1] / e[0])
    return float(slope), float(np.log(mu[1]) - slope * np.log(e[1]))


def edge_step_mu_rho(element: str, edge: str, E) -> np.ndarray:
    """Edge contribution of one element, cm2 per gram of that element.

    The photoabsorption above the edge minus the pre-edge extrapolation; zero
    below the edge. Decays with energy like the real atomic background, so it
    is the right normalization for chi(k) noise at high k.
    """
    E = np.atleast_1d(np.asarray(E, float))
    E0 = edge_energy(element, edge)
    slope, icpt = _pre_edge_fit(element, edge)
    pre = np.exp(icpt + slope * np.log(E))
    above = mu_rho_element(element, E, kind="photo")
    step = np.where(E >= E0, above - pre, 0.0)
    return np.clip(step, 0, None)


def interfering_edges(elements, E_lo: float, E_hi: float, exclude: tuple[str, str] | None = None):
    """Edges of any of `elements` inside [E_lo, E_hi]: list of (element, edge, energy)."""
    found = []
    for el in elements:
        for name, e in element_edges(el, E_lo, E_hi).items():
            if exclude and (el, name) == exclude:
                continue
            found.append((el, name, e))
    return sorted(found, key=lambda t: t[2])
