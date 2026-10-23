"""Predicted spectra: tabulated absorption, optionally shaped by a reference
XANES/EXAFS, with counting noise.

Tabulated cross sections give a smooth edge step with no white line. To
predict peak absorbance, supply a reference spectrum of a similar compound;
it is normalized (flattened), aligned to the tabulated edge, and scaled by the
predicted edge step.
"""
from __future__ import annotations

import numpy as np

from .flux import read_table


def normalize(E, mu, e0=None, pre=(-150.0, -30.0), post=(50.0, None)):
    """Flattened normalization: 0 before the edge, ~1 (plus oscillations) after.

    Returns (e0, norm). e0 defaults to the maximum of the derivative.
    """
    E = np.asarray(E, float)
    mu = np.asarray(mu, float)
    if e0 is None:
        d = np.gradient(mu, E)
        e0 = float(E[np.argmax(d)])
    pm = (E >= e0 + pre[0]) & (E <= e0 + pre[1])
    if pm.sum() < 3:
        pm = E <= e0 - 10
    if pm.sum() < 2:
        pm = np.arange(E.size) < 3
    pre_fit = np.polyfit(E[pm], mu[pm], 1)
    hi = post[1] if post[1] is not None else E[-1] - e0
    qm = (E >= e0 + post[0]) & (E <= e0 + hi)
    if qm.sum() < 4:
        qm = E >= e0 + 10
    post_fit = np.polyfit(E[qm], mu[qm], 2 if qm.sum() > 6 else 1)
    pre_line = np.polyval(pre_fit, E)
    jump = np.polyval(post_fit, E) - pre_line
    jump0 = np.polyval(post_fit, e0) - np.polyval(pre_fit, e0)
    norm = (mu - pre_line) / jump0
    above = E >= e0
    norm[above] = (mu[above] - pre_line[above]) / jump[above]
    return e0, norm


def load_reference(source, energy_col=0, value_col=1, energy_unit="auto",
                   normalized=False, kind="mu"):
    """Load a reference spectrum and return (E_relative_to_e0, norm).

    kind="mu" for absorbance; kind="transmission" for I/I0 (converted with -ln).
    """
    df = read_table(source)
    cols = list(df.columns)
    E = df[cols[energy_col]].to_numpy(float)
    y = df[cols[value_col]].to_numpy(float)
    if energy_unit == "keV" or (energy_unit == "auto" and E.max() < 200):
        E = E * 1000
    order = np.argsort(E)
    E, y = E[order], y[order]
    if kind == "transmission":
        y = -np.log(np.clip(y, 1e-12, None))
    if normalized:
        d = np.gradient(y, E)
        e0 = float(E[np.argmax(d)])
        return E - e0, y
    e0, norm = normalize(E, y)
    return E - e0, norm


def model_mut(spectra: dict, E0: float, reference=None, shift: float = 0.0):
    """mu*t(E) on the scan grid, with the edge step shaped by `reference`.

    spectra: dict with E, mut, step (from a TransmissionPlan).
    reference: (E_rel, norm) from load_reference, or None for the bare tables.
    """
    E = spectra["E"]
    mut, step = spectra["mut"], spectra["step"]
    if reference is None:
        return mut.copy()
    above = E >= E0
    if not above.any():
        return mut.copy()
    step_c = np.interp(E, E[above], step[above], left=step[above][0])
    Er, norm = reference
    shape = np.interp(E - E0 - shift, Er, norm, left=0.0, right=norm[-1])
    return mut - step + step_c * shape


def add_noise(mut_model, spectra: dict, det, i0_dwell_factor: float = 1.0, rng=None):
    """Simulate a measured spectrum: I0 and I drawn with the detector noise model.

    Returns (mut_noisy, sigma) where sigma is recomputed for the shaped model.
    """
    from .detector import sigma_mut

    rng = np.random.default_rng(rng)
    t = spectra["dwell"]
    sig = sigma_mut(spectra["rate"], mut_model, t, t * i0_dwell_factor, det)
    noise = rng.standard_normal(mut_model.size) * np.where(np.isfinite(sig), sig, 0.0)
    return mut_model + noise, sig
