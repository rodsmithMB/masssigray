"""Measured incident flux, I0(E).

I0 is the count rate the detector records with no sample in the beam, as a
function of energy. Rates are in *effective counts* per second: detector units
divided by the gain from detector calibration (see detector.py), so that
Poisson statistics apply directly. With an ideal photon counter, effective
counts are photons.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FluxSpectrum:
    energy: np.ndarray  # eV, increasing
    rate: np.ndarray  # effective counts / s
    label: str = ""

    def __post_init__(self):
        e = np.asarray(self.energy, float)
        r = np.asarray(self.rate, float)
        order = np.argsort(e)
        self.energy, self.rate = e[order], np.clip(r[order], 0, None)

    @classmethod
    def constant(cls, rate: float, emin: float = 1000.0, emax: float = 40000.0):
        return cls(np.array([emin, emax]), np.array([rate, rate]), label=f"constant {rate:g} /s")

    def __call__(self, E) -> np.ndarray:
        """Interpolated rate; zero outside the measured range."""
        return np.interp(np.asarray(E, float), self.energy, self.rate, left=0.0, right=0.0)

    def covers(self, emin: float, emax: float) -> bool:
        return self.energy[0] <= emin and emax <= self.energy[-1]

    def scaled(self, factor: float) -> "FluxSpectrum":
        return FluxSpectrum(self.energy, self.rate * factor, self.label)


def read_table(source) -> pd.DataFrame:
    """Read a whitespace/comma/tab separated numeric table, skipping text headers.

    `source` is a path, a file-like object, or bytes. Comment lines start with #.
    Column names are taken from the last comment line if it has the right
    number of fields, otherwise they are col0, col1, ...
    """
    if isinstance(source, (bytes, bytearray)):
        text = source.decode("utf-8", errors="replace")
    elif hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    else:
        with open(source, encoding="utf-8", errors="replace") as fh:
            text = fh.read()

    header, rows = None, []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        fields = s.replace(",", " ").replace(";", " ").split()
        try:
            rows.append([float(f) for f in fields])
        except ValueError:
            if not rows:
                header = s.lstrip("#").replace(",", " ").replace(";", " ").split()
            continue
    if not rows:
        raise ValueError("no numeric data found")
    width = min(len(r) for r in rows)
    data = np.array([r[:width] for r in rows])
    names = header if header and len(header) == width else [f"col{i}" for i in range(width)]
    return pd.DataFrame(data, columns=names)


def load_flux(source, energy_col=0, value_col=1, value_kind: str = "rate",
              dwell: float | None = None, energy_unit: str = "auto",
              gain: float = 1.0, label: str = "") -> FluxSpectrum:
    """Load an I0 spectrum.

    value_kind: "rate" (per second) or "counts" (per point; needs `dwell` in s).
    energy_unit: "eV", "keV" or "auto" (keV if all energies are below 200).
    gain: detector units per effective count (1 for a photon counter).
    """
    df = read_table(source)
    cols = list(df.columns)
    ecol = cols[energy_col] if isinstance(energy_col, int) else energy_col
    vcol = cols[value_col] if isinstance(value_col, int) else value_col
    E = df[ecol].to_numpy(float)
    v = df[vcol].to_numpy(float)
    if energy_unit == "keV" or (energy_unit == "auto" and E.max() < 200):
        E = E * 1000.0
    if value_kind == "counts":
        if not dwell:
            raise ValueError("dwell time is needed to convert counts to a rate")
        v = v / dwell
    elif value_kind != "rate":
        raise ValueError("value_kind must be 'rate' or 'counts'")
    return FluxSpectrum(E, v / gain, label=label or "measured I0")


def example_flux_csv(emin=4000.0, emax=25000.0, step=10.0, peak=2e5) -> str:
    """A synthetic I0 file for demos: smooth bremsstrahlung-like curve with
    a dip, written as CSV text with energy in eV and rate in counts/s."""
    E = np.arange(emin, emax + step, step)
    shape = (emax * 1.2 - E) / E * np.exp(-((4500.0 / E) ** 3))
    shape /= shape.max()
    shape *= 1 - 0.35 * np.exp(-0.5 * ((E - 11000) / 300) ** 2)
    buf = io.StringIO()
    buf.write("# energy_eV, rate_cps\n")
    for e, r in zip(E, peak * shape):
        buf.write(f"{e:.1f}, {r:.2f}\n")
    return buf.getvalue()
