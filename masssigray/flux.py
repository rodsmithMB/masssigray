"""Measured incident flux, I0(E).

I0 is the count rate the detector records with no sample in the beam, as a
function of energy. Rates are in *effective counts* per second: detector units
divided by the gain from detector calibration (see detector.py), so that
Poisson statistics apply directly. With an ideal photon counter, effective
counts are photons.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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


def _read_text(source) -> str:
    """Text of a path, a file-like object, or bytes."""
    if isinstance(source, (bytes, bytearray)):
        return source.decode("utf-8", errors="replace")
    if hasattr(source, "read"):
        raw = source.read()
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    with open(source, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def read_table(source) -> pd.DataFrame:
    """Read a whitespace/comma/tab separated numeric table, skipping text headers.

    `source` is a path, a file-like object, or bytes. Comment lines start with #.
    Column names are taken from the last comment line if it has the right
    number of fields, otherwise they are col0, col1, ...
    """
    text = _read_text(source)

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


# ------------------------------------------------------------------ spectrometer exports

_REF_PIXEL = re.compile(r"(?:^|\s)R\s+\d+$")  # "Algn R 1" ... "Algn R 100"; not "R I0"


def read_reference_scan(source) -> tuple[np.ndarray, np.ndarray]:
    """Energy (eV) and reference signal (ADU per point) from a spectrometer CSV export.

    The export has one header row; the reference (no-sample) beam is recorded in
    per-pixel columns "Algn R 1" ... "Algn R n", and the I0 signal is their sum.
    Energy is taken from "Calibrated Energy" (else "Energy"). Rows with missing
    pixels or a zero sum (the scan's start and end markers) are dropped.
    """
    rows = list(csv.reader(io.StringIO(_read_text(source))))
    header = [h.strip() for h in rows[0]]
    pix = [i for i, h in enumerate(header) if _REF_PIXEL.search(h)]
    if not pix:
        raise ValueError("no reference pixel columns ('Algn R 1', ...) in the header")
    ecol = next((header.index(n) for n in ("Calibrated Energy", "Energy") if n in header), None)
    if ecol is None:
        raise ValueError("no 'Calibrated Energy' or 'Energy' column in the header")
    E, total = [], []
    for r in rows[1:]:
        if len(r) <= max(pix) or any(not r[i].strip() for i in pix):
            continue
        s = sum(float(r[i]) for i in pix)
        if s > 0:
            E.append(float(r[ecol]))
            total.append(s)
    if not E:
        raise ValueError("no complete reference rows")
    return np.array(E), np.array(total)


def load_reference_scan(source, dwell: float = 1.0, gain: float = 1.0, label: str = "") -> FluxSpectrum:
    """I0 from a spectrometer export: summed reference pixels / dwell / gain."""
    E, adu = read_reference_scan(source)
    return FluxSpectrum(E, adu / dwell / gain, label=label or "reference scan")


# Built-in I0 of the lab spectrometer, made from the raw exports by
# scripts/build_i0_reference.py: the measured windows (one per element edge) and
# a smoothed curve over their full span.
DATA_DIR = Path(__file__).with_name("data")
INSTRUMENT_FILE = DATA_DIR / "i0_reference.csv"
SMOOTH_FILE = DATA_DIR / "i0_smooth.csv"


def _read_data(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(line for line in fh if not line.startswith("#")))
    return rows[1:]  # skip the header row


@lru_cache(maxsize=None)
def instrument_segments() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{window name: (energy eV, reference ADU per point)} for the measured windows."""
    out: dict[str, tuple[list, list]] = {}
    for name, e, a in _read_data(INSTRUMENT_FILE):
        E, A = out.setdefault(name, ([], []))
        E.append(float(e))
        A.append(float(a))
    segs = {k: (np.array(e), np.array(a)) for k, (e, a) in out.items()}
    return dict(sorted(segs.items(), key=lambda kv: kv[1][0][0]))


@lru_cache(maxsize=None)
def instrument_smooth() -> tuple[np.ndarray, np.ndarray]:
    """(energy eV, reference ADU per point) of the smoothed full-span I0."""
    a = np.array(_read_data(SMOOTH_FILE), float)
    return a[:, 0], a[:, 1]


def smooth_windows(segs, step: float = 10.0, bin_ev: float = 100.0, sigma_ev: float = 300.0):
    """Broad I0 estimate over the full span of the windows.

    log10(I0) is reduced to medians in `bin_ev` bins, linearly interpolated across
    the gaps between windows onto a `step` grid, and smoothed with a Gaussian of
    width `sigma_ev` (normalized at the ends). This keeps the order of magnitude
    and the broad trend; the jumps between separately measured windows average out.
    """
    E = np.concatenate([e for e, _ in segs.values()])
    L = np.log10(np.concatenate([a for _, a in segs.values()]))
    order = np.argsort(E)
    E, L = E[order], L[order]
    idx = np.digitize(E, np.arange(E[0], E[-1] + bin_ev, bin_ev))
    bins = np.unique(idx)
    bc = np.array([np.median(E[idx == i]) for i in bins])
    bl = np.array([np.median(L[idx == i]) for i in bins])
    grid = np.arange(E[0], E[-1] + step / 2, step)
    lg = np.interp(grid, bc, bl)
    half = int(3 * sigma_ev / step)
    w = np.exp(-0.5 * (np.arange(-half, half + 1) * step / sigma_ev) ** 2)
    sm = np.convolve(lg, w, "same") / np.convolve(np.ones_like(lg), w, "same")
    return grid, 10 ** sm


def instrument_window(E0: float) -> str | None:
    """Name of the measured window that contains E0, or None."""
    for name, (E, _) in instrument_segments().items():
        if E[0] <= E0 <= E[-1]:
            return name
    return None


def instrument_flux(window: str | None = None, dwell: float = 1.0, gain: float = 1.0) -> FluxSpectrum:
    """Built-in I0: the smoothed full-span curve (window=None) or one measured window.

    dwell: counting time per point of the reference scans (s).
    gain: detector units per effective count (from the detector calibration).
    """
    if window is None:
        E, adu = instrument_smooth()
        label = "instrument I0, smoothed"
    else:
        E, adu = instrument_segments()[window]
        label = f"instrument I0, {window} window"
    return FluxSpectrum(E, adu / dwell / gain, label=label)


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
