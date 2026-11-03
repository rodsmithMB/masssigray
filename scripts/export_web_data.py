"""Export the tables the web version needs from xraydb, plus the instrument I0.

    python scripts/export_web_data.py

Writes web/src/data/xraydb.json (Elam cross-section splines, edges, fluorescence
lines, atomic masses and densities for Z = 1..98) and copies the I0 tables from
masssigray/data. Run again after rebuilding the I0 tables.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import xraydb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from masssigray.fluorescence import EDGE_LINE  # noqa: E402
from masssigray.flux import INSTRUMENT_FILE, SMOOTH_FILE  # noqa: E402
from masssigray.xray import EDGE_NAMES  # noqa: E402

OUT = ROOT / "web" / "src" / "data"


def floats(a):
    return [float(x) for x in np.asarray(a, float)]


def main():
    db = xraydb.get_xraydb()
    elements = {}
    for z in range(1, 99):
        sym = xraydb.atomic_symbol(z)
        photo = db.get_cache("photoabsorption", column="element", value=sym)[0]
        scat = db.get_cache("scattering", column="element", value=sym)[0]
        edges = {}
        for name, e in xraydb.xray_edges(sym).items():
            if name in EDGE_NAMES:
                edges[name] = [float(e.energy), float(e.fyield)]
        lines = {}
        for edge, fam in EDGE_LINE.items():
            if edge in edges:
                # fluor_yield at an energy above the edge: (edge yield, mean line energy, probability)
                _, e_line, prob = xraydb.fluor_yield(sym, edge, fam, edges[edge][0] + 1.0)
                if e_line:
                    lines[edge] = [float(e_line), float(prob)]
        elements[sym] = dict(
            z=z, mass=float(xraydb.atomic_mass(sym)), density=float(xraydb.atomic_density(sym)),
            edges=edges, lines=lines,
            photo=[floats(json.loads(photo.log_energy)), floats(json.loads(photo.log_photoabsorption)),
                   floats(json.loads(photo.log_photoabsorption_spline))],
            scatter_lne=floats(json.loads(scat.log_energy)),
            coh=[floats(json.loads(scat.log_coherent_scatter)),
                 floats(json.loads(scat.log_coherent_scatter_spline))],
            incoh=[floats(json.loads(scat.log_incoherent_scatter)),
                   floats(json.loads(scat.log_incoherent_scatter_spline))],
        )
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(source=f"xraydb {xraydb.__version__} (Elam, Ravel and Sieber tables)",
                line_family=EDGE_LINE)
    with open(OUT / "xraydb.json", "w", encoding="utf-8") as fh:
        json.dump(dict(meta=meta, elements=elements), fh, separators=(",", ":"))
    for f in (INSTRUMENT_FILE, SMOOTH_FILE):
        shutil.copy(f, OUT / f.name)
    print(f"wrote {OUT / 'xraydb.json'} ({(OUT / 'xraydb.json').stat().st_size / 1e6:.2f} MB) "
          f"and copied {INSTRUMENT_FILE.name}, {SMOOTH_FILE.name}")


if __name__ == "__main__":
    main()
