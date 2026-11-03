"""Reference results from the frozen Python code, for the web port's tests.

    python scripts/make_golden.py

Writes web/tests/golden.json. Each case holds its input spec (built the same
way on both sides) and the Python results the TypeScript port must reproduce.
"""
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from masssigray import fluorescence as fl  # noqa: E402
from masssigray import simulate  # noqa: E402
from masssigray.detector import DetectorModel, Scan, calibrate, simulate_scans  # noqa: E402
from masssigray.flux import FluxSpectrum, instrument_flux, read_table  # noqa: E402
from masssigray.materials import Material, matrix_preset, molar_mass, parse_formula  # noqa: E402
from masssigray.scan import ScanPlan  # noqa: E402
from masssigray.transmission import (PelletConstraint, TransmissionSetup,  # noqa: E402
                                     dilution_recipe, evaluate_masses, foil_setup, optimize,
                                     time_for_target)
from masssigray.xray import edge_energy, edge_step_mu_rho, element_edges, mu_rho  # noqa: E402

OUT = ROOT / "web" / "tests" / "golden.json"


def num(x):
    """JSON-safe number: NaN/inf become strings the TS side decodes."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def arr(a):
    return [num(x) for x in np.asarray(a, float).ravel()]


# ------------------------------------------------------------------ builders (mirrored in TS)

def material(spec):
    if spec is None:
        return None
    if "preset" in spec:
        return matrix_preset(spec["preset"])
    return Material.from_text(spec["text"], density=spec.get("density"), name=spec.get("name", ""))


def flux(spec):
    if spec["kind"] == "constant":
        return FluxSpectrum.constant(spec["rate"])
    if spec["kind"] == "instrument":
        return instrument_flux(spec.get("window"), spec.get("dwell", 1.0), spec.get("gain", 1.0))
    raise ValueError(spec)


def detector(spec):
    spec = spec or {}
    return DetectorModel(spec.get("read_var", 0.0), spec.get("dark_rate", 0.0),
                         spec.get("instability", 0.0), spec.get("frame_time"), spec.get("max_rate"))


def scan(spec, E0):
    return ScanPlan(E0=E0, **(spec or {}))


def constraint(spec):
    return PelletConstraint(**(spec or {}))


def tsetup(spec):
    E0 = edge_energy(spec["absorber"], spec["edge"])
    return TransmissionSetup(material(spec["sample"]), material(spec.get("matrix")), spec["area"],
                             spec["absorber"], spec["edge"], flux(spec["flux"]),
                             detector(spec.get("detector")), scan(spec.get("scan"), E0),
                             spec.get("objective", "edge"), 50.0, tuple(spec.get("k_range", (3, 10))),
                             spec.get("k_weight", 2))


def fsetup(spec):
    E0 = edge_energy(spec["absorber"], spec["edge"])
    f = spec.get("fluo", {})
    return fl.FluorescenceSetup(material(spec["sample"]), material(spec.get("matrix")), spec["area"],
                                spec["absorber"], spec["edge"], flux(spec["flux"]),
                                scan(spec.get("scan"), E0), f.get("theta_in", 45.0),
                                f.get("theta_out", 45.0), f.get("omega", 0.01), f.get("eff", 1.0),
                                f.get("flux_scale", 1.0), f.get("bg", 0.0),
                                detector(spec.get("detector")), spec.get("objective", "exafs"), 50.0,
                                tuple(spec.get("k_range", (3, 10))), spec.get("k_weight", 2),
                                f.get("min_retention", 0.9), f.get("max_mass", 300.0))


def plan_out(p, curve_keys):
    out = {k: num(getattr(p, k)) for k in ("m_sample_mg", "m_matrix_mg", "edge_step", "mut_below",
                                          "mut_above", "mut_max", "snr_edge", "chi_noise_rms",
                                          "absorber_mass_mg")}
    out["thickness_mm"] = num(p.thickness_mm)
    out["binding"] = p.binding
    out["feasible"] = bool(p.feasible)
    out["warnings"] = p.warnings
    out["dilution"] = {k: (v if isinstance(v, (bool, str)) else num(v)) for k, v in p.dilution.items()} \
        if p.dilution else None
    sp = p.spectra
    out["spectra"] = {k: arr(sp[k]) for k in ("E", "dwell", "mut", "step", "sigma", "sigma_chi",
                                              "counts_I", "counts_I0", "rate")}
    if curve_keys and p.curve:
        out["curve"] = {k: arr(p.curve[k]) for k in curve_keys}
    return out


def time_out(te):
    return {k: (num(v) if not isinstance(v, int) else v) for k, v in te.__dict__.items()}


# ------------------------------------------------------------------ cases

FLAT = {"kind": "constant", "rate": 1e5}
SMOOTH = {"kind": "instrument", "dwell": 2.0}
NOISY = {"read_var": 400.0, "dark_rate": 50.0, "instability": 0.002, "frame_time": 0.5}

T_CASES = [
    dict(name="CuO cellulose min total", sample={"text": "CuO", "density": 6.31},
         matrix={"preset": "Cellulose"}, area=1.327, absorber="Cu", edge="K", flux=FLAT),
    dict(name="CuO smooth I0 exafs k3", sample={"text": "CuO", "density": 6.31},
         matrix={"preset": "Cellulose"}, area=1.327, absorber="Cu", edge="K", flux=SMOOTH,
         objective="exafs", k_weight=3, k_range=[3, 11], detector=NOISY),
    dict(name="Fe2O3 BN fixed total", sample={"text": "Fe2O3", "density": 5.24},
         matrix={"preset": "Boron nitride (BN)"}, area=1.0, absorber="Fe", edge="K", flux=SMOOTH,
         constraint={"mode": "fixed_total_mass", "total_mass_mg": 80, "min_matrix_fraction": 0.2}),
    dict(name="dilute Pt on alumina min thickness", sample={"text": "Pt:1, Al2O3", "density": 3.9},
         matrix={"preset": "Polyethylene (PE)"}, area=1.33, absorber="Pt", edge="L3", flux=SMOOTH,
         constraint={"mode": "min_thickness", "min_thickness_mm": 0.8}),
    dict(name="NiO min matrix mass, custom scan", sample={"text": "NiO", "density": 6.67},
         matrix={"preset": "Cellulose"}, area=0.785, absorber="Ni", edge="K", flux=FLAT,
         constraint={"mode": "min_matrix_mass", "min_matrix_mg": 40},
         scan={"pre_start": -200, "pre_step": 10, "xanes_step": 0.3, "kmax": 14, "kstep": 0.04,
               "dwell_exafs": 2, "k_time_weight": 2, "i0_dwell_factor": 0.5}, objective="exafs"),
    dict(name="V2O5 no matrix no constraint", sample={"text": "V2O5", "density": 3.36}, matrix=None,
         area=1.0, absorber="V", edge="K", flux=SMOOTH, constraint={"mode": "none"},
         detector={"read_var": 100.0, "max_rate": 5e4}),
    dict(name="KBr window flux caps off", sample={"text": "KBr", "density": 2.75},
         matrix={"preset": "Sucrose"}, area=1.327, absorber="Br", edge="K",
         flux={"kind": "instrument", "window": "Br", "dwell": 1.0, "gain": 2.5},
         constraint={"max_edge_step": None, "max_mut": None}, objective="exafs", k_weight=1),
    dict(name="ZnO in graphite matrix with Zn absent sample", sample={"text": "SiO2", "density": 2.2},
         matrix={"preset": "Graphite"}, area=1.0, absorber="Zn", edge="K", flux=FLAT),
]

F_CASES = [
    dict(name="dilute Cu fluo", sample={"text": "Cu:0.5, SiO2", "density": 2.5},
         matrix={"preset": "Cellulose"}, area=1.327, absorber="Cu", edge="K", flux=SMOOTH,
         fluo={"bg": 1e-4, "omega": 0.02}),
    dict(name="concentrated Fe fluo edge", sample={"text": "Fe2O3", "density": 5.24},
         matrix={"preset": "Boron nitride (BN)"}, area=1.0, absorber="Fe", edge="K", flux=FLAT,
         objective="edge", fluo={"theta_in": 30, "theta_out": 60, "min_retention": 0.95}),
    dict(name="Pt L3 fluo no matrix", sample={"text": "Pt:2, Al2O3", "density": 3.9}, matrix=None,
         area=1.0, absorber="Pt", edge="L3", flux=SMOOTH, detector=NOISY,
         fluo={"flux_scale": 0.5, "max_mass": 200}),
    dict(name="fixed total fluo", sample={"text": "NiO", "density": 6.67},
         matrix={"preset": "Cellulose"}, area=1.327, absorber="Ni", edge="K", flux=FLAT,
         constraint={"mode": "fixed_total_mass", "total_mass_mg": 150}, fluo={"min_retention": 0.8}),
]


def main():
    g = {}
    formulas = ["CuO", "Fe2O3", "Ni0.5Co0.5(OH)2", "Mn(SO4)2(H2O)7", "Fe.7Mg.3O", "C6H10O5", "Zn1.e-5Fe3O4",
                "D2O", "Pt"]
    g["formulas"] = [{"formula": f, "comp": parse_formula(f), "molar_mass": molar_mass(f)} for f in formulas]
    mixtures = ["Pt:5, Al2O3:95", "Pt:1, Al2O3", "CuO", "Fe2O3:30; SiO2:50, C"]
    g["mixtures"] = [{"text": t, "fractions": Material.from_text(t).mass_fractions,
                      "elements": Material.from_text(t).elements, "label": Material.from_text(t).label}
                     for t in mixtures]
    E = np.concatenate([np.geomspace(150, 70000, 60), [8978.9, 8979.0, 8979.1, 11564, 11565]])
    g["energies"] = arr(E)
    mats = ["CuO", "Pt:1, Al2O3", "C6H10O5", "PbI2", "U3O8", "Cf"]
    g["mu_rho"] = [{"text": t, "total": arr(mu_rho(Material.from_text(t), E)),
                    "photo": arr(mu_rho(Material.from_text(t), E, kind="photo"))} for t in mats]
    steps = [("Cu", "K"), ("Pt", "L3"), ("Pt", "L1"), ("Fe", "K"), ("U", "L3"), ("Ba", "L2"), ("V", "K")]
    g["edge_step"] = [{"element": el, "edge": ed, "values": arr(edge_step_mu_rho(el, ed, E))}
                      for el, ed in steps]
    g["edges"] = {el: element_edges(el, 1000.0, 120000.0) for el in ("Cu", "Pt", "H", "U", "Kr")}

    g["transmission"] = []
    for spec in T_CASES:
        s = tsetup(spec)
        c = constraint(spec.get("constraint"))
        p = optimize(s, c)
        entry = dict(spec=spec, plan=plan_out(p, ("m_sample", "m_matrix", "snr", "chi", "step",
                                                      "mut_max", "feasible")))
        ev = evaluate_masses(s, p.m_sample_mg * 1.3 + 0.5, p.m_matrix_mg + 5.0, c)
        entry["evaluate"] = dict(m_s=p.m_sample_mg * 1.3 + 0.5, m_m=p.m_matrix_mg + 5.0,
                                 plan=plan_out(ev, ()))
        try:
            target = 0.5 * p.chi_noise_rms if np.isfinite(p.chi_noise_rms) else 0.1
            entry["time"] = dict(target=target, overhead=0.3,
                                 result=time_out(time_for_target(s, p.m_sample_mg, p.m_matrix_mg,
                                                                 target, 0.3)))
        except ValueError as err:
            entry["time"] = dict(target=0.1, overhead=0.3, error=str(err))
        g["transmission"].append(entry)

    foil = foil_setup(Material.from_formula("Cu", 8.96), "Cu", "K", FluxSpectrum.constant(1e5))
    fp = optimize(foil, PelletConstraint(mode="none"))
    g["foil"] = dict(plan=plan_out(fp, ()))

    g["fluorescence"] = []
    for spec in F_CASES:
        s = fsetup(spec)
        c = constraint(spec.get("constraint"))
        p = fl.optimize(s, c)
        out = {k: num(getattr(p, k)) for k in ("m_sample_mg", "m_matrix_mg", "retention_ref",
                                              "retention_min_k", "fluo_rate_ref", "snr_edge",
                                              "chi_noise_rms", "line_energy")}
        out["thickness_mm"] = num(p.thickness_mm)
        out["binding"] = p.binding
        out["warnings"] = p.warnings
        out["curve"] = {k: arr(p.curve[k]) for k in ("m_sample", "best_score", "retention")}
        out["spectra"] = {k: arr(p.spectra[k]) for k in ("E", "R", "counts_f", "sigma_chi", "rate")}
        entry = dict(spec=spec, plan=out)
        target = 0.5 * p.chi_noise_rms if np.isfinite(p.chi_noise_rms) else 0.1
        try:
            entry["time"] = dict(target=target, overhead=0.0,
                                 result=time_out(fl.time_for_target(s, p.m_sample_mg, p.m_matrix_mg,
                                                                    target, 0.0)))
        except ValueError as err:
            entry["time"] = dict(target=target, overhead=0.0, error=str(err))
        g["fluorescence"].append(entry)

    g["dilution"] = [dict(args=a, result=dilution_recipe(*a)) for a in
                     [(0.3, 99.7, 2.0), (1.5, 50, 2.0), (0.01, 0.5, 2.0), (5, 95, 2.0), (0.07, 120, 1.0)]]

    # calibration: fixed synthetic scans, stored so both sides fit identical data
    e = np.arange(8000, 9000, 10.0)
    scans = simulate_scans(lambda x: 2e4 * (1 + 0.3 * np.sin(x / 150)), e, rng=3)
    res = calibrate(scans)
    g["calibration"] = dict(
        scans=[dict(dwell=s.dwell, dark=s.dark, energy=arr(s.energy), values=arr(s.values)) for s in scans],
        summary={k: num(v) for k, v in res.summary().items()})
    one = [s for s in scans if s.dwell == 1 and not s.dark]
    g["calibration_one_dwell"] = {k: num(v) for k, v in calibrate(one).summary().items()}

    Er = np.linspace(8800, 9600, 801)
    mu = 0.1 + 1 / (1 + np.exp(-(Er - 8980) / 2)) + 0.5 * np.exp(-0.5 * ((Er - 8995) / 4) ** 2) \
        + 0.0003 * (Er - 8800)
    txt = "\n".join(f"{a:.3f} {b:.6f}" for a, b in zip(Er, mu))
    e0, norm = simulate.normalize(Er, mu)
    ref = simulate.load_reference(io.StringIO(txt))
    p0 = optimize(tsetup(T_CASES[0]), constraint(None))
    g["simulate"] = dict(text=txt, e0=e0, norm=arr(norm), ref_E=arr(ref[0]), ref_norm=arr(ref[1]),
                         model_shift2=arr(simulate.model_mut(p0.spectra, edge_energy("Cu", "K"), ref, 2.0)))
    tab = read_table(io.StringIO("# energy, rate, extra\n1 2 3\n4,5,6\nbad line\n7\t8\t9\n"))
    g["read_table"] = dict(columns=list(tab.columns), values=tab.to_numpy().tolist())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(g))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
