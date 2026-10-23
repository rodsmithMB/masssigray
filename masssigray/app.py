"""Streamlit front end for massSigray.

Run with:  streamlit run masssigray/app.py   (or the `masssigray-app` command)
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xraydb
from plotly.subplots import make_subplots

from masssigray import fluorescence as fluo
from masssigray import simulate
from masssigray.detector import DetectorModel, Scan, calibrate, simulate_scans
from masssigray.flux import FluxSpectrum, example_flux_csv, load_flux, read_table
from masssigray.materials import MATRIX_PRESETS, Material, matrix_preset
from masssigray.scan import ScanPlan
from masssigray.transmission import (CONSTRAINT_MODES, PelletConstraint, TransmissionSetup,
                                     areal_to_thickness_um, evaluate_masses, foil_setup, optimize,
                                     time_for_target)
from masssigray.xray import element_edges

# Categorical slots 1-3 of the validated reference palette (dataviz skill)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "rgba(128,128,128,0.25)"

st.set_page_config(page_title="massSigray", page_icon="🧪", layout="wide")


# ------------------------------------------------------------------ helpers

def fig_layout(fig, height=360, title="", **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=80, b=10),
                      title=dict(text=title, y=0.98, yanchor="top"), hovermode="x unified",
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), **kw)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=MUTED)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=MUTED)
    return fig


def fmt(x, unit="", digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}g}{(' ' + unit) if unit else ''}"


def fmt_time(s):
    if s is None:
        return "unreachable"
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


INV_ANGSTROM = {0: "", 1: " (Å⁻¹)", 2: " (Å⁻²)", 3: " (Å⁻³)"}


def show_warnings(warnings):
    for w in warnings:
        st.warning(w, icon="⚠️")


def parse_material(text, density, label):
    try:
        return Material.from_text(text, density=density or None, name=label)
    except Exception as err:  # bad formula
        st.error(f"{label}: {err}")
        st.stop()


@st.cache_data(show_spinner=False)
def edges_for(element):
    return element_edges(element, 1000.0, 120000.0)


# ------------------------------------------------------------------ sidebar: instrument

st.sidebar.title("Instrument")

with st.sidebar.expander("Incident flux I₀(E)", expanded=True):
    flux_src = st.radio("Source", ["Upload measured I₀", "Constant rate", "Synthetic example"],
                        index=1, key="flux_src")
    flux = None
    if flux_src == "Upload measured I₀":
        up = st.file_uploader("I₀ file (columns: energy, rate or counts)", key="i0_file")
        if up is not None:
            try:
                df = read_table(up.getvalue())
                cols = list(df.columns)
                c1, c2 = st.columns(2)
                ecol = c1.selectbox("Energy column", cols, index=0)
                vcol = c2.selectbox("Value column", cols, index=min(1, len(cols) - 1))
                kind = st.radio("Values are", ["rate", "counts"], horizontal=True)
                dwell = st.number_input("Dwell per point (s)", 0.001, 1e5, 1.0) if kind == "counts" else None
                unit = st.radio("Energy unit", ["auto", "eV", "keV"], horizontal=True)
                cal = st.session_state.get("calibration")
                default_gain = float(cal.gain) if cal else 1.0
                gain = st.number_input("Gain (detector units per effective count)", 1e-6, 1e6,
                                       default_gain, format="%.4g",
                                       help="1 for a photon counter. From the Detector calibration tab "
                                            "for an integrating camera.")
                flux = load_flux(io.BytesIO(up.getvalue()), cols.index(ecol), cols.index(vcol),
                                 kind, dwell, unit, gain, label=up.name)
            except Exception as err:
                st.error(f"Could not read I₀ file: {err}")
    elif flux_src == "Constant rate":
        rate = st.number_input("I₀ rate (effective counts/s per point)", 1.0, 1e12, 1e5, format="%.3g")
        flux = FluxSpectrum.constant(rate)
    else:
        flux = load_flux(io.StringIO(example_flux_csv()), label="synthetic example")
        st.caption("Smooth bremsstrahlung-like curve, 4 to 25 keV, with a dip at 11 keV.")
    scale = st.number_input("Flux scale factor", 1e-6, 1e6, 1.0, format="%.3g",
                            help="Multiply I₀, e.g. for a different tube power.")
    if flux is None:
        flux = FluxSpectrum.constant(1e5)
        st.info("No file yet; using 1e5 counts/s.")
    flux = flux.scaled(scale)

with st.sidebar.expander("Detector noise model", expanded=False):
    cal = st.session_state.get("calibration")
    opts = ["Ideal Poisson", "Custom"] + (["From calibration"] if cal else [])
    dmode = st.radio("Model", opts, index=len(opts) - 1 if cal else 0)
    frame_time = st.number_input("Max exposure per frame (s, 0 = one frame per point)", 0.0, 1e4, 0.0)
    max_rate = st.number_input("Max count rate (effective counts/s, 0 = none)", 0.0, 1e12, 0.0,
                               format="%.3g")
    if dmode == "Custom":
        read_sd = st.number_input("Read noise per frame, ROI sum (effective counts rms)", 0.0, 1e6, 0.0)
        dark = st.number_input("Dark variance rate (effective counts²/s)", 0.0, 1e9, 0.0)
        inst = st.number_input("Point-to-point flux instability (%)", 0.0, 50.0, 0.0) / 100
        det = DetectorModel(read_sd**2, dark, inst, frame_time or None, max_rate or None, "custom")
    elif dmode == "From calibration":
        det = cal.to_detector(frame_time or None, max_rate or None)
        st.caption(f"read var {det.read_var:.3g}, dark {det.dark_rate:.3g}/s, "
                   f"instability {100 * det.instability:.3g} %")
    else:
        det = DetectorModel(frame_time=frame_time or None, max_rate=max_rate or None)


# ------------------------------------------------------------------ main: sample

st.title("massSigray: XAS sample planner")
st.caption("Pellet masses, foil thickness and fluorescence loading from tabulated cross sections "
           "(xraydb/Elam) and your measured I₀.")

c1, c2, c3 = st.columns([2, 1, 1])
sample_txt = c1.text_input("Sample composition", "CuO",
                           help="Formula (Fe2O3, Ni0.5Co0.5(OH)2) or wt% mixture: 'Pt:1, Al2O3' "
                                "(one component may omit its wt% and takes the balance).")
sample_rho = c2.number_input("Sample density (g/cm³, 0 = unknown)", 0.0, 30.0, 6.31)
area = c3.number_input("Pellet area (cm²)", 0.01, 100.0, 1.0, format="%.3g",
                       help="Area the sample is pressed over, any shape. A 13 mm round die is 1.33 cm².")
sample = parse_material(sample_txt, sample_rho, "Sample")

c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
elements = sample.elements
heavy = max(elements, key=xraydb.atomic_number)
absorber = c1.selectbox("Absorber", elements, index=elements.index(heavy), key=f"abs_{sample_txt}")
edges = edges_for(absorber)
if not edges:
    st.error(f"No edges for {absorber} in 1-120 keV.")
    st.stop()
edge_names = list(edges)
lo, hi = max(flux.energy[0], 2000.0), min(flux.energy[-1], 40000.0)
usable = [e for e in ("K", "L3", "L2", "L1", "M5") if e in edges and lo <= edges[e] <= hi]
edge = c2.selectbox("Edge", edge_names, format_func=lambda e: f"{e} ({edges[e]:.0f} eV)",
                    index=edge_names.index(usable[0]) if usable else 0, key=f"edge_{absorber}")
E0 = edges[edge]
mat_choice = c3.selectbox("Matrix", list(MATRIX_PRESETS) + ["Custom", "None"], index=1)
if mat_choice == "Custom":
    mt = c4.text_input("Matrix formula", "C6H10O5")
    mr = c4.number_input("Matrix density", 0.01, 30.0, 1.5)
    matrix = parse_material(mt, mr, "Matrix")
elif mat_choice == "None":
    matrix = None
else:
    matrix = matrix_preset(mat_choice)
    c4.caption(f"{matrix.label}, {matrix.density:g} g/cm³")

with st.expander(f"Scan plan around {absorber} {edge} = {E0:.0f} eV"):
    s1, s2, s3, s4 = st.columns(4)
    pre_start = s1.number_input("Pre-edge start (eV rel.)", -2000.0, -10.0, -150.0)
    pre_step = s1.number_input("Pre-edge step (eV)", 0.1, 100.0, 5.0)
    xs = s2.number_input("XANES start (eV rel.)", -500.0, 0.0, -30.0)
    xe = s2.number_input("XANES end (eV rel.)", 1.0, 500.0, 40.0)
    xstep = s2.number_input("XANES step (eV)", 0.05, 20.0, 0.5)
    kmax = s3.number_input("k max (Å⁻¹, ≤ XANES end for none)", 0.0, 25.0, 12.0)
    kstep = s3.number_input("k step (Å⁻¹)", 0.01, 1.0, 0.05)
    dpre = s4.number_input("Dwell pre-edge (s)", 0.001, 1e4, 1.0)
    dx = s4.number_input("Dwell XANES (s)", 0.001, 1e4, 1.0)
    dex = s4.number_input("Dwell EXAFS at start (s)", 0.001, 1e4, 1.0)
    kwt = s3.number_input("EXAFS dwell ∝ kⁿ, n =", 0.0, 4.0, 0.0)
    i0f = s1.number_input("I₀ scan dwell / sample dwell", 0.01, 100.0, 1.0,
                          help="Longer I₀ scans reduce the I₀ noise contribution.")
    scan = ScanPlan(E0, pre_start, pre_step, xs, xe, xstep, kmax, kstep, dpre, dx, dex, kwt, i0f)
    ts, t0 = scan.total_time()
    st.caption(f"{scan.grid()[0].size} points; counting time {ts / 60:.1f} min sample + "
               f"{t0 / 60:.1f} min I₀ (excluding overhead). Range {scan.emin:.0f} to {scan.emax:.0f} eV.")

tab_t, tab_f, tab_foil, tab_cal, tab_about = st.tabs(
    ["Transmission pellet", "Fluorescence", "Foil / film", "Detector calibration", "Method"])


def objective_inputs(prefix, default="edge"):
    c1, c2, c3 = st.columns(3)
    obj = c1.radio("Optimize for", ["edge", "exafs"], index=0 if default == "edge" else 1,
                   key=prefix + "obj", horizontal=True,
                   format_func=lambda o: "Edge-step SNR" if o == "edge" else "EXAFS χ(k) noise")
    kr = c2.slider("k range for χ noise (Å⁻¹)", 0.0, 20.0, (3.0, min(10.0, max(kmax, 4.0))),
                   key=prefix + "kr")
    kw = c3.selectbox("k weight", [0, 1, 2, 3], index=2, key=prefix + "kw")
    return obj, kr, kw


def constraint_inputs(prefix, with_matrix=True):
    c1, c2, c3, c4 = st.columns(4)
    if with_matrix:
        labels = {"min_total_mass": "Minimum total mass", "fixed_total_mass": "Fixed total mass",
                  "min_thickness": "Minimum thickness", "min_matrix_mass": "Minimum matrix mass",
                  "none": "No pellet constraint"}
        mode = c1.selectbox("Pellet rule", CONSTRAINT_MODES, format_func=labels.get, key=prefix + "mode")
        total = c1.number_input("Total mass (mg)", 1.0, 5000.0, 100.0, key=prefix + "tot",
                                disabled=mode not in ("min_total_mass", "fixed_total_mass"))
        mmin = c1.number_input("Min matrix mass (mg)", 0.0, 5000.0, 50.0, key=prefix + "mmin",
                               disabled=mode != "min_matrix_mass",
                               help="Matrix needed to bind the pellet, regardless of sample mass.")
        tmin = c2.number_input("Min thickness (mm)", 0.01, 20.0, 0.5, key=prefix + "tmin",
                               disabled=mode != "min_thickness")
        pack = c2.number_input("Packing fraction", 0.1, 1.0, 0.7, key=prefix + "pack",
                               help="Pellet density / ideal mixture density; for thickness estimates.")
        fmin = c3.number_input("Min matrix wt%", 0.0, 99.0, 0.0, key=prefix + "fmin") / 100
        weigh = c3.number_input("Min weighable mass (mg)", 0.0, 100.0, 2.0, key=prefix + "weigh")
    else:
        mode, total, tmin, pack, fmin, weigh, mmin = "none", 100.0, 0.5, 1.0, 0.0, 0.0, 0.0
    step_cap = c4.number_input("Max edge step Δμt", 0.0, 10.0, 1.5, key=prefix + "step",
                               help="0 disables. Above ~1.5 thickness effects grow.")
    mut_cap = c4.number_input("Max total μt", 0.0, 20.0, 4.0, key=prefix + "mut", help="0 disables.")
    return PelletConstraint(mode, total, tmin, pack, fmin, step_cap or None, mut_cap or None, weigh,
                            min_matrix_mg=mmin)


def plot_optimization(curve, plan, xlabel="Sample mass (mg)", mark=None):
    ms = curve["m_sample"]
    feas = curve["feasible"]
    y = curve["snr"] if curve["objective"] == "edge" else curve["chi"]
    ylab = "Edge-step SNR per point" if curve["objective"] == "edge" else "rms noise of kʷχ(k)"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Scatter(x=ms, y=y, name="all loadings", line=dict(color=MUTED, width=2),
                             hovertemplate="%{y:.3g}"), 1, 1)
    fig.add_trace(go.Scatter(x=ms, y=np.where(feas, y, np.nan), name="allowed",
                             line=dict(color=C1, width=2), hovertemplate="%{y:.3g}"), 1, 1)
    fig.add_trace(go.Scatter(x=ms, y=curve["step"], name="Δμt", line=dict(color=C2, width=2),
                             hovertemplate="%{y:.3g}", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=ms, y=curve["mut_max"], name="max μt", line=dict(color=C3, width=2),
                             hovertemplate="%{y:.3g}", showlegend=False), 3, 1)
    x0 = mark if mark is not None else plan.m_sample_mg
    for r in (1, 2, 3):
        fig.add_vline(x=x0, line=dict(color=C1, dash="dot", width=1), row=r, col=1)
    fig.update_xaxes(type="log", range=[np.log10(x0 / 30), np.log10(ms[-1])])
    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    fig.update_yaxes(title_text=ylab, type="log" if curve["objective"] == "exafs" else "linear", row=1, col=1)
    fig.update_yaxes(title_text="Δμt", row=2, col=1)
    fig.update_yaxes(title_text="max μt", row=3, col=1)
    return fig_layout(fig, height=520, title="Objective and constraints vs loading")


def time_section(time_fn, setup, plan, kw, key):
    with st.expander(f"Counting time for a target rms σ(k^{kw}χ)", expanded=True):
        a, b = st.columns(2)
        default = {0: 0.002, 1: 0.02, 2: 0.1, 3: 1.0}[kw]
        target = a.number_input(f"Target rms σ(k^{kw}χ) per point{INV_ANGSTROM[kw]}", 0.0, 1e6, default,
                                format="%.3g", key=f"{key}target{kw}",
                                help="Compare with the k^w χ amplitude you expect over the k range.")
        overhead = b.number_input("Overhead per point (s)", 0.0, 60.0, 0.0, key=key + "ovh",
                                  help="Motor move and readout; paid once per point in every scan.")
        if target <= 0:
            return
        try:
            te = time_fn(setup, plan.m_sample_mg, plan.m_matrix_mg, target, overhead)
        except ValueError as err:
            st.info(str(err))
            return
        v = st.columns(4)
        v[0].metric("One scan pair (planned dwells)", fmt_time(te.time_one_s),
                    help=f"rms σ = {te.sigma_one:.3g}")
        v[1].metric("Longer dwells", fmt_time(te.time_dwell_s),
                    help=None if te.dwell_factor is None else f"all dwells × {te.dwell_factor:.3g}")
        v[2].metric("Repeated scans", fmt_time(te.time_scans_s), help=f"{te.n_scans} scan pairs averaged")
        v[3].metric("Floor for long dwells", fmt(te.floor),
                    help="Set by source instability, which longer dwells do not average out.")
        msg = (f"One sample + I₀ scan gives rms σ = {te.sigma_one:.3g}. Average **{te.n_scans}** "
               f"scan pairs ({fmt_time(te.time_scans_s)})")
        if te.dwell_factor is None:
            msg += (f"; longer dwells alone cannot reach {target:g} because source instability "
                    f"sets a floor of {te.floor:.3g}, so repeat scans instead.")
        else:
            msg += f", or multiply every dwell by {te.dwell_factor:.3g} ({fmt_time(te.time_dwell_s)})."
        st.caption(msg + " Times include the I₀ scan and the per-point overhead, but not "
                   "sample changes or scan-to-scan drift.")


def spectrum_section(plan, key, kw=2):
    sp = plan.spectra
    with st.expander("Reference spectrum for peak absorbance (optional)"):
        st.caption("Upload a XANES/EXAFS of a similar compound (energy, μ). It is normalized and "
                   "scaled by the predicted edge step, so white-line μt and counts become realistic.")
        ref_up = st.file_uploader("Reference spectrum", key=key + "ref")
        rc1, rc2 = st.columns(2)
        normalized = rc1.checkbox("Already normalized", key=key + "norm")
        shift = rc2.number_input("Energy shift (eV)", -100.0, 100.0, 0.0, key=key + "shift")
    ref = None
    if ref_up is not None:
        try:
            ref = simulate.load_reference(io.BytesIO(ref_up.getvalue()), normalized=normalized)
        except Exception as err:
            st.error(f"Could not read reference: {err}")
    model = simulate.model_mut(sp, E0, ref, shift)
    noisy, sig = simulate.add_noise(model, sp, det, scan.i0_dwell_factor, rng=1)
    peak_i = int(np.nanargmax(model))
    n_peak = sp["rate"][peak_i] * sp["dwell"][peak_i] * np.exp(-model[peak_i])
    m1, m2, m3 = st.columns(3)
    m1.metric("Peak μt", fmt(model[peak_i]), help=f"at {sp['E'][peak_i]:.1f} eV")
    m2.metric("Transmitted counts at peak", fmt(n_peak))
    m3.metric("σ(μt) at peak", fmt(sig[peak_i]))
    if ref is None:
        st.caption("Without a reference the edge is the bare tabulated step (no white line).")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.65, 0.35])
    fig.add_trace(go.Scatter(x=sp["E"], y=noisy, name="simulated measurement", mode="markers",
                             marker=dict(color=C1, size=4), hovertemplate="%{y:.4f}"), 1, 1)
    fig.add_trace(go.Scatter(x=sp["E"], y=model, name="model", line=dict(color=C2, width=2),
                             hovertemplate="%{y:.4f}"), 1, 1)
    fig.add_trace(go.Scatter(x=sp["E"], y=sp["counts_I"], name="I counts", line=dict(color=C1, width=2),
                             hovertemplate="%{y:.3g}"), 2, 1)
    fig.add_trace(go.Scatter(x=sp["E"], y=sp["counts_I0"], name="I₀ counts", line=dict(color=C3, width=2),
                             hovertemplate="%{y:.3g}"), 2, 1)
    fig.update_yaxes(title_text="μt", row=1, col=1)
    fig.update_yaxes(title_text="counts per point", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Energy (eV)", row=2, col=1)
    st.plotly_chart(fig_layout(fig, height=520, title="Predicted spectrum"), width="stretch",
                    key=key + "spec")

    k = sp["k"]
    m = (k >= 2) & (sp["E"] > E0) & np.isfinite(sp["sigma_chi"])
    if m.any():
        fig = go.Figure(go.Scatter(x=k[m], y=sp["sigma_chi"][m] * k[m] ** kw, name=f"σ(k^{kw}χ)",
                                   line=dict(color=C1, width=2), hovertemplate="%{y:.3g}"))
        fig.update_xaxes(title_text="k (Å⁻¹)")
        fig.update_yaxes(title_text=f"σ of k^{kw}χ(k) per point{INV_ANGSTROM[kw]}", type="log")
        st.plotly_chart(fig_layout(fig, height=300, title="EXAFS noise per point"),
                        width="stretch", key=key + "chi")
    out = pd.DataFrame(dict(energy_eV=sp["E"], k=k, dwell_s=sp["dwell"], I0_rate=sp["rate"],
                            mut_model=model, sigma_mut=sig, mut_simulated=noisy,
                            counts_I=sp["rate"] * sp["dwell"] * np.exp(-model),
                            counts_I0=sp["counts_I0"]))
    st.download_button("Download predicted spectrum (CSV)", out.to_csv(index=False), "prediction.csv",
                       key=key + "dl")


def recipe_box(plan):
    lines = [f"Pellet: {area:.3g} cm²",
             f"  {sample.label}: {plan.m_sample_mg:.2f} mg",
             f"  {matrix.name if matrix else 'no matrix'}: {plan.m_matrix_mg:.2f} mg",
             f"  total: {plan.total_mg:.2f} mg  ({plan.sample_wt_pct:.2f} wt% sample)"]
    if plan.thickness_mm:
        lines.append(f"  estimated thickness: {plan.thickness_mm:.2f} mm")
    d = plan.dilution
    if d and d.get("possible"):
        lines += ["", f"Sample mass is too small to weigh: dilute 1:{d['factor']:g} first",
                  f"  stock: {d['stock_sample_mg']:.1f} mg sample + {d['stock_matrix_mg']:.1f} mg matrix, grind well",
                  f"  pellet: {d['stock_for_pellet_mg']:.2f} mg stock + {d['extra_matrix_mg']:.2f} mg matrix"]
    elif d:
        lines += ["", d["message"]]
    st.code("\n".join(lines), language=None)


# ------------------------------------------------------------------ transmission

with tab_t:
    obj, kr, kw = objective_inputs("t_")
    c = constraint_inputs("t_", with_matrix=matrix is not None)
    setup = TransmissionSetup(sample, matrix, area, absorber, edge, flux, det, scan, obj, 50.0, kr, kw)
    plan = optimize(setup, c)

    st.subheader("Optimized pellet")
    m = st.columns(6)
    m[0].metric("Sample", fmt(plan.m_sample_mg, "mg"))
    m[1].metric("Matrix", fmt(plan.m_matrix_mg, "mg"))
    m[2].metric("Edge step Δμt", fmt(plan.edge_step))
    m[3].metric("μt above edge", fmt(plan.mut_above), help=f"below edge: {plan.mut_below:.3g}")
    m[4].metric("Edge SNR per point", fmt(plan.snr_edge))
    m[5].metric(f"rms σ(k^{kw}χ)", fmt(plan.chi_noise_rms))
    st.caption(f"Limited by: **{plan.binding}**. Absorber mass {plan.absorber_mass_mg:.3g} mg. "
               f"Max μt in scan {plan.mut_max:.3g}.")
    recipe_box(plan)
    show_warnings(plan.warnings)
    spectrum_section(plan, "t_", kw)
    time_section(time_for_target, setup, plan, kw, "t_")

    with st.expander("Check masses you choose or actually weighed"):
        u1, u2 = st.columns(2)
        um_s = u1.number_input("Sample (mg)", 0.0, 5000.0, float(round(plan.m_sample_mg, 2)), key=f"u_ms_{sample_txt}_{absorber}_{edge}")
        um_m = u2.number_input("Matrix (mg)", 0.0, 5000.0, float(round(plan.m_matrix_mg, 2)), key=f"u_mm_{sample_txt}_{absorber}_{edge}")
        if um_s > 0:
            up = evaluate_masses(setup, um_s, um_m, c)
            v = st.columns(5)
            v[0].metric("Edge step", fmt(up.edge_step), delta=fmt(up.edge_step - plan.edge_step))
            v[1].metric("μt above", fmt(up.mut_above))
            v[2].metric("Edge SNR", fmt(up.snr_edge), delta=fmt(up.snr_edge - plan.snr_edge))
            v[3].metric("rms σ(χ)", fmt(up.chi_noise_rms))
            v[4].metric("Thickness", fmt(up.thickness_mm, "mm"))
            show_warnings(up.warnings)

    st.plotly_chart(plot_optimization(plan.curve, plan), width="stretch", key="t_opt")


# ------------------------------------------------------------------ fluorescence

with tab_f:
    st.caption("For dilute samples. Signal and self-absorption for a uniform pellet; the planner "
               "searches sample and matrix mass together (extra matrix reduces self-absorption).")
    g1, g2, g3, g4 = st.columns(4)
    th_in = g1.number_input("Incidence angle (deg from surface)", 1.0, 90.0, 45.0)
    th_out = g1.number_input("Exit angle (deg from surface)", 1.0, 90.0, 45.0)
    omega = g2.number_input("Solid angle Ω/4π", 1e-6, 0.5, 0.01, format="%.4g")
    eff = g2.number_input("Detector efficiency", 0.0, 1.0, 1.0)
    fscale = g3.number_input("Incident rate / I₀ file rate", 1e-6, 1e6, 1.0, format="%.3g",
                             help="If the transmission camera sees only part of the beam.")
    bg = g3.number_input("Scatter background (counts per incident photon)", 0.0, 1.0, 0.0, format="%.3g")
    rmin = g4.number_input("Min amplitude retention", 0.0, 1.0, 0.9)
    mmax = g4.number_input("Max pellet mass (mg)", 1.0, 5000.0, 300.0)
    obj_f, kr_f, kw_f = objective_inputs("f_", default="exafs")
    cf = constraint_inputs("f_", with_matrix=matrix is not None)
    fs = fluo.FluorescenceSetup(sample, matrix, area, absorber, edge, flux, scan, th_in, th_out, omega,
                                eff, fscale, bg, det, obj_f, 50.0, kr_f, kw_f, rmin, mmax)
    try:
        fp = fluo.optimize(fs, cf)
    except ValueError as err:
        st.error(str(err))
        fp = None
    if fp is not None:
        m = st.columns(6)
        m[0].metric("Sample", fmt(fp.m_sample_mg, "mg"))
        m[1].metric("Matrix", fmt(fp.m_matrix_mg, "mg"))
        m[2].metric("Retention (min over k)", fmt(fp.retention_min_k))
        m[3].metric("Fluorescence rate", fmt(fp.fluo_rate_ref, "/s"))
        m[4].metric("Edge SNR per point", fmt(fp.snr_edge))
        m[5].metric(f"rms σ(k^{kw_f}χ)", fmt(fp.chi_noise_rms))
        st.caption(f"Line energy {fp.line_energy:.0f} eV. Limited by: **{fp.binding}**.")
        recipe_box(fp)
        show_warnings(fp.warnings)
        time_section(fluo.time_for_target, fs, fp, kw_f, "f_")

        cv = fp.curve
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=cv["m_sample"], y=cv["best_score"], name="best allowed score",
                                 line=dict(color=C1, width=2), hovertemplate="%{y:.3g}"), 1, 1)
        fig.add_trace(go.Scatter(x=cv["m_sample"], y=cv["retention"], name="retention (least matrix)",
                                 line=dict(color=C2, width=2), hovertemplate="%{y:.3f}"), 2, 1)
        fig.add_hline(y=rmin, line=dict(color=C2, dash="dot", width=1), row=2, col=1)
        for r in (1, 2):
            fig.add_vline(x=fp.m_sample_mg, line=dict(color=C1, dash="dot", width=1), row=r, col=1)
        fig.update_xaxes(type="log", title_text="Sample mass (mg)", row=2, col=1)
        fig.update_yaxes(title_text="SNR" if obj_f == "edge" else "1 / rms σ(χ)", row=1, col=1)
        fig.update_yaxes(title_text="retention", row=2, col=1)
        st.plotly_chart(fig_layout(fig, height=460, title="Fluorescence score and self-absorption"),
                        width="stretch", key="f_opt")

        sp = fp.spectra
        mk = (sp["k"] >= 2) & (sp["E"] > E0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        fig.add_trace(go.Scatter(x=sp["k"][mk], y=sp["R"][mk], name="retention",
                                 line=dict(color=C2, width=2), hovertemplate="%{y:.3f}"), 1, 1)
        fig.add_trace(go.Scatter(x=sp["k"][mk], y=sp["sigma_chi"][mk] * sp["k"][mk] ** kw_f,
                                 name=f"σ(k^{kw_f}χ)",
                                 line=dict(color=C1, width=2), hovertemplate="%{y:.3g}"), 2, 1)
        fig.update_yaxes(title_text="retention", row=1, col=1)
        fig.update_yaxes(title_text=f"σ(k^{kw_f}χ) per point", type="log", row=2, col=1)
        fig.update_xaxes(title_text="k (Å⁻¹)", row=2, col=1)
        st.plotly_chart(fig_layout(fig, height=420, title="Self-absorption and noise vs k"),
                        width="stretch", key="f_k")


# ------------------------------------------------------------------ foil / film

with tab_foil:
    st.caption("Matrix-free samples of known density: foils, films, membranes, liquids in a cell.")
    f1, f2 = st.columns(2)
    foil_txt = f1.text_input("Foil/film composition", absorber, key=f"foil_txt_{absorber}")
    try:
        default_rho = float(xraydb.atomic_density(foil_txt))
    except Exception:
        default_rho = 1.0
    foil_rho = f2.number_input("Density (g/cm³)", 0.001, 30.0, default_rho, key=f"foil_rho_{foil_txt}")
    foil_mat = parse_material(foil_txt, foil_rho, "Foil")
    if foil_mat.mass_fraction(absorber) == 0:
        st.info(f"The foil contains no {absorber}; showing its absorption only.")
    obj_o, kr_o, kw_o = objective_inputs("o_")
    co = constraint_inputs("o_", with_matrix=False)
    fsetup = foil_setup(foil_mat, absorber, edge, flux, det, scan, obj_o, k_range=kr_o, k_weight=kw_o)
    fplan = optimize(fsetup, co)
    t_um = areal_to_thickness_um(fplan.m_sample_mg, foil_rho)
    m = st.columns(5)
    m[0].metric("Thickness", fmt(t_um, "µm"))
    m[1].metric("Areal density", fmt(fplan.m_sample_mg, "mg/cm²"))
    m[2].metric("Edge step Δμt", fmt(fplan.edge_step))
    m[3].metric("μt above edge", fmt(fplan.mut_above))
    m[4].metric("Edge SNR per point", fmt(fplan.snr_edge))
    st.caption(f"Limited by: **{fplan.binding}**. On your {area:.3g} cm² area that is "
               f"{fplan.m_sample_mg * area:.3g} mg.")
    show_warnings(fplan.warnings)
    with st.expander("Check a thickness you have"):
        have = st.number_input("Thickness (µm)", 0.0, 1e5, float(round(t_um, 2)), key=f"foil_have_{foil_txt}_{edge}")
        if have > 0:
            fu = evaluate_masses(fsetup, have * 1e-4 * foil_rho * 1e3, 0.0, co)
            v = st.columns(3)
            v[0].metric("Edge step", fmt(fu.edge_step))
            v[1].metric("μt above", fmt(fu.mut_above))
            v[2].metric("Edge SNR", fmt(fu.snr_edge))
            show_warnings(fu.warnings)
    st.plotly_chart(plot_optimization(fplan.curve, fplan, xlabel="Areal density (mg/cm²)"),
                    width="stretch", key="o_opt")
    spectrum_section(fplan, "o_")


# ------------------------------------------------------------------ calibration

with tab_cal:
    st.markdown("""
**Photon-transfer calibration.** Separates read noise, dark current, shot noise (and so the
camera gain) and source instability, so the planner's noise predictions match your instrument.

Protocol, with the ROI and binning you use for measurements and **no sample**:
1. 2 to 3 **dark** scans (shutter closed or tube off), at two or more dwell times.
2. **I₀** scans at 4 to 5 dwell times spanning ≥10× (e.g. 0.5, 1, 2, 5, 10 s), **3+ repeats each**.
3. Optional: a known foil at two dwell times to validate predicted σ(μt).

Values must be raw detector units (ADU) summed over the ROI, one value per energy point.
Whole-scan intensity drift between repeats is removed and reported separately: it only
shifts μt by a constant and does not add point-to-point noise.
""")
    files = st.file_uploader("Calibration scans", accept_multiple_files=True, key="cal_files")
    demo = st.checkbox("Use synthetic demo data instead", value=not files)
    scans = []
    if demo:
        e = np.arange(8000, 10000, 5.0)
        scans = simulate_scans(lambda x: flux(x) / max(scale, 1e-12) if flux_src != "Constant rate"
                               else np.full_like(x, 2e4), e, rng=0)
        st.caption("Demo: gain 2.5 ADU/count, read noise 30 ADU, dark 50 ADU/s, 0.2 % instability, "
                   "1 % drift.")
    elif files:
        cc1, cc2 = st.columns(2)
        ecol = cc1.number_input("Energy column index", 0, 50, 0)
        vcol = cc2.number_input("Value column index", 0, 50, 1)
        meta = pd.DataFrame(dict(file=[f.name for f in files], dwell_s=[1.0] * len(files),
                                 dark=["dark" in f.name.lower() for f in files]))
        meta = st.data_editor(meta, disabled=["file"], hide_index=True, key="cal_meta")
        for f, row in zip(files, meta.itertuples()):
            df = read_table(f.getvalue())
            scans.append(Scan(float(row.dwell_s), df.iloc[:, int(ecol)].to_numpy(),
                              df.iloc[:, int(vcol)].to_numpy(), bool(row.dark)))
    if scans:
        try:
            res = calibrate(scans)
        except ValueError as err:
            st.error(str(err))
            res = None
        if res is not None:
            summ = res.summary()
            st.dataframe(pd.DataFrame({"parameter": list(summ), "value": [f"{v:.4g}" for v in summ.values()]}),
                         hide_index=True, width="stretch")
            p = res.points
            fig = go.Figure()
            colors = [C1, C2, C3]
            for i, t in enumerate(sorted(set(p["t"]))):
                sel = p["t"] == t
                order = np.argsort(p["mean"][sel])
                col = colors[i] if i < 3 else "gray"
                fig.add_trace(go.Scatter(x=p["mean"][sel], y=p["var"][sel], mode="markers",
                                         name=f"{t:g} s", marker=dict(size=8, color=col, opacity=0.5),
                                         hovertemplate="mean %{x:.3g}<br>var %{y:.3g}"))
                fig.add_trace(go.Scatter(x=p["mean"][sel][order],
                                         y=res.predicted_var(p["mean"][sel][order], t), showlegend=False,
                                         line=dict(color=col, width=2), hoverinfo="skip"))
            fig.update_xaxes(type="log", title_text="mean signal (ADU)")
            fig.update_yaxes(type="log", title_text="variance (ADU²)")
            fig_layout(fig, height=420, title="Photon transfer: variance vs signal")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, width="stretch", key="cal_ptc")
            if len(set(p["t"])) > 3:
                st.caption("Dwell times beyond the third share a gray color; see the hover labels.")
            b1, b2 = st.columns(2)
            if b1.button("Use this calibration in the planner"):
                st.session_state["calibration"] = res
                st.success("Saved. Select 'From calibration' in the detector settings, and set the "
                           "I₀ file gain to the fitted value.")
                st.rerun()
            b2.download_button("Download calibration (JSON)",
                               json.dumps({k: float(v) for k, v in summ.items()}, indent=2),
                               "detector_calibration.json")


# ------------------------------------------------------------------ method

with tab_about:
    st.markdown(r"""
### What is calculated

**Cross sections.** Mass attenuation coefficients μ/ρ(E) from the Elam tables (xraydb), summed
over element mass fractions. The edge step of the absorber is its photoabsorption above the
edge minus a log-log extrapolation of the pre-edge, so it decays with energy like a real
atomic background.

**Transmission.** For sample mass $m_s$ and matrix mass $m_m$ on area $A$:
$\mu t(E) = [m_s(\mu/\rho)_s + m_m(\mu/\rho)_m]/A$ and $\Delta\mu t = m_{abs}\,\Delta(\mu/\rho)_{abs}/A$.
Counts are $N_0 = I_0(E)\,t_0$ and $N = I_0(E)\,t\,e^{-\mu t}$, and

$$\sigma^2(\mu t) = \frac{\mathrm{var}(N)}{N^2} + \frac{\mathrm{var}(N_0)}{N_0^2},\qquad
\mathrm{var}(S) = n_f\sigma_r^2 + D t + S + (\varepsilon S)^2$$

in effective counts. The matrix only adds absorption, so the planner uses the least matrix the
pellet rule allows and scans the sample mass for the best edge-step SNR
($\Delta\mu t/\sigma$ at E₀+50 eV) or the lowest rms noise of $k^w\chi(k)$, where
$\sigma_\chi = \sigma(\mu t)/\Delta\mu t(E)$. Caps on Δμt and μt guard against thickness
effects and starved detector counts.

**Fluorescence.** Uniform pellet, angles θ_in/θ_out from the surface:
$I_f/I_0 = \frac{\Omega}{4\pi}\eta Y \frac{a/\sin\theta_{in}}{X}(1-e^{-X})$ with
$X = \tau/\sin\theta_{in} + \tau_f/\sin\theta_{out}$. The EXAFS amplitude retention
$R = \partial\ln I_f/\partial\ln a$ measures self-absorption.

**Counting time.** For a target rms σ(kʷχ), two options are compared at the planned masses.
Averaging N independent scan pairs divides every noise term by √N. Multiplying every dwell by f
reduces shot, dark and read noise roughly as 1/√f, but not source instability, which sets a
floor (≈ √2·ε/Δμt per point in transmission). If the target is below that floor, repeat scans instead. A fixed
overhead per point is charged once per point in every scan.

**Limitations.** Tabulated cross sections have no white line or EXAFS: load a reference
spectrum for peak values. Pellet inhomogeneity, harmonics, and particle-size effects are not
modeled; the edge-step cap is the practical guard.
""")
