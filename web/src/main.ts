/**
 * massSigray web app: XAS sample planner. Layout follows the original
 * Streamlit app; every input change recomputes the visible results.
 */
import "katex/dist/katex.min.css";
import "./style.css";
import { calibrate, CalibrationResult, DetectorModel, simulateScans, type Scan } from "./core/detector";
import * as fluo from "./core/fluorescence";
import {
  exampleFluxCsv, FluxSpectrum, instrumentFlux, instrumentSegments, instrumentWindow, loadFlux,
  loadReferenceScan, readTable, type EnergyUnit, type ValueKind,
} from "./core/flux";
import { Material, MATRIX_PRESETS, matrixPreset } from "./core/materials";
import { arange, fmtFixed, fmtG, linspace, type Vec } from "./core/num";
import { ScanPlan } from "./core/scan";
import { addNoise, loadReference, modelMut, type Reference } from "./core/simulate";
import type { TimeEstimate } from "./core/timing";
import {
  arealToThicknessUm, CONSTRAINT_MODES, evaluateMasses, foilSetup, optimize, pelletConstraint, timeForTarget,
  TransmissionSetup, type ConstraintMode, type Objective, type PelletConstraint, type TransmissionPlan,
} from "./core/transmission";
import { EOfK, elementEdges, kOfE } from "./core/xray";
import { atomicDensity, atomicNumber, DATA_SOURCE } from "./core/xraydb";
import { methodHtml } from "./ui/method";
import { C1, C2, C3, Chart, gaps, GRAY, hline, MUTED, stacked, vline, type Trace } from "./ui/plots";
import {
  checkbox, col, downloadButton, errorBox, expander, fileInput, h, infoBox, metrics, numberInput, radio, rich,
  row, select, setChangeHandler, Slot, successBox, textInput, warningBox, type Option,
} from "./ui/widgets";

// ------------------------------------------------------------------ formatting

const fmt = (x: number | null | undefined, unit = "", digits = 3): string =>
  x === null || x === undefined || !Number.isFinite(x) ? "n/a" : `${fmtG(x, digits)}${unit ? ` ${unit}` : ""}`;

function fmtTime(s: number | null): string {
  if (s === null) return "unreachable";
  if (s < 120) return `${fmtFixed(s, 0)} s`;
  if (s < 7200) return `${fmtFixed(s / 60, 1)} min`;
  return `${fmtFixed(s / 3600, 1)} h`;
}

const INV_ANGSTROM: Record<number, string> = { 0: "", 1: " (Å⁻¹)", 2: " (Å⁻²)", 3: " (Å⁻³)" };
const warnings = (list: string[]) => list.map(warningBox);

let calibration: CalibrationResult | null = null;

// ================================================================== sidebar: instrument

type FluxSource = "instrument" | "export" | "table" | "constant" | "synthetic";
const fluxSrc = radio<FluxSource>("Source", [
  { value: "instrument", label: "Instrument (built-in)" },
  { value: "export", label: "Upload spectrometer export" },
  { value: "table", label: "Upload measured I₀" },
  { value: "constant", label: "Constant rate" },
  { value: "synthetic", label: "Synthetic example" },
], "instrument", false);

const segs = instrumentSegments();
const curveSel = select("I₀ curve", [
  { value: "", label: "Smoothed, full span" },
  ...[...segs].map(([w, s]) => ({
    value: w, label: `${w} window only: ${fmtFixed(s.energy[0], 0)}-${fmtFixed(s.energy[s.energy.length - 1], 0)} eV`,
  })),
], "", "The smoothed curve joins all measured windows into one broad estimate. A single window is the raw measurement over its range.");
const exportFile = fileInput("Spectrometer CSV export");
const GAIN_HELP = "Detector units (ADU) per effective count: 1 for a photon counter, else the gain from the Detector calibration tab.";
const refDwell = numberInput("Dwell per point of the reference scan (s)", 1, { min: 0.001, max: 1e5 });
const refGain = numberInput("Gain (ADU per effective count)", 1, { min: 1e-6, max: 1e6, help: GAIN_HELP, digits: 4 });

const tableFile = fileInput("I₀ file (columns: energy, rate or counts)");
const tEcol = select<string>("Energy column", [{ value: "0", label: "col0" }]);
const tVcol = select<string>("Value column", [{ value: "1", label: "col1" }]);
const tKind = radio<ValueKind>("Values are", [{ value: "rate", label: "rate" }, { value: "counts", label: "counts" }], "rate");
const tDwell = numberInput("Dwell per point (s)", 1, { min: 0.001, max: 1e5 });
const tUnit = radio<EnergyUnit>("Energy unit", [{ value: "auto", label: "auto" }, { value: "eV", label: "eV" }, { value: "keV", label: "keV" }], "auto");
const tGain = numberInput("Gain (detector units per effective count)", 1, { min: 1e-6, max: 1e6, help: GAIN_HELP, digits: 4 });
const constRate = numberInput("I₀ rate (effective counts/s per point)", 1e5, { min: 1, max: 1e12 });
const fluxScale = numberInput("Flux scale factor", 1, { min: 1e-6, max: 1e6, help: "Multiply I₀, e.g. for a different tube power." });

const instrPanel = h("div", {}, rich("I₀ is the sum of the reference pixels (Algn R 1 … R n) per point."), curveSel.el);
const exportPanel = h("div", {}, rich("I₀ is the sum of the reference pixels (Algn R 1 … R n) per point."), exportFile.el);
const refShared = h("div", {}, refDwell.el, refGain.el);
const tableCols = h("div", {}, row(tEcol.el, tVcol.el), tKind.el, tDwell.el, tUnit.el, tGain.el);
const tablePanel = h("div", {}, tableFile.el, tableCols);
const constPanel = h("div", {}, constRate.el);
const synthPanel = h("div", {}, rich("Smooth bremsstrahlung-like curve, 4 to 25 keV, with a dip at 11 keV."));
const fluxMsg = new Slot();
const i0Chart = new Chart(230, false);
const i0Caption = new Slot();

type DetMode = "ideal" | "custom" | "calibrated";
const detModel = radio<DetMode>("Model", [
  { value: "ideal", label: "Ideal Poisson" }, { value: "custom", label: "Custom" },
], "ideal", false);
const frameTime = numberInput("Max exposure per frame (s, 0 = one frame per point)", 0, { min: 0, max: 1e4 });
const maxRate = numberInput("Max count rate (effective counts/s, 0 = none)", 0, { min: 0, max: 1e12 });
const readSd = numberInput("Read noise per frame, ROI sum (effective counts rms)", 0, { min: 0, max: 1e6 });
const darkRate = numberInput("Dark variance rate (effective counts²/s)", 0, { min: 0, max: 1e9 });
const instab = numberInput("Point-to-point flux instability (%)", 0, { min: 0, max: 50 });
const customPanel = h("div", {}, readSd.el, darkRate.el, instab.el);
const detCaption = new Slot();

const sidebar = h("aside.sidebar", {},
  h("h2", {}, "Instrument"),
  expander("Incident flux I₀(E)", true, fluxSrc.el, instrPanel, exportPanel, refShared, tablePanel, constPanel,
           synthPanel, fluxScale.el, fluxMsg.el, i0Chart.el, i0Caption.el),
  expander("Detector noise model", false, detModel.el, frameTime.el, maxRate.el, customPanel, detCaption.el),
  h("p.caption.footer", {}, `Cross sections: ${DATA_SOURCE}.`));

interface FluxResult {
  flux: FluxSpectrum;
  instr: string | null; // built-in window name, "" for the smoothed curve, null if not built-in
}

function readFlux(): FluxResult {
  const src = fluxSrc.value;
  instrPanel.hidden = src !== "instrument";
  exportPanel.hidden = src !== "export";
  refShared.hidden = src !== "instrument" && src !== "export";
  tablePanel.hidden = src !== "table";
  constPanel.hidden = src !== "constant";
  synthPanel.hidden = src !== "synthetic";
  const msgs: HTMLElement[] = [];
  let flux: FluxSpectrum | null = null;
  let instr: string | null = null;
  if (src === "instrument") {
    instr = curveSel.value;
    flux = instrumentFlux(instr || null, refDwell.value, refGain.value);
  } else if (src === "export" && exportFile.files.length) {
    const f = exportFile.files[0];
    try {
      flux = loadReferenceScan(f.text, refDwell.value, refGain.value, f.name);
    } catch (err) {
      msgs.push(errorBox(`Could not read export: ${(err as Error).message}`));
    }
  } else if (src === "table") {
    tableCols.hidden = !tableFile.files.length;
    tDwell.el.hidden = tKind.value !== "counts";
    if (tableFile.files.length) {
      const f = tableFile.files[0];
      try {
        const t = readTable(f.text);
        const opts = t.columns.map((c, i) => ({ value: String(i), label: c }));
        tEcol.setOptions(opts, "0");
        tVcol.setOptions(opts, String(Math.min(1, opts.length - 1)));
        flux = loadFlux(f.text, Number(tEcol.value), Number(tVcol.value), tKind.value,
                        tKind.value === "counts" ? tDwell.value : null, tUnit.value, tGain.value, f.name);
      } catch (err) {
        msgs.push(errorBox(`Could not read I₀ file: ${(err as Error).message}`));
      }
    }
  } else if (src === "constant") {
    flux = FluxSpectrum.constant(constRate.value);
  } else if (src === "synthetic") {
    flux = loadFlux(exampleFluxCsv(), 0, 1, "rate", null, "auto", 1, "synthetic example");
  }
  if (!flux) {
    flux = FluxSpectrum.constant(1e5);
    msgs.push(infoBox("No file yet; using 1e5 counts/s."));
  }
  flux = flux.scaled(fluxScale.value);
  fluxMsg.set(...msgs);
  drawI0(flux, src === "instrument" || src === "export");
  return { flux, instr };
}

function drawI0(flux: FluxSpectrum, showWindows: boolean) {
  const traces: Trace[] = [];
  if (showWindows && fluxSrc.value === "instrument") {
    const k = fluxScale.value / refDwell.value / refGain.value;
    [...segs].forEach(([name, s], i) => traces.push({
      x: s.energy, y: s.adu.map((a) => a * k), mode: "lines", name: "measured windows", legendgroup: "w",
      showlegend: i === 0, line: { color: MUTED, width: 1.5 }, hoverinfo: "skip", text: name,
    }));
  }
  traces.push({ x: flux.energy, y: flux.rate, mode: "lines", name: "I₀ used", line: { color: C1, width: 2 },
                hovertemplate: "%{y:.3g}<extra></extra>" });
  i0Chart.draw(traces, {
    xaxis: { title: { text: "Energy (eV)" } }, yaxis: { title: { text: "counts/s" }, rangemode: "tozero" },
    margin: { l: 10, r: 10, t: 10, b: 10 }, hovermode: "closest", showlegend: false,
  });
  const windows = showWindows && fluxSrc.value === "instrument" ? " Gray: the measured windows." : "";
  i0Caption.set(rich(`Blue: ${flux.label}, ${fmtFixed(flux.emin, 0)}-${fmtFixed(flux.emax, 0)} eV.${windows}`));
}

function readDetector(): DetectorModel {
  const opts: Option<DetMode>[] = [{ value: "ideal", label: "Ideal Poisson" }, { value: "custom", label: "Custom" }];
  if (calibration) opts.push({ value: "calibrated", label: "From calibration" });
  detModel.setOptions(opts, "ideal");
  customPanel.hidden = detModel.value !== "custom";
  const ft = frameTime.value || null;
  const mr = maxRate.value || null;
  if (detModel.value === "custom") {
    detCaption.set();
    return new DetectorModel(readSd.value ** 2, darkRate.value, instab.value / 100, ft, mr, "custom");
  }
  if (detModel.value === "calibrated" && calibration) {
    const d = calibration.toDetector(ft, mr);
    detCaption.set(rich(`read var ${fmtG(d.read_var, 3)}, dark ${fmtG(d.dark_rate, 3)}/s, ` +
                        `instability ${fmtG(100 * d.instability, 3)} %`));
    return d;
  }
  detCaption.set();
  return new DetectorModel(0, 0, 0, ft, mr);
}

// ================================================================== main: sample, edge, matrix, scan

const sampleTxt = textInput("Sample composition", "CuO",
  "Formula (Fe2O3, Ni0.5Co0.5(OH)2) or wt% mixture: 'Pt:1, Al2O3' (one component may omit its wt% and takes the balance).");
const sampleRho = numberInput("Sample density (g/cm³, 0 = unknown)", 6.31, { min: 0, max: 30 });
const areaIn = numberInput("Pellet area (cm²)", 1, { min: 0.01, max: 100,
  help: "Area the sample is pressed over, any shape. A 13 mm round die is 1.33 cm²." });
const absorberSel = select<string>("Absorber", [{ value: "Cu", label: "Cu" }]);
const edgeSel = select<string>("Edge", [{ value: "K", label: "K" }]);
const MATRIX_CHOICES = [...Object.keys(MATRIX_PRESETS), "Custom", "None"];
const matrixSel = select<string>("Matrix", MATRIX_CHOICES.map((m) => ({ value: m, label: m })), "Cellulose");
const matrixFormula = textInput("Matrix formula", "C6H10O5");
const matrixRho = numberInput("Matrix density", 1.5, { min: 0.01, max: 30 });
const matrixCustom = h("div", {}, matrixFormula.el, matrixRho.el);
const matrixCaption = new Slot();
const contextMsg = new Slot();

const sPreStart = numberInput("Pre-edge start (eV rel.)", -150, { min: -2000, max: -10 });
const sPreStep = numberInput("Pre-edge step (eV)", 5, { min: 0.1, max: 100 });
const sI0f = numberInput("I₀ scan dwell / sample dwell", 1, { min: 0.01, max: 100,
  help: "Longer I₀ scans reduce the I₀ noise contribution." });
const sXs = numberInput("XANES start (eV rel.)", -30, { min: -500, max: 0 });
const sXe = numberInput("XANES end (eV rel.)", 40, { min: 1, max: 500 });
const sXstep = numberInput("XANES step (eV)", 0.5, { min: 0.05, max: 20 });
const sKmax = numberInput("k max (Å⁻¹, ≤ XANES end for none)", 12, { min: 0, max: 25,
  help: "Defaults to the end of the I₀ data when it is shorter." });
const sKstep = numberInput("k step (Å⁻¹)", 0.05, { min: 0.01, max: 1 });
const sKwt = numberInput("EXAFS dwell ∝ kⁿ, n =", 0, { min: 0, max: 4 });
const sDpre = numberInput("Dwell pre-edge (s)", 1, { min: 0.001, max: 1e4 });
const sDx = numberInput("Dwell XANES (s)", 1, { min: 0.001, max: 1e4 });
const sDex = numberInput("Dwell EXAFS at start (s)", 1, { min: 0.001, max: 1e4 });
const scanCaption = new Slot();
const scanExpander = expander("Scan plan", false,
  row(col(sPreStart.el, sPreStep.el, sI0f.el), col(sXs.el, sXe.el, sXstep.el), col(sKmax.el, sKstep.el, sKwt.el),
      col(sDpre.el, sDx.el, sDex.el)),
  scanCaption.el);

interface Context {
  flux: FluxSpectrum;
  fluxSrc: FluxSource;
  det: DetectorModel;
  sample: Material;
  sampleTxt: string;
  area: number;
  absorber: string;
  edge: string;
  E0: number;
  matrix: Material | null;
  scan: ScanPlan;
  kmax: number;
}

const edgesFor = (el: string) => elementEdges(el, 1000, 120000);

function parseMaterial(text: string, density: number, label: string): Material {
  try {
    return Material.fromText(text, density || null, label);
  } catch (err) {
    throw new Error(`${label}: ${(err as Error).message}`);
  }
}

function readContext(): Context {
  const { flux, instr } = readFlux();
  const det = readDetector();
  const sample = parseMaterial(sampleTxt.value, sampleRho.value, "Sample");
  const elements = sample.elements;
  const heavy = elements.reduce((a, b) => (atomicNumber(b) > atomicNumber(a) ? b : a));
  absorberSel.setOptions(elements.map((e) => ({ value: e, label: e })), heavy);
  absorberSel.resetOn(sampleTxt.value, heavy);
  const absorber = absorberSel.value;
  const edges = edgesFor(absorber);
  const names = Object.keys(edges);
  if (!names.length) throw new Error(`No edges for ${absorber} in 1-120 keV.`);
  const lo = Math.max(flux.emin, 2000);
  const hi = Math.min(flux.emax, 40000);
  const usable = ["K", "L3", "L2", "L1", "M5"].filter((e) => e in edges && lo <= edges[e] && edges[e] <= hi);
  const def = usable[0] ?? names[0];
  edgeSel.setOptions(names.map((e) => ({ value: e, label: `${e} (${fmtFixed(edges[e], 0)} eV)` })), def);
  edgeSel.resetOn(`${sampleTxt.value}/${absorber}`, def);
  const edge = edgeSel.value;
  const E0 = edges[edge];

  const ctxMsgs: (HTMLElement | null)[] = [];
  if (instr !== null) {
    if (instr && instrumentWindow(E0) !== instr) {
      ctxMsgs.push(warningBox(`The ${instr} I₀ window does not contain the ${absorber} ${edge} edge (${fmtFixed(E0, 0)} eV).`));
    }
    const kEnd = flux.emax > E0 && flux.emax < EOfK(20, E0) ? ` (k up to ${fmtFixed(kOfE(flux.emax, E0), 1)} Å⁻¹).` : ".";
    ctxMsgs.push(rich(`I₀: ${flux.label}, ${fmtFixed(flux.emin, 0)}-${fmtFixed(flux.emax, 0)} eV${kEnd}`));
  }
  contextMsg.set(...ctxMsgs);

  const mc = matrixSel.value;
  matrixCustom.hidden = mc !== "Custom";
  let matrix: Material | null = null;
  if (mc === "Custom") {
    matrix = parseMaterial(matrixFormula.value, matrixRho.value, "Matrix");
    matrixCaption.set();
  } else if (mc !== "None") {
    matrix = matrixPreset(mc);
    matrixCaption.set(rich(`${matrix.label}, ${fmtG(matrix.density!)} g/cm³`));
  } else {
    matrixCaption.set();
  }

  const kFlux = flux.emax > E0 ? kOfE(flux.emax, E0) : 0;
  const kmaxDefault = instr === null ? 12 : Math.round(Math.min(12, Math.max(kFlux - 0.05, 0)) * 10) / 10;
  sKmax.follow(`${flux.label}/${E0}`, kmaxDefault);
  const scan = new ScanPlan(E0, {
    pre_start: sPreStart.value, pre_step: sPreStep.value, xanes_start: sXs.value, xanes_end: sXe.value,
    xanes_step: sXstep.value, kmax: sKmax.value, kstep: sKstep.value, dwell_pre: sDpre.value,
    dwell_xanes: sDx.value, dwell_exafs: sDex.value, k_time_weight: sKwt.value, i0_dwell_factor: sI0f.value,
  });
  const [ts, t0] = scan.totalTime();
  scanExpander.setTitle(`Scan plan around ${absorber} ${edge} = ${fmtFixed(E0, 0)} eV`);
  scanCaption.set(rich(`${scan.grid().E.length} points; counting time ${fmtFixed(ts / 60, 1)} min sample + ` +
                       `${fmtFixed(t0 / 60, 1)} min I₀ (excluding overhead). Range ${fmtFixed(scan.emin, 0)} to ` +
                       `${fmtFixed(scan.emax, 0)} eV.`));
  return { flux, fluxSrc: fluxSrc.value, det, sample, sampleTxt: sampleTxt.value, area: areaIn.value, absorber,
           edge, E0, matrix, scan, kmax: sKmax.value };
}

// ================================================================== shared tab sections

interface ObjectiveInputs {
  el: HTMLElement;
  read(kmax: number): { obj: Objective; kr: [number, number]; kw: number };
}

function objectiveInputs(defaultObj: Objective): ObjectiveInputs {
  const obj = radio<Objective>("Optimize for", [
    { value: "edge", label: "Edge-step SNR" }, { value: "exafs", label: "EXAFS χ(k) noise" },
  ], defaultObj);
  const kLo = numberInput("k range for χ noise: from (Å⁻¹)", 3, { min: 0, max: 20 });
  const kHi = numberInput("to (Å⁻¹)", 10, { min: 0, max: 20 });
  const kw = select<string>("k weight", ["0", "1", "2", "3"].map((v) => ({ value: v, label: v })), "2");
  return {
    el: row(obj.el, row(kLo.el, kHi.el), kw.el),
    read(kmax) {
      kHi.follow("", Math.min(10, Math.max(kmax, 4)));
      return { obj: obj.value, kr: [kLo.value, kHi.value], kw: Number(kw.value) };
    },
  };
}

interface ConstraintInputs {
  el: HTMLElement;
  read(withMatrix: boolean): PelletConstraint;
}

const MODE_LABELS: Record<ConstraintMode, string> = {
  min_total_mass: "Minimum total mass", fixed_total_mass: "Fixed total mass", min_thickness: "Minimum thickness",
  min_matrix_mass: "Minimum matrix mass", none: "No pellet constraint",
};

function constraintInputs(allowMatrix = true): ConstraintInputs {
  const mode = select<ConstraintMode>("Pellet rule", CONSTRAINT_MODES.map((m) => ({ value: m, label: MODE_LABELS[m] })));
  const total = numberInput("Total mass (mg)", 100, { min: 1, max: 5000 });
  const mmin = numberInput("Min matrix mass (mg)", 50, { min: 0, max: 5000,
    help: "Matrix needed to bind the pellet, regardless of sample mass." });
  const tmin = numberInput("Min thickness (mm)", 0.5, { min: 0.01, max: 20 });
  const pack = numberInput("Packing fraction", 0.7, { min: 0.1, max: 1,
    help: "Pellet density / ideal mixture density; for thickness estimates." });
  const fmin = numberInput("Min matrix wt%", 0, { min: 0, max: 99 });
  const weigh = numberInput("Min weighable mass (mg)", 2, { min: 0, max: 100 });
  const stepCap = numberInput("Max edge step Δμt", 1.5, { min: 0, max: 10, help: "0 disables. Above ~1.5 thickness effects grow." });
  const mutCap = numberInput("Max total μt", 4, { min: 0, max: 20, help: "0 disables." });
  const matrixPart = [col(mode.el, total.el, mmin.el), col(tmin.el, pack.el), col(fmin.el, weigh.el)];
  const el = allowMatrix ? row(...matrixPart, col(stepCap.el, mutCap.el)) : row(col(stepCap.el, mutCap.el));
  return {
    el,
    read(withMatrix) {
      matrixPart.forEach((p) => p.classList.toggle("muted-block", !withMatrix));
      const m = mode.value;
      total.disabled = !withMatrix || (m !== "min_total_mass" && m !== "fixed_total_mass");
      mmin.disabled = !withMatrix || m !== "min_matrix_mass";
      tmin.disabled = !withMatrix || m !== "min_thickness";
      for (const w of [pack, fmin, weigh]) w.disabled = !withMatrix;
      mode.el.classList.toggle("disabled", !withMatrix);
      const caps = { max_edge_step: stepCap.value || null, max_mut: mutCap.value || null };
      if (!withMatrix || !allowMatrix) {
        return pelletConstraint({ mode: "none", total_mass_mg: 100, min_thickness_mm: 0.5, packing_fraction: 1,
                                  min_matrix_fraction: 0, min_weighable_mg: 0, min_matrix_mg: 0, ...caps });
      }
      return pelletConstraint({ mode: m, total_mass_mg: total.value, min_thickness_mm: tmin.value,
                                packing_fraction: pack.value, min_matrix_fraction: fmin.value / 100,
                                min_weighable_mg: weigh.value, min_matrix_mg: mmin.value, ...caps });
    },
  };
}

function optimizationChart(chart: Chart, curve: NonNullable<TransmissionPlan["curve"]>, x0: number,
                           xlabel = "Sample mass (mg)") {
  const ms = curve.m_sample;
  const y = curve.objective === "edge" ? curve.snr : curve.chi;
  const ylab = curve.objective === "edge" ? "Edge-step SNR per point" : "rms noise of kʷχ(k)";
  const d = stacked([0.5, 0.25, 0.25], 0.05);
  chart.draw([
    { x: ms, y: gaps(y), name: "all loadings", mode: "lines", line: { color: MUTED, width: 2 }, hovertemplate: "%{y:.3g}" },
    { x: ms, y: gaps(y.map((v, i) => (curve.feasible[i] ? v : NaN))), name: "allowed", mode: "lines",
      line: { color: C1, width: 2 }, hovertemplate: "%{y:.3g}" },
    { x: ms, y: gaps(curve.step), name: "Δμt", mode: "lines", yaxis: "y2", line: { color: C2, width: 2 },
      hovertemplate: "%{y:.3g}", showlegend: false },
    { x: ms, y: gaps(curve.mut_max), name: "max μt", mode: "lines", yaxis: "y3", line: { color: C3, width: 2 },
      hovertemplate: "%{y:.3g}", showlegend: false },
  ], {
    xaxis: { type: "log", range: [Math.log10(x0 / 30), Math.log10(ms[ms.length - 1])], anchor: "y3",
             title: { text: xlabel } },
    yaxis: { domain: d[0], title: { text: ylab }, type: curve.objective === "exafs" ? "log" : "linear" },
    yaxis2: { domain: d[1], title: { text: "Δμt" } },
    yaxis3: { domain: d[2], title: { text: "max μt" } },
    shapes: [vline(x0)],
  }, "Objective and constraints vs loading");
}

interface TimeSection {
  el: HTMLElement;
  render(fn: () => (target: number, overhead: number) => TimeEstimate, kw: number): void;
}

function timeSection(): TimeSection {
  const target = numberInput("Target rms σ per point", 0.1, { min: 0, max: 1e6,
    help: "Compare with the k^w χ amplitude you expect over the k range." });
  const overhead = numberInput("Overhead per point (s)", 0, { min: 0, max: 60,
    help: "Motor move and readout; paid once per point in every scan." });
  const out = new Slot();
  const exp = expander("Counting time for a target rms σ(kʷχ)", true, row(target.el, overhead.el), out.el);
  return {
    el: exp,
    render(fn, kw) {
      exp.setTitle(`Counting time for a target rms σ(k^${kw}χ)`);
      (target.el.querySelector(".label") as HTMLElement).firstChild!.textContent =
        `Target rms σ(k^${kw}χ) per point${INV_ANGSTROM[kw]}`;
      target.follow(String(kw), { 0: 0.002, 1: 0.02, 2: 0.1, 3: 1.0 }[kw] ?? 0.1);
      const tgt = target.value;
      if (tgt <= 0) {
        out.set();
        return;
      }
      let te: TimeEstimate;
      try {
        te = fn()(tgt, overhead.value);
      } catch (err) {
        out.set(infoBox((err as Error).message));
        return;
      }
      let msg = `One sample + I₀ scan gives rms σ = ${fmtG(te.sigma_one, 3)}. Average **${te.n_scans}** ` +
                `scan pairs (${fmtTime(te.time_scans_s)})`;
      if (te.dwell_factor === null) {
        msg += `; longer dwells alone cannot reach ${fmtG(tgt)} because source instability sets a floor of ` +
               `${fmtG(te.floor, 3)}, so repeat scans instead.`;
      } else {
        msg += `, or multiply every dwell by ${fmtG(te.dwell_factor, 3)} (${fmtTime(te.time_dwell_s)}).`;
      }
      out.set(
        metrics([
          { label: "One scan pair (planned dwells)", value: fmtTime(te.time_one_s), help: `rms σ = ${fmtG(te.sigma_one, 3)}` },
          { label: "Longer dwells", value: fmtTime(te.time_dwell_s),
            help: te.dwell_factor === null ? undefined : `all dwells × ${fmtG(te.dwell_factor, 3)}` },
          { label: "Repeated scans", value: fmtTime(te.time_scans_s), help: `${te.n_scans} scan pairs averaged` },
          { label: "Floor for long dwells", value: fmt(te.floor),
            help: "Set by source instability, which longer dwells do not average out." },
        ]),
        rich(msg + " Times include the I₀ scan and the per-point overhead, but not sample changes or scan-to-scan drift."),
      );
    },
  };
}

interface SpectrumSection {
  el: HTMLElement;
  render(plan: TransmissionPlan, ctx: Context, kw: number): void;
}

function spectrumSection(): SpectrumSection {
  const refFile = fileInput("Reference spectrum");
  const normalized = checkbox("Already normalized", false);
  const shift = numberInput("Energy shift (eV)", 0, { min: -100, max: 100 });
  const refMsg = new Slot();
  const head = new Slot();
  const specChart = new Chart(520);
  const chiChart = new Chart(300);
  const tail = new Slot();
  const el = h("div", {},
    expander("Reference spectrum for peak absorbance (optional)", false,
      rich("Upload a XANES/EXAFS of a similar compound (energy, μ). It is normalized and scaled by the predicted " +
           "edge step, so white-line μt and counts become realistic."),
      refFile.el, row(normalized.el, shift.el), refMsg.el),
    head.el, specChart.el, chiChart.el, tail.el);
  return {
    el,
    render(plan, ctx, kw) {
      const sp = plan.spectra;
      let ref: Reference | null = null;
      refMsg.set();
      if (refFile.files.length) {
        try {
          ref = loadReference(refFile.files[0].text, 0, 1, "auto", normalized.value);
        } catch (err) {
          refMsg.set(errorBox(`Could not read reference: ${(err as Error).message}`));
        }
      }
      const model = modelMut(sp, ctx.E0, ref, shift.value);
      const { noisy, sigma } = addNoise(model, sp, ctx.det, ctx.scan.i0_dwell_factor, 1);
      let pk = 0;
      model.forEach((v, i) => {
        if (v > model[pk] || Number.isNaN(model[pk])) pk = i;
      });
      const nPeak = sp.rate[pk] * sp.dwell[pk] * Math.exp(-model[pk]);
      head.set(
        metrics([
          { label: "Peak μt", value: fmt(model[pk]), help: `at ${fmtFixed(sp.E[pk], 1)} eV` },
          { label: "Transmitted counts at peak", value: fmt(nPeak) },
          { label: "σ(μt) at peak", value: fmt(sigma[pk]) },
        ]),
        ref ? null : rich("Without a reference the edge is the bare tabulated step (no white line)."),
      );
      const countsI = sp.rate.map((r, i) => r * sp.dwell[i] * Math.exp(-model[i]));
      const d = stacked([0.65, 0.35], 0.06);
      specChart.draw([
        { x: sp.E, y: gaps(noisy), name: "simulated measurement", mode: "markers", marker: { color: C1, size: 4 },
          hovertemplate: "%{y:.4f}" },
        { x: sp.E, y: gaps(model), name: "model", mode: "lines", line: { color: C2, width: 2 }, hovertemplate: "%{y:.4f}" },
        { x: sp.E, y: gaps(sp.counts_I), name: "I counts", mode: "lines", yaxis: "y2", line: { color: C1, width: 2 },
          hovertemplate: "%{y:.3g}" },
        { x: sp.E, y: gaps(sp.counts_I0), name: "I₀ counts", mode: "lines", yaxis: "y2", line: { color: C3, width: 2 },
          hovertemplate: "%{y:.3g}" },
      ], {
        xaxis: { anchor: "y2", title: { text: "Energy (eV)" } },
        yaxis: { domain: d[0], title: { text: "μt" } },
        yaxis2: { domain: d[1], title: { text: "counts per point" }, type: "log" },
      }, "Predicted spectrum");

      const idx = sp.k.flatMap((k, i) => (k >= 2 && sp.E[i] > ctx.E0 && Number.isFinite(sp.sigma_chi[i]) ? [i] : []));
      chiChart.el.hidden = !idx.length;
      if (idx.length) {
        chiChart.draw([{
          x: idx.map((i) => sp.k[i]), y: idx.map((i) => sp.sigma_chi[i] * sp.k[i] ** kw), name: `σ(k^${kw}χ)`,
          mode: "lines", line: { color: C1, width: 2 }, hovertemplate: "%{y:.3g}",
        }], {
          xaxis: { title: { text: "k (Å⁻¹)" } },
          yaxis: { title: { text: `σ of k^${kw}χ(k) per point${INV_ANGSTROM[kw]}` }, type: "log" },
        }, "EXAFS noise per point");
      }
      const csv = ["energy_eV,k,dwell_s,I0_rate,mut_model,sigma_mut,mut_simulated,counts_I,counts_I0",
        ...sp.E.map((e, i) => [e, sp.k[i], sp.dwell[i], sp.rate[i], model[i], sigma[i], noisy[i], countsI[i],
                               sp.counts_I0[i]].map(String).join(","))].join("\n") + "\n";
      tail.set(downloadButton("Download predicted spectrum (CSV)", csv, "prediction.csv"));
    },
  };
}

function recipe(plan: { m_sample_mg: number; m_matrix_mg: number; total_mg: number; sample_wt_pct: number;
                        thickness_mm: number | null; dilution: TransmissionPlan["dilution"] }, ctx: Context): HTMLElement {
  const lines = [`Pellet: ${fmtG(ctx.area, 3)} cm²`,
    `  ${ctx.sample.label}: ${fmtFixed(plan.m_sample_mg, 2)} mg`,
    `  ${ctx.matrix ? ctx.matrix.name : "no matrix"}: ${fmtFixed(plan.m_matrix_mg, 2)} mg`,
    `  total: ${fmtFixed(plan.total_mg, 2)} mg  (${fmtFixed(plan.sample_wt_pct, 2)} wt% sample)`];
  if (plan.thickness_mm) lines.push(`  estimated thickness: ${fmtFixed(plan.thickness_mm, 2)} mm`);
  const d = plan.dilution;
  if (d && d.possible) {
    lines.push("", `Sample mass is too small to weigh: dilute 1:${fmtG(d.factor!)} first`,
      `  stock: ${fmtFixed(d.stock_sample_mg!, 1)} mg sample + ${fmtFixed(d.stock_matrix_mg!, 1)} mg matrix, grind well`,
      `  pellet: ${fmtFixed(d.stock_for_pellet_mg!, 2)} mg stock + ${fmtFixed(d.extra_matrix_mg!, 2)} mg matrix`);
  } else if (d) {
    lines.push("", d.message!);
  }
  return h("pre.recipe", {}, lines.join("\n"));
}

const signed = (x: number) => (Number.isFinite(x) ? (x < 0 ? `-${fmtG(-x, 3)}` : fmtG(x, 3)) : undefined);

// ================================================================== transmission tab

const tObj = objectiveInputs("edge");
const tCon = constraintInputs();
const tHead = new Slot();
const tSpec = spectrumSection();
const tTime = timeSection();
const tUms = numberInput("Sample (mg)", 0, { min: 0, max: 5000 });
const tUmm = numberInput("Matrix (mg)", 0, { min: 0, max: 5000 });
const tUser = new Slot();
const tOptChart = new Chart(520);
const tabTransmission = h("div", {}, tObj.el, tCon.el, tHead.el, tSpec.el, tTime.el,
  expander("Check masses you choose or actually weighed", false, row(tUms.el, tUmm.el), tUser.el), tOptChart.el);

function renderTransmission(ctx: Context) {
  const { obj, kr, kw } = tObj.read(ctx.kmax);
  const c = tCon.read(ctx.matrix !== null);
  const setup = new TransmissionSetup({ sample: ctx.sample, matrix: ctx.matrix, area_cm2: ctx.area,
    absorber: ctx.absorber, edge: ctx.edge, flux: ctx.flux, detector: ctx.det, scan: ctx.scan, objective: obj,
    k_range: kr, k_weight: kw });
  const plan = optimize(setup, c);
  tHead.set(
    h("h3", {}, "Optimized pellet"),
    metrics([
      { label: "Sample", value: fmt(plan.m_sample_mg, "mg") },
      { label: "Matrix", value: fmt(plan.m_matrix_mg, "mg") },
      { label: "Edge step Δμt", value: fmt(plan.edge_step) },
      { label: "μt above edge", value: fmt(plan.mut_above), help: `below edge: ${fmtG(plan.mut_below, 3)}` },
      { label: "Edge SNR per point", value: fmt(plan.snr_edge) },
      { label: `rms σ(k^${kw}χ)`, value: fmt(plan.chi_noise_rms) },
    ]),
    rich(`Limited by: **${plan.binding}**. Absorber mass ${fmtG(plan.absorber_mass_mg, 3)} mg. ` +
         `Max μt in scan ${fmtG(plan.mut_max, 3)}.`),
    recipe(plan, ctx),
    ...warnings(plan.warnings),
  );
  tSpec.render(plan, ctx, kw);
  tTime.render(() => (target, ovh) => timeForTarget(setup, plan.m_sample_mg, plan.m_matrix_mg, target, ovh), kw);
  const key = `${ctx.sampleTxt}/${ctx.absorber}/${ctx.edge}`;
  tUms.follow(key, Math.round(plan.m_sample_mg * 100) / 100);
  tUmm.follow(key, Math.round(plan.m_matrix_mg * 100) / 100);
  if (tUms.value > 0) {
    const up = evaluateMasses(setup, tUms.value, tUmm.value, c);
    tUser.set(metrics([
      { label: "Edge step", value: fmt(up.edge_step), delta: signed(up.edge_step - plan.edge_step) },
      { label: "μt above", value: fmt(up.mut_above) },
      { label: "Edge SNR", value: fmt(up.snr_edge), delta: signed(up.snr_edge - plan.snr_edge) },
      { label: "rms σ(χ)", value: fmt(up.chi_noise_rms) },
      { label: "Thickness", value: fmt(up.thickness_mm, "mm") },
    ]), ...warnings(up.warnings));
  } else {
    tUser.set();
  }
  optimizationChart(tOptChart, plan.curve!, plan.m_sample_mg);
}

// ================================================================== fluorescence tab

const fThIn = numberInput("Incidence angle (deg from surface)", 45, { min: 1, max: 90 });
const fThOut = numberInput("Exit angle (deg from surface)", 45, { min: 1, max: 90 });
const fOmega = numberInput("Solid angle Ω/4π", 0.01, { min: 1e-6, max: 0.5 });
const fEff = numberInput("Detector efficiency", 1, { min: 0, max: 1 });
const fScale = numberInput("Incident rate / I₀ file rate", 1, { min: 1e-6, max: 1e6,
  help: "If the transmission camera sees only part of the beam." });
const fBg = numberInput("Scatter background (counts per incident photon)", 0, { min: 0, max: 1 });
const fRmin = numberInput("Min amplitude retention", 0.9, { min: 0, max: 1 });
const fMmax = numberInput("Max pellet mass (mg)", 300, { min: 1, max: 5000 });
const fObj = objectiveInputs("exafs");
const fCon = constraintInputs();
const fHead = new Slot();
const fTime = timeSection();
const fScoreChart = new Chart(460);
const fKChart = new Chart(420);
const tabFluorescence = h("div", {},
  rich("For dilute samples. Signal and self-absorption for a uniform pellet; the planner searches sample and " +
       "matrix mass together (extra matrix reduces self-absorption)."),
  row(col(fThIn.el, fThOut.el), col(fOmega.el, fEff.el), col(fScale.el, fBg.el), col(fRmin.el, fMmax.el)),
  fObj.el, fCon.el, fHead.el, fTime.el, fScoreChart.el, fKChart.el);

function renderFluorescence(ctx: Context) {
  const { obj, kr, kw } = fObj.read(ctx.kmax);
  const c = fCon.read(ctx.matrix !== null);
  let fp: fluo.FluorescencePlan;
  let fs: fluo.FluorescenceSetup;
  try {
    fs = new fluo.FluorescenceSetup({ sample: ctx.sample, matrix: ctx.matrix, area_cm2: ctx.area,
      absorber: ctx.absorber, edge: ctx.edge, flux: ctx.flux, scan: ctx.scan, theta_in: fThIn.value,
      theta_out: fThOut.value, solid_angle_frac: fOmega.value, det_efficiency: fEff.value, flux_scale: fScale.value,
      background_per_photon: fBg.value, i0_detector: ctx.det, objective: obj, k_range: kr, k_weight: kw,
      min_retention: fRmin.value, max_total_mass_mg: fMmax.value });
    fp = fluo.optimize(fs, c);
  } catch (err) {
    fHead.set(errorBox((err as Error).message));
    for (const el of [fTime.el, fScoreChart.el, fKChart.el]) el.hidden = true;
    return;
  }
  for (const el of [fTime.el, fScoreChart.el, fKChart.el]) el.hidden = false;
  fHead.set(
    metrics([
      { label: "Sample", value: fmt(fp.m_sample_mg, "mg") },
      { label: "Matrix", value: fmt(fp.m_matrix_mg, "mg") },
      { label: "Retention (min over k)", value: fmt(fp.retention_min_k) },
      { label: "Fluorescence rate", value: fmt(fp.fluo_rate_ref, "/s") },
      { label: "Edge SNR per point", value: fmt(fp.snr_edge) },
      { label: `rms σ(k^${kw}χ)`, value: fmt(fp.chi_noise_rms) },
    ]),
    rich(`Line energy ${fmtFixed(fp.line_energy, 0)} eV. Limited by: **${fp.binding}**.`),
    recipe(fp, ctx),
    ...warnings(fp.warnings),
  );
  fTime.render(() => (target, ovh) => fluo.timeForTarget(fs, fp.m_sample_mg, fp.m_matrix_mg, target, ovh), kw);

  const cv = fp.curve;
  const d = stacked([1, 1], 0.06);
  fScoreChart.draw([
    { x: cv.m_sample, y: gaps(cv.best_score), name: "best allowed score", mode: "lines", line: { color: C1, width: 2 },
      hovertemplate: "%{y:.3g}" },
    { x: cv.m_sample, y: gaps(cv.retention), name: "retention (least matrix)", mode: "lines", yaxis: "y2",
      line: { color: C2, width: 2 }, hovertemplate: "%{y:.3f}" },
  ], {
    xaxis: { type: "log", anchor: "y2", title: { text: "Sample mass (mg)" } },
    yaxis: { domain: d[0], title: { text: obj === "edge" ? "SNR" : "1 / rms σ(χ)" } },
    yaxis2: { domain: d[1], title: { text: "retention" } },
    shapes: [vline(fp.m_sample_mg), hline(fRmin.value, "y2")],
  }, "Fluorescence score and self-absorption");

  const sp = fp.spectra;
  const idx = sp.k.flatMap((k, i) => (k >= 2 && sp.E[i] > ctx.E0 ? [i] : []));
  const d2 = stacked([1, 1], 0.08);
  fKChart.draw([
    { x: idx.map((i) => sp.k[i]), y: gaps(idx.map((i) => sp.R[i])), name: "retention", mode: "lines",
      line: { color: C2, width: 2 }, hovertemplate: "%{y:.3f}" },
    { x: idx.map((i) => sp.k[i]), y: gaps(idx.map((i) => sp.sigma_chi[i] * sp.k[i] ** kw)), name: `σ(k^${kw}χ)`,
      mode: "lines", yaxis: "y2", line: { color: C1, width: 2 }, hovertemplate: "%{y:.3g}" },
  ], {
    xaxis: { anchor: "y2", title: { text: "k (Å⁻¹)" } },
    yaxis: { domain: d2[0], title: { text: "retention" } },
    yaxis2: { domain: d2[1], title: { text: `σ(k^${kw}χ) per point` }, type: "log" },
  }, "Self-absorption and noise vs k");
}

// ================================================================== foil tab

const foilTxt = textInput("Foil/film composition", "Cu");
const foilRho = numberInput("Density (g/cm³)", 8.96, { min: 0.001, max: 30 });
const foilMsg = new Slot();
const oObj = objectiveInputs("edge");
const oCon = constraintInputs(false);
const oHead = new Slot();
const oHave = numberInput("Thickness (µm)", 0, { min: 0, max: 1e5 });
const oUser = new Slot();
const oOptChart = new Chart(520);
const oSpec = spectrumSection();
const tabFoil = h("div", {},
  rich("Matrix-free samples of known density: foils, films, membranes, liquids in a cell."),
  row(foilTxt.el, foilRho.el), foilMsg.el, oObj.el, oCon.el, oHead.el,
  expander("Check a thickness you have", false, oHave.el, oUser.el), oOptChart.el, oSpec.el);

function renderFoil(ctx: Context) {
  foilTxt.follow(ctx.absorber, ctx.absorber);
  let defRho = 1;
  try {
    defRho = atomicDensity(foilTxt.value.trim());
  } catch {
    /* not an element symbol */
  }
  foilRho.follow(foilTxt.value, defRho);
  const mat = parseMaterial(foilTxt.value, foilRho.value, "Foil");
  foilMsg.set(mat.massFraction(ctx.absorber) === 0 ? infoBox(`The foil contains no ${ctx.absorber}; showing its absorption only.`) : null);
  const { obj, kr, kw } = oObj.read(ctx.kmax);
  const c = oCon.read(false);
  const setup = foilSetup(mat, ctx.absorber, ctx.edge, ctx.flux, { detector: ctx.det, scan: ctx.scan, objective: obj,
                                                                  k_range: kr, k_weight: kw });
  const plan = optimize(setup, c);
  const tUm = arealToThicknessUm(plan.m_sample_mg, foilRho.value);
  oHead.set(
    metrics([
      { label: "Thickness", value: fmt(tUm, "µm") },
      { label: "Areal density", value: fmt(plan.m_sample_mg, "mg/cm²") },
      { label: "Edge step Δμt", value: fmt(plan.edge_step) },
      { label: "μt above edge", value: fmt(plan.mut_above) },
      { label: "Edge SNR per point", value: fmt(plan.snr_edge) },
    ]),
    rich(`Limited by: **${plan.binding}**. On your ${fmtG(ctx.area, 3)} cm² area that is ` +
         `${fmtG(plan.m_sample_mg * ctx.area, 3)} mg.`),
    ...warnings(plan.warnings),
  );
  oHave.follow(`${foilTxt.value}/${ctx.edge}`, Math.round(tUm * 100) / 100);
  if (oHave.value > 0) {
    const fu = evaluateMasses(setup, oHave.value * 1e-4 * foilRho.value * 1e3, 0, c);
    oUser.set(metrics([
      { label: "Edge step", value: fmt(fu.edge_step) },
      { label: "μt above", value: fmt(fu.mut_above) },
      { label: "Edge SNR", value: fmt(fu.snr_edge) },
    ]), ...warnings(fu.warnings));
  } else {
    oUser.set();
  }
  optimizationChart(oOptChart, plan.curve!, plan.m_sample_mg, "Areal density (mg/cm²)");
  oSpec.render(plan, ctx, 2);
}

// ================================================================== calibration tab

const calFiles = fileInput("Calibration scans", true);
const calDemo = checkbox("Use synthetic demo data instead", true);
const calEcol = numberInput("Energy column index", 0, { min: 0, max: 50, step: 1 });
const calVcol = numberInput("Value column index", 1, { min: 0, max: 50, step: 1 });
const calMetaBox = h("div");
const calFileOpts = h("div", {}, row(calEcol.el, calVcol.el), calMetaBox);
const calOut = new Slot();
const calChart = new Chart(420);
const calNote = new Slot();
const calMeta = new Map<string, { dwell: ReturnType<typeof numberInput>; dark: ReturnType<typeof checkbox> }>();
let calFileKey = "";
let calDemoTouched = false;
calDemo.el.addEventListener("change", () => {
  calDemoTouched = true;
});
let lastCalibration: CalibrationResult | null = null;
const calApply = h("button", { type: "button" }, "Use this calibration in the planner");
const calApplied = new Slot();
calApply.addEventListener("click", () => {
  if (!lastCalibration) return;
  calibration = lastCalibration;
  detModel.setOptions([{ value: "ideal", label: "Ideal Poisson" }, { value: "custom", label: "Custom" },
                       { value: "calibrated", label: "From calibration" }], "calibrated");
  detModel.value = "calibrated";
  refGain.set(calibration.gain);
  tGain.set(calibration.gain);
  calApplied.set(successBox(`Saved. The detector model is now "From calibration", and the I₀ gain is set to ` +
                            `${fmtG(calibration.gain, 4)} ADU per count.`));
  schedule();
});
const calJson = h("span");

const tabCalibration = h("div", {},
  h("div.prose", { html: `
<p><strong>Photon-transfer calibration.</strong> Separates read noise, dark current, shot noise (and so the
camera gain) and source instability, so the planner's noise predictions match your instrument.</p>
<p>Protocol, with the ROI and binning you use for measurements and <strong>no sample</strong>:</p>
<ol>
<li>2 to 3 <strong>dark</strong> scans (shutter closed or tube off), at two or more dwell times.</li>
<li><strong>I₀</strong> scans at 4 to 5 dwell times spanning ≥10× (e.g. 0.5, 1, 2, 5, 10 s), <strong>3+ repeats each</strong>.</li>
<li>Optional: a known foil at two dwell times to validate predicted σ(μt).</li>
</ol>
<p>Values must be raw detector units (ADU) summed over the ROI, one value per energy point.
Whole-scan intensity drift between repeats is removed and reported separately: it only
shifts μt by a constant and does not add point-to-point noise.</p>` }),
  calFiles.el, calDemo.el, calFileOpts, calOut.el, calChart.el, calNote.el, h("div.buttons", {}, calApply, calJson),
  calApplied.el);

function renderCalibration(ctx: Context) {
  const files = calFiles.files;
  if (!calDemoTouched) calDemo.value = !files.length;
  const key = files.map((f) => f.name).join("|");
  if (key !== calFileKey) {
    calFileKey = key;
    calMetaBox.replaceChildren();
    if (files.length) {
      const tbody = h("tbody");
      for (const f of files) {
        if (!calMeta.has(f.name)) {
          calMeta.set(f.name, { dwell: numberInput("", 1, { min: 0.001, max: 1e5 }),
                                dark: checkbox("dark", f.name.toLowerCase().includes("dark")) });
        }
        const m = calMeta.get(f.name)!;
        tbody.append(h("tr", {}, h("td", {}, f.name), h("td", {}, m.dwell.input), h("td", {}, m.dark.el)));
      }
      calMetaBox.append(h("table.meta", {}, h("thead", {}, h("tr", {}, h("th", {}, "file"), h("th", {}, "dwell (s)"),
                                                                 h("th", {}, "dark"))), tbody));
    }
  }
  calFileOpts.hidden = calDemo.value || !files.length;
  let scans: Scan[] = [];
  const notes: (HTMLElement | null)[] = [];
  if (calDemo.value) {
    const e = ctx.fluxSrc === "constant" ? arange(8000, 10000, 5) : linspace(ctx.flux.emin, ctx.flux.emax, 400);
    const scale = Math.max(fluxScale.value, 1e-12);
    const rateFn = ctx.fluxSrc === "constant" ? (x: Vec) => x.map(() => 2e4) : (x: Vec) => ctx.flux.at(x).map((r) => r / scale);
    scans = simulateScans(rateFn, e, { seed: 0 });
    notes.push(rich("Demo: gain 2.5 ADU/count, read noise 30 ADU, dark 50 ADU/s, 0.2 % instability, 1 % drift."));
  } else if (files.length) {
    try {
      scans = files.map((f) => {
        const t = readTable(f.text);
        const m = calMeta.get(f.name)!;
        return { dwell: m.dwell.value, energy: t.column(calEcol.value), values: t.column(calVcol.value), dark: m.dark.value };
      });
    } catch (err) {
      notes.push(errorBox((err as Error).message));
    }
  }
  lastCalibration = null;
  calChart.el.hidden = true;
  calApply.hidden = true;
  calJson.replaceChildren();
  if (scans.length) {
    try {
      lastCalibration = calibrate(scans);
    } catch (err) {
      notes.push(errorBox((err as Error).message));
    }
  }
  const res = lastCalibration;
  if (res) {
    const summ = res.summary();
    notes.push(h("table.summary", {}, h("thead", {}, h("tr", {}, h("th", {}, "parameter"), h("th", {}, "value"))),
      h("tbody", {}, ...Object.entries(summ).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, fmtG(v, 4)))))));
    const p = res.points;
    const times = [...new Set(p.t)].sort((a, b) => a - b);
    const colors = [C1, C2, C3];
    const traces: Trace[] = [];
    times.forEach((t, i) => {
      const sel = p.t.flatMap((tt, j) => (tt === t ? [j] : []));
      const means = sel.map((j) => p.mean[j]);
      const order = means.map((_, j) => j).sort((a, b) => means[a] - means[b]);
      const color = i < 3 ? colors[i] : GRAY;
      traces.push({ x: means, y: sel.map((j) => p.var[j]), mode: "markers", name: `${fmtG(t)} s`,
                    marker: { size: 8, color, opacity: 0.5 }, hovertemplate: "mean %{x:.3g}<br>var %{y:.3g}" });
      const xs = order.map((j) => means[j]);
      traces.push({ x: xs, y: xs.map((m) => res.predictedVar(m, t)), mode: "lines", showlegend: false,
                    line: { color, width: 2 }, hoverinfo: "skip" });
    });
    calChart.el.hidden = false;
    calChart.draw(traces, {
      xaxis: { type: "log", title: { text: "mean signal (ADU)" } },
      yaxis: { type: "log", title: { text: "variance (ADU²)" } }, hovermode: "closest",
    }, "Photon transfer: variance vs signal");
    if (times.length > 3) notes.push(rich("Dwell times beyond the third share a gray color; see the hover labels."));
    calApply.hidden = false;
    calJson.append(downloadButton("Download calibration (JSON)", JSON.stringify(summ, null, 2),
                                  "detector_calibration.json", "application/json"));
  }
  calOut.set(...notes);
  calNote.set();
}

// ================================================================== tabs and layout

const TABS = [
  { id: "transmission", label: "Transmission pellet", el: tabTransmission, render: renderTransmission },
  { id: "fluorescence", label: "Fluorescence", el: tabFluorescence, render: renderFluorescence },
  { id: "foil", label: "Foil / film", el: tabFoil, render: renderFoil },
  { id: "calibration", label: "Detector calibration", el: tabCalibration, render: renderCalibration },
  { id: "method", label: "Method", el: h("div.prose", { html: methodHtml() }), render: null },
] as const;
const hashTab = location.hash.slice(1);
let activeTab: string = TABS.some((t) => t.id === hashTab) ? hashTab : TABS[0].id;
const tabBar = h("nav.tabs", { role: "tablist" });
const tabError = new Slot();
for (const t of TABS) {
  const b = h("button.tab", { type: "button", role: "tab" }, t.label);
  b.addEventListener("click", () => {
    activeTab = t.id;
    history.replaceState(null, "", `#${t.id}`);
    schedule(0);
  });
  tabBar.append(b);
}

const main = h("main.main", {},
  h("h1", {}, "massSigray: XAS sample planner"),
  rich("Pellet masses, foil thickness and fluorescence loading from tabulated cross sections (xraydb/Elam) and your measured I₀."),
  row(h("div.wide", {}, sampleTxt.el), sampleRho.el, areaIn.el),
  row(absorberSel.el, edgeSel.el, h("div.wide", {}, matrixSel.el), h("div", {}, matrixCustom, matrixCaption.el)),
  contextMsg.el, scanExpander, tabBar, tabError.el, ...TABS.map((t) => t.el));

document.querySelector("#app")!.replaceChildren(h("div.layout", {}, sidebar, main));

// ================================================================== recompute

const status = h("div.status", {}, "Calculating…");
document.body.append(status);

function compute() {
  [...tabBar.children].forEach((b, i) => b.classList.toggle("active", TABS[i].id === activeTab));
  TABS.forEach((t) => (t.el.hidden = t.id !== activeTab));
  tabError.set();
  let ctx: Context;
  try {
    ctx = readContext();
  } catch (err) {
    tabError.set(errorBox((err as Error).message));
    TABS.forEach((t) => (t.el.hidden = t.id !== "method" || activeTab !== "method"));
    return;
  }
  const tab = TABS.find((t) => t.id === activeTab)!;
  if (tab.render) {
    try {
      tab.render(ctx);
    } catch (err) {
      console.error(err);
      tabError.set(errorBox((err as Error).message));
    }
  }
}

let timer: number | undefined;
function schedule(delay = 120) {
  status.classList.add("busy");
  clearTimeout(timer);
  timer = window.setTimeout(() => {
    try {
      compute();
    } finally {
      status.classList.remove("busy");
    }
  }, delay);
}
setChangeHandler(() => schedule());
schedule(0);
