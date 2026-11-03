/**
 * Transmission planner: optimal sample and matrix masses for a pressed pellet,
 * or optimal thickness for a foil or film.
 *
 * For sample mass m_s and matrix mass m_m pressed over area A:
 *   mu t(E)  = [m_s (mu/rho)_s + m_m (mu/rho)_m] / A
 *   dmu t(E) = [m_s w_s + m_m w_m] dmu/rho_abs(E) / A       (edge step)
 *   sigma(mu t) from counting statistics of the I0 and I scans
 *
 * The matrix only adds absorption, so for a given sample mass the planner uses
 * the least matrix the pellet constraint allows, and scans m_s on a log grid:
 *   "edge":  edge-step SNR at E0 + 50 eV
 *   "exafs": 1 / rms noise of k^w chi(k) over the k range
 */
import { DetectorModel, sigmaMut } from "./detector";
import type { FluxSpectrum } from "./flux";
import type { Material } from "./materials";
import { argmax, fmtFixed, fmtG, geomspace, interp1, max, type Vec } from "./num";
import { ScanPlan } from "./scan";
import { estimate, type ChiModel, type TimeEstimate } from "./timing";
import { edgeEnergy, edgeStepMuRho, interferingEdges, kOfE, muRho } from "./xray";

export const CONSTRAINT_MODES = ["min_total_mass", "fixed_total_mass", "min_thickness", "min_matrix_mass", "none"] as const;
export type ConstraintMode = (typeof CONSTRAINT_MODES)[number];
export type Objective = "edge" | "exafs";

export interface PelletConstraint {
  mode: ConstraintMode;
  total_mass_mg: number; // minimum or fixed total, by mode
  min_thickness_mm: number;
  packing_fraction: number; // pellet density / ideal mixture density
  min_matrix_fraction: number; // matrix wt fraction of the pellet, lower bound
  max_edge_step: number | null; // null or 0 disables
  max_mut: number | null; // total mu t at any scan energy; null or 0 disables
  min_weighable_mg: number;
  min_matrix_mg: number; // matrix needed to bind the pellet, for "min_matrix_mass"
}

export function pelletConstraint(opts: Partial<PelletConstraint> = {}): PelletConstraint {
  const c: PelletConstraint = {
    mode: "min_total_mass", total_mass_mg: 100, min_thickness_mm: 0.5, packing_fraction: 0.7,
    min_matrix_fraction: 0, max_edge_step: 1.5, max_mut: 4, min_weighable_mg: 2, min_matrix_mg: 50,
  };
  for (const [k, v] of Object.entries(opts)) if (v !== undefined) (c as unknown as Record<string, unknown>)[k] = v;
  if (!CONSTRAINT_MODES.includes(c.mode)) throw new Error(`mode must be one of ${CONSTRAINT_MODES.join(", ")}`);
  return c;
}

export interface TransmissionSetupInit {
  sample: Material;
  matrix: Material | null;
  area_cm2: number;
  absorber: string;
  edge: string;
  flux: FluxSpectrum;
  detector?: DetectorModel;
  scan?: ScanPlan | null;
  objective?: Objective;
  E_ref_offset?: number;
  k_range?: [number, number];
  k_weight?: number;
}

export class TransmissionSetup {
  readonly sample: Material;
  readonly matrix: Material | null;
  readonly area_cm2: number;
  readonly absorber: string;
  readonly edge: string;
  readonly flux: FluxSpectrum;
  readonly detector: DetectorModel;
  readonly scan: ScanPlan;
  readonly objective: Objective;
  readonly E_ref_offset: number;
  readonly k_range: [number, number];
  readonly k_weight: number;
  readonly E0: number;

  constructor(o: TransmissionSetupInit) {
    this.sample = o.sample;
    this.matrix = o.matrix;
    this.area_cm2 = o.area_cm2;
    this.absorber = o.absorber;
    this.edge = o.edge;
    this.flux = o.flux;
    this.detector = o.detector ?? new DetectorModel();
    this.E0 = edgeEnergy(o.absorber, o.edge);
    this.scan = o.scan ?? new ScanPlan(this.E0);
    this.objective = o.objective ?? "edge";
    if (this.objective !== "edge" && this.objective !== "exafs") throw new Error("objective must be 'edge' or 'exafs'");
    this.E_ref_offset = o.E_ref_offset ?? 50;
    this.k_range = o.k_range ?? [3, 10];
    this.k_weight = o.k_weight ?? 2;
  }
}

export interface Dilution {
  possible: boolean;
  message?: string;
  factor?: number;
  stock_sample_mg?: number;
  stock_matrix_mg?: number;
  stock_for_pellet_mg?: number;
  extra_matrix_mg?: number;
}

export interface TransmissionSpectra {
  E: Vec; dwell: Vec; k: Vec; mut: Vec; step: Vec; sigma: Vec; rate: Vec;
  counts_I0: Vec; counts_I: Vec; sigma_chi: Vec;
}

export interface TransmissionCurve {
  m_sample: Vec; m_matrix: Vec; score: Vec; snr: Vec; chi: Vec; step: Vec; mut_max: Vec;
  feasible: boolean[]; objective: Objective;
}

export interface TransmissionPlan {
  m_sample_mg: number;
  m_matrix_mg: number;
  thickness_mm: number | null;
  edge_step: number;
  mut_below: number;
  mut_above: number;
  mut_max: number;
  snr_edge: number;
  chi_noise_rms: number;
  binding: string;
  feasible: boolean;
  warnings: string[];
  curve: TransmissionCurve | null;
  spectra: TransmissionSpectra;
  dilution: Dilution | null;
  area_cm2: number;
  absorber_mass_mg: number;
  total_mg: number;
  sample_wt_pct: number;
}

/** Results for one loading. Rows run over the scan points (reference points excluded). */
interface Eval {
  mut: Vec; step: Vec; sigma: Vec;
  step_ref: number; mut_above: number; mut_below: number; mut_max: number; snr: number; chi: number;
}

/** Precomputed cross sections on the scan grid, plus two reference points (E0+50, E0-20). */
export class TransmissionModel implements ChiModel {
  readonly E: Vec;
  readonly dwell: Vec;
  readonly rate: Vec;
  readonly t: Vec;
  readonly nPoints: number;
  private readonly mu_s: Vec;
  private readonly mu_m: Vec;
  private readonly dmu_s: Vec;
  private readonly dmu_m: Vec;
  private readonly kIdx: number[];
  private readonly kw: Vec;

  constructor(readonly s: TransmissionSetup) {
    const { E, t } = s.scan.grid();
    this.E = E;
    this.dwell = t;
    this.nPoints = E.length;
    const Eref = s.E0 + s.E_ref_offset;
    const pts = [...E, Eref, s.E0 - 20];
    this.mu_s = muRho(s.sample, pts);
    this.mu_m = s.matrix ? muRho(s.matrix, pts) : pts.map(() => 0);
    const step = edgeStepMuRho(s.absorber, s.edge, pts);
    const ws = s.sample.massFraction(s.absorber);
    const wm = s.matrix ? s.matrix.massFraction(s.absorber) : 0;
    this.dmu_s = step.map((v) => ws * v);
    this.dmu_m = step.map((v) => wm * v);
    this.rate = s.flux.at(pts);
    const tref = interp1(Eref, E, t);
    this.t = [...t, tref, tref];
    const k = pts.map((e) => kOfE(e, s.E0));
    this.kIdx = [];
    for (let i = 0; i < E.length; i++) {
      if (pts[i] >= s.E0 && k[i] >= s.k_range[0] && k[i] <= s.k_range[1]) this.kIdx.push(i);
    }
    this.kw = this.kIdx.map((i) => k[i] ** s.k_weight);
  }

  get muSampleMax(): number {
    return max(this.mu_s.slice(0, this.nPoints));
  }

  /** Evaluate one loading (masses in mg); dwell times scaled by `factor`. */
  evaluate(mS: number, mM: number, factor = 1, rows = true): Eval {
    const s = this.s;
    const ms = mS * 1e-3;
    const mm = mM * 1e-3;
    const n = this.nPoints;
    const N = n + 2;
    const i0f = s.scan.i0_dwell_factor;
    const mut = new Array<number>(N);
    const step = new Array<number>(N);
    const sig = new Array<number>(N);
    let mutMax = -Infinity;
    for (let i = 0; i < N; i++) {
      mut[i] = (ms * this.mu_s[i] + mm * this.mu_m[i]) / s.area_cm2;
      step[i] = (ms * this.dmu_s[i] + mm * this.dmu_m[i]) / s.area_cm2;
      const t = this.t[i] * factor;
      sig[i] = sigmaMut(this.rate[i], mut[i], t, t * i0f, s.detector);
      if (i < n && (mut[i] > mutMax || Number.isNaN(mut[i]))) mutMax = mut[i];
    }
    let chi = NaN;
    if (this.kIdx.length) {
      let acc = 0;
      this.kIdx.forEach((i, j) => {
        const v = (sig[i] / step[i]) * this.kw[j];
        acc += v * v;
      });
      chi = Math.sqrt(acc / this.kIdx.length);
    }
    return {
      mut: rows ? mut.slice(0, n) : [], step: rows ? step.slice(0, n) : [], sigma: rows ? sig.slice(0, n) : [],
      step_ref: step[n], mut_above: mut[n], mut_below: mut[n + 1], mut_max: mutMax,
      snr: step[n] / sig[n], chi,
    };
  }

  chiRms(mS: number, mM: number, factor: number): number {
    return this.evaluate(mS, mM, factor, false).chi;
  }
}

/** Least matrix mass (mg) satisfying the constraint for sample mass mS (mg); NaN if impossible. */
export function matrixMass(mS: number, setup: { matrix: Material | null; sample: Material; area_cm2: number },
                           c: PelletConstraint): number {
  if (setup.matrix === null || c.mode === "none") return 0;
  const f = Math.min(Math.max(c.min_matrix_fraction, 0), 0.999);
  const byFraction = (f / (1 - f)) * mS;
  switch (c.mode) {
    case "min_total_mass":
      return Math.max(c.total_mass_mg - mS, byFraction);
    case "fixed_total_mass": {
      const mm = c.total_mass_mg - mS;
      return mm >= byFraction - 1e-12 ? mm : NaN;
    }
    case "min_matrix_mass":
      return Math.max(c.min_matrix_mg, byFraction);
    default: {
      const rhoS = setup.sample.density;
      const rhoM = setup.matrix.density;
      if (!(rhoS && rhoM)) throw new Error("min_thickness mode needs sample and matrix densities");
      const volNeeded = (c.packing_fraction * setup.area_cm2 * c.min_thickness_mm) / 10; // cm³
      return Math.max(rhoM * 1e3 * (volNeeded - (mS * 1e-3) / rhoS), byFraction);
    }
  }
}

export function pelletThicknessMm(mS: number, mM: number,
                                  setup: { sample: Material; matrix: Material | null; area_cm2: number },
                                  packing: number): number | null {
  const rhoS = setup.sample.density;
  const rhoM = setup.matrix ? setup.matrix.density : null;
  if (!rhoS || (mM > 0 && !rhoM)) return null;
  const vol = (mS * 1e-3) / rhoS + (mM > 0 ? (mM * 1e-3) / rhoM! : 0);
  return (10 * vol) / (setup.area_cm2 * packing);
}

function niceCeiling(x: number): number {
  const exp = 10 ** Math.floor(Math.log10(x));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * exp >= x - 1e-12) return m * exp;
  return 10 * exp;
}

/** Two-step recipe when the sample mass is too small to weigh directly. */
export function dilutionRecipe(mS: number, mM: number, minWeigh: number): Dilution | null {
  if (mS <= 0 || mS >= minWeigh) return null;
  const need = minWeigh / mS;
  let D = niceCeiling(need);
  const dMax = 1 + mM / mS;
  if (D > dMax) D = dMax;
  if (D < need * 0.999) {
    return { possible: false, message: "Total pellet mass is below the weighable limit; increase the pellet mass or the area." };
  }
  const stock = Math.max(minWeigh, 10);
  return {
    possible: true, factor: D, stock_sample_mg: stock, stock_matrix_mg: stock * (D - 1),
    stock_for_pellet_mg: mS * D, extra_matrix_mg: mM - mS * (D - 1),
  };
}

function elementsOf(s: { sample: Material; matrix: Material | null }): string[] {
  return [...new Set([...s.sample.elements, ...(s.matrix ? s.matrix.elements : [])])];
}

function describe(s: TransmissionSetup, c: PelletConstraint, model: TransmissionModel, mS: number, mM: number,
                  ev: Eval, binding: string, feasible: boolean, extraWarnings: string[] = []): TransmissionPlan {
  const warnings = [...extraWarnings];
  const E = model.E;
  const rate = model.rate.slice(0, E.length);
  const i0f = s.scan.i0_dwell_factor;
  const spectra: TransmissionSpectra = {
    E, dwell: model.dwell, k: E.map((e) => kOfE(e, s.E0)), mut: ev.mut, step: ev.step, sigma: ev.sigma, rate,
    counts_I0: rate.map((r, i) => r * model.dwell[i] * i0f),
    counts_I: rate.map((r, i) => r * model.dwell[i] * Math.exp(-ev.mut[i])),
    sigma_chi: ev.sigma.map((sg, i) => (ev.step[i] > 0 ? sg / ev.step[i] : NaN)),
  };
  const lo = E[0];
  const hi = E[E.length - 1];
  if (!s.flux.covers(lo, hi)) {
    warnings.push(`I0 spectrum (${fmtFixed(s.flux.emin, 0)}-${fmtFixed(s.flux.emax, 0)} eV) does not ` +
                  `cover the scan (${fmtFixed(lo, 0)}-${fmtFixed(hi, 0)} eV); points outside have no flux.`);
  }
  const others = interferingEdges(elementsOf(s), lo, hi, [s.absorber, s.edge]);
  if (others.length) {
    const txt = others.map(([el, ed, e]) => `${el} ${ed} (${fmtFixed(e, 0)} eV)`).join(", ");
    warnings.push(`Other absorption edges inside the scan range: ${txt}.`);
  }
  if (s.matrix && s.matrix.massFraction(s.absorber) > 0) {
    warnings.push(`The matrix contains ${s.absorber}; it contributes to the edge step.`);
  }
  if (s.sample.massFraction(s.absorber) === 0) warnings.push(`The sample contains no ${s.absorber}.`);
  const step = ev.step_ref;
  if (step < 0.1) {
    warnings.push(`Edge step ${fmtFixed(step, 3)} is small: transmission will be noise-limited; ` +
                  "consider fluorescence detection.");
  }
  if (s.detector.max_rate && max(rate.filter((r) => !Number.isNaN(r))) > s.detector.max_rate) {
    warnings.push("I0 count rate exceeds the detector's maximum rate somewhere in the scan.");
  }
  const dil = s.matrix ? dilutionRecipe(mS, mM, c.min_weighable_mg) : null;
  if (dil) {
    warnings.push(`Sample mass ${fmtFixed(mS, 2)} mg is below the ${fmtG(c.min_weighable_mg)} mg weighing ` +
                  "limit; use the dilution recipe.");
  }
  const wAbs = s.sample.massFraction(s.absorber) * mS + (s.matrix ? s.matrix.massFraction(s.absorber) * mM : 0);
  const total = mS + mM;
  return {
    m_sample_mg: mS, m_matrix_mg: mM,
    thickness_mm: pelletThicknessMm(mS, mM, s, s.matrix ? c.packing_fraction : 1),
    edge_step: step, mut_below: ev.mut_below, mut_above: ev.mut_above, mut_max: ev.mut_max,
    snr_edge: ev.snr, chi_noise_rms: ev.chi, binding, feasible, warnings, curve: null, spectra,
    dilution: dil, area_cm2: s.area_cm2, absorber_mass_mg: wAbs,
    total_mg: total, sample_wt_pct: total ? (100 * mS) / total : 0,
  };
}

/** Find the sample mass (and matrix mass) that maximizes the objective. */
export function optimize(setup: TransmissionSetup, c: PelletConstraint = pelletConstraint(),
                         nGrid = 1500): TransmissionPlan {
  const model = new TransmissionModel(setup);
  const cap = c.max_mut || 8;
  let hi = ((1.5 * cap * setup.area_cm2) / model.muSampleMax) * 1e3; // sample alone at 1.5x the cap
  if (setup.matrix !== null && c.mode === "fixed_total_mass") {
    const f = Math.min(Math.max(c.min_matrix_fraction, 0), 0.999);
    hi = Math.min(hi, c.total_mass_mg * (1 - f));
  }
  const ms = geomspace(hi * 1e-6, hi, nGrid);
  const mm = ms.map((m) => matrixMass(m, setup, c));
  const evs = ms.map((m, i) => model.evaluate(m, Number.isNaN(mm[i]) ? 0 : mm[i], 1, false));
  const snr = evs.map((e) => e.snr);
  const chi = evs.map((e) => e.chi);
  const score = setup.objective === "edge" ? snr : chi.map((x) => 1 / x);

  const okMass = mm.map(Number.isFinite);
  const okStep = evs.map((e) => (c.max_edge_step ? e.step_ref <= c.max_edge_step : true));
  const okMut = evs.map((e) => (c.max_mut ? e.mut_max <= c.max_mut : true));
  let feasible = ms.map((_, i) => okMass[i] && okStep[i] && okMut[i] && Number.isFinite(score[i]));
  const warnings: string[] = [];
  const isFeasible = feasible.some(Boolean);
  if (!isFeasible) {
    const relaxed = ms.map((_, i) => okMass[i] && okStep[i] && Number.isFinite(score[i]));
    if (relaxed.some(Boolean)) {
      warnings.push(`No loading keeps mu*t below ${fmtG(c.max_mut ?? NaN)} everywhere in the scan ` +
                    "(the matrix alone is too absorbing); the mu*t cap was ignored.");
      feasible = relaxed;
    } else {
      warnings.push("No feasible loading found; check the constraints.");
      feasible = ms.map((_, i) => okMass[i] && Number.isFinite(score[i]));
    }
  }
  const best = argmax(score.map((v, i) => (feasible[i] ? v : -Infinity)));

  let binding = "SNR optimum";
  if (best + 1 < nGrid && !feasible[best + 1]) {
    if (!okStep[best + 1]) binding = `edge-step cap (${fmtG(c.max_edge_step!)})`;
    else if (!okMut[best + 1]) binding = `mu*t cap (${fmtG(c.max_mut!)})`;
    else if (!okMass[best + 1]) binding = "fixed total mass";
  } else if (best === nGrid - 1) {
    binding = "search limit";
  }

  const mBest = Number.isNaN(mm[best]) ? 0 : mm[best];
  const plan = describe(setup, c, model, ms[best], mBest, model.evaluate(ms[best], mBest), binding, isFeasible, warnings);
  plan.curve = {
    m_sample: ms, m_matrix: mm, score, snr, chi, step: evs.map((e) => e.step_ref),
    mut_max: evs.map((e) => e.mut_max), feasible, objective: setup.objective,
  };
  return plan;
}

/** Predictions for masses the user chose (or actually weighed). */
export function evaluateMasses(setup: TransmissionSetup, mS: number, mM: number,
                               c: PelletConstraint = pelletConstraint()): TransmissionPlan {
  const model = new TransmissionModel(setup);
  const ev = model.evaluate(mS, mM);
  const notes: string[] = [];
  if (c.max_edge_step && ev.step_ref > c.max_edge_step) notes.push(`Edge step exceeds the cap of ${fmtG(c.max_edge_step)}.`);
  if (c.max_mut && ev.mut_max > c.max_mut) notes.push(`mu*t exceeds the cap of ${fmtG(c.max_mut)} in part of the scan.`);
  return describe(setup, c, model, mS, mM, ev, "user masses", !notes.length, notes);
}

/** Counting time to reach rms sigma(k^w chi) = target at the given masses. */
export function timeForTarget(setup: TransmissionSetup, mS: number, mM: number, target: number,
                              overheadS = 0): TimeEstimate {
  return estimate(new TransmissionModel(setup), setup.scan, mS, mM, target, overheadS);
}

/** A matrix-free setup over 1 cm²: sample "mass" in mg is the areal density in mg/cm². */
export function foilSetup(material: Material, absorber: string, edge: string, flux: FluxSpectrum,
                          opts: Partial<TransmissionSetupInit> = {}): TransmissionSetup {
  return new TransmissionSetup({ ...opts, sample: material, matrix: null, area_cm2: 1, absorber, edge, flux });
}

export const arealToThicknessUm = (arealMgCm2: number, density: number): number => ((arealMgCm2 * 1e-3) / density) * 1e4;
