/**
 * Fluorescence planner for pellets: signal, self-absorption and noise.
 *
 * Uniform pellet, optical depths tau(E) (total), tau_f (total at the line
 * energy) and a(E) (absorber edge part), angles from the surface:
 *   X     = tau/sin(th_in) + tau_f/sin(th_out)
 *   If/I0 = (Omega/4pi) eta Y (a/sin th_in) (1 - exp(-X)) / X
 * EXAFS amplitude retention (self-absorption):
 *   R = d ln If / d ln a = 1 - (a/sin th_in) (1/X - exp(-X)/(1 - exp(-X)))
 * Extra matrix can help (it lowers a/X), so sample and matrix mass are searched together.
 */
import { DetectorModel } from "./detector";
import type { FluxSpectrum } from "./flux";
import type { Material } from "./materials";
import { argmax, fmtFixed, fmtG, geomspace, interp1, linspace, max, type Vec } from "./num";
import { ScanPlan } from "./scan";
import { estimate, type ChiModel, type TimeEstimate } from "./timing";
import { dilutionRecipe, matrixMass, pelletConstraint, pelletThicknessMm, type Dilution,
         type Objective, type PelletConstraint } from "./transmission";
import { edgeEnergy, edgeStepMuRho, kOfE, muRho } from "./xray";
import { edgeData, fluorLine, xrayEdges } from "./xraydb";

export const EDGE_LINE: Record<string, string> = { K: "Ka", L3: "La", L2: "Lb", L1: "Lb", M5: "Ma", M4: "Mb" };

export interface FluorescenceSetupInit {
  sample: Material;
  matrix: Material | null;
  area_cm2: number;
  absorber: string;
  edge: string;
  flux: FluxSpectrum;
  scan?: ScanPlan | null;
  theta_in?: number;
  theta_out?: number;
  solid_angle_frac?: number; // Omega / 4pi
  det_efficiency?: number;
  flux_scale?: number; // incident rate on the sample / I0 rate in the flux file
  background_per_photon?: number; // scatter counts in the ROI per incident photon
  i0_detector?: DetectorModel;
  objective?: Objective;
  E_ref_offset?: number;
  k_range?: [number, number];
  k_weight?: number;
  min_retention?: number;
  max_total_mass_mg?: number;
}

export class FluorescenceSetup {
  readonly sample: Material;
  readonly matrix: Material | null;
  readonly area_cm2: number;
  readonly absorber: string;
  readonly edge: string;
  readonly flux: FluxSpectrum;
  readonly scan: ScanPlan;
  readonly theta_in: number;
  readonly theta_out: number;
  readonly solid_angle_frac: number;
  readonly det_efficiency: number;
  readonly flux_scale: number;
  readonly background_per_photon: number;
  readonly i0_detector: DetectorModel;
  readonly objective: Objective;
  readonly E_ref_offset: number;
  readonly k_range: [number, number];
  readonly k_weight: number;
  readonly min_retention: number;
  readonly max_total_mass_mg: number;
  readonly E0: number;

  constructor(o: FluorescenceSetupInit) {
    this.sample = o.sample;
    this.matrix = o.matrix;
    this.area_cm2 = o.area_cm2;
    this.absorber = o.absorber;
    this.edge = o.edge;
    this.flux = o.flux;
    this.E0 = edgeEnergy(o.absorber, o.edge);
    this.scan = o.scan ?? new ScanPlan(this.E0);
    this.theta_in = o.theta_in ?? 45;
    this.theta_out = o.theta_out ?? 45;
    this.solid_angle_frac = o.solid_angle_frac ?? 0.01;
    this.det_efficiency = o.det_efficiency ?? 1;
    this.flux_scale = o.flux_scale ?? 1;
    this.background_per_photon = o.background_per_photon ?? 0;
    this.i0_detector = o.i0_detector ?? new DetectorModel();
    this.objective = o.objective ?? "exafs";
    this.E_ref_offset = o.E_ref_offset ?? 50;
    this.k_range = o.k_range ?? [3, 10];
    this.k_weight = o.k_weight ?? 2;
    this.min_retention = o.min_retention ?? 0.9;
    this.max_total_mass_mg = o.max_total_mass_mg ?? 300;
  }

  /** [yield x probability, mean line energy] of the main fluorescence line. */
  line(): [number, number] {
    const fam = EDGE_LINE[this.edge] ?? "Ka";
    const ln = EDGE_LINE[this.edge] ? fluorLine(this.absorber, this.edge) : null;
    if (!ln) throw new Error(`no ${fam} fluorescence line for ${this.absorber} ${this.edge}`);
    const { energy: e0, fyield } = edgeData(this.absorber, this.edge);
    const y = this.E0 + this.E_ref_offset < e0 ? 0 : fyield;
    return [y * ln.prob, ln.energy];
  }
}

export interface FluorescencePlan {
  m_sample_mg: number;
  m_matrix_mg: number;
  thickness_mm: number | null;
  retention_ref: number;
  retention_min_k: number;
  fluo_rate_ref: number; // counts/s in the fluorescence ROI at E_ref
  snr_edge: number;
  chi_noise_rms: number;
  line_energy: number;
  binding: string;
  warnings: string[];
  curve: { m_sample: Vec; best_score: Vec; retention: Vec; objective: Objective };
  spectra: { E: Vec; k: Vec; dwell: Vec; R: Vec; counts_f: Vec; sigma_chi: Vec; rate: Vec };
  dilution: Dilution | null;
  total_mg: number;
  sample_wt_pct: number;
}

interface FEval {
  R: Vec; n_f: Vec; sig_chi: Vec;
  R_ref: number; rate_ref: number; snr: number; chi: number; R_min: number;
}

export class FluorescenceModel implements ChiModel {
  readonly E: Vec;
  readonly dwell: Vec;
  readonly rate: Vec;
  readonly nPoints: number;
  readonly Y: number;
  readonly E_f: number;
  readonly mu_s: Vec;
  private readonly mu_m: Vec;
  private readonly muf_s: number;
  private readonly muf_m: number;
  private readonly dmu_s: Vec;
  private readonly dmu_m: Vec;
  private readonly t: Vec;
  private readonly kIdx: number[];
  private readonly kw: Vec;
  private readonly sinIn: number;
  private readonly sinOut: number;

  constructor(readonly s: FluorescenceSetup) {
    const { E, t } = s.scan.grid();
    this.E = E;
    this.dwell = t;
    this.nPoints = E.length;
    const Eref = s.E0 + s.E_ref_offset;
    const pts = [...E, Eref];
    [this.Y, this.E_f] = s.line();
    this.mu_s = muRho(s.sample, pts);
    this.mu_m = s.matrix ? muRho(s.matrix, pts) : pts.map(() => 0);
    this.muf_s = muRho(s.sample, [this.E_f])[0];
    this.muf_m = s.matrix ? muRho(s.matrix, [this.E_f])[0] : 0;
    const step = edgeStepMuRho(s.absorber, s.edge, pts);
    const ws = s.sample.massFraction(s.absorber);
    const wm = s.matrix ? s.matrix.massFraction(s.absorber) : 0;
    this.dmu_s = step.map((v) => ws * v);
    this.dmu_m = step.map((v) => wm * v);
    this.rate = s.flux.at(pts).map((r) => r * s.flux_scale);
    this.t = [...t, interp1(Eref, E, t)];
    const k = pts.map((e) => kOfE(e, s.E0));
    this.kIdx = [];
    for (let i = 0; i < E.length; i++) {
      if (pts[i] >= s.E0 && k[i] >= s.k_range[0] && k[i] <= s.k_range[1]) this.kIdx.push(i);
    }
    this.kw = this.kIdx.map((i) => k[i] ** s.k_weight);
    this.sinIn = Math.sin((s.theta_in * Math.PI) / 180);
    this.sinOut = Math.sin((s.theta_out * Math.PI) / 180);
  }

  evaluate(mS: number, mM: number, factor = 1, rows = true): FEval {
    const s = this.s;
    const ms = mS * 1e-3;
    const mm = mM * 1e-3;
    const A = s.area_cm2;
    const N = this.nPoints + 1;
    const i0f = s.scan.i0_dwell_factor;
    const tauF = (ms * this.muf_s + mm * this.muf_m) / A;
    const R = new Array<number>(N);
    const nF = new Array<number>(N);
    const sigChi = new Array<number>(N);
    const sigRel = new Array<number>(N);
    let yldRef = 0;
    for (let i = 0; i < N; i++) {
      const tau = (ms * this.mu_s[i] + mm * this.mu_m[i]) / A;
      const a = (ms * this.dmu_s[i] + mm * this.dmu_m[i]) / A;
      const X = tau / this.sinIn + tauF / this.sinOut;
      const esc = X > 1e-8 ? -Math.expm1(-X) / X : 1 - X / 2;
      const yld = s.solid_angle_frac * s.det_efficiency * this.Y * (a / this.sinIn) * esc;
      const corr = X > 1e-6 ? 1 / X - 1 / Math.expm1(X) : 0.5 - X / 12;
      R[i] = 1 - (a / this.sinIn) * corr;
      const t = this.t[i] * factor;
      const t0 = t * i0f;
      nF[i] = this.rate[i] * t * yld;
      const nBg = this.rate[i] * t * s.background_per_photon;
      const n0 = this.rate[i] * t * i0f;
      const rel = (nF[i] + nBg) / (nF[i] * nF[i]) + s.i0_detector.variance(n0, t0) / (n0 * n0);
      sigRel[i] = nF[i] > 0 ? Math.sqrt(rel) : Infinity;
      sigChi[i] = sigRel[i] / R[i];
      if (i === N - 1) yldRef = yld;
    }
    let chi = NaN;
    let rmin = R[N - 1];
    if (this.kIdx.length) {
      let acc = 0;
      rmin = Infinity;
      this.kIdx.forEach((i, j) => {
        const v = sigChi[i] * this.kw[j];
        acc += v * v;
        if (R[i] < rmin || Number.isNaN(R[i])) rmin = R[i];
      });
      chi = Math.sqrt(acc / this.kIdx.length);
    }
    const n = this.nPoints;
    return {
      R: rows ? R.slice(0, n) : [], n_f: rows ? nF.slice(0, n) : [], sig_chi: rows ? sigChi.slice(0, n) : [],
      R_ref: R[n], rate_ref: this.rate[n] * yldRef, snr: 1 / sigRel[n], chi, R_min: rmin,
    };
  }

  chiRms(mS: number, mM: number, factor: number): number {
    return this.evaluate(mS, mM, factor, false).chi;
  }
}

export function optimize(setup: FluorescenceSetup, c: PelletConstraint = pelletConstraint(),
                         nSample = 300, nMatrix = 40): FluorescencePlan {
  const model = new FluorescenceModel(setup);
  const s = setup;
  const Mmax = Math.max(s.max_total_mass_mg, c.total_mass_mg);
  const hi = s.matrix ? Mmax : (50 * s.area_cm2 * 1e3) / max(model.mu_s);
  const ms = geomspace(hi * 1e-6, hi, nSample);
  const mm0 = ms.map((m) => matrixMass(m, s, c));
  const lam = s.matrix === null || c.mode === "fixed_total_mass" || c.mode === "none" ? [0] : linspace(0, 1, nMatrix);
  const L = lam.length;
  const MS: Vec = [];
  const MM0: Vec = [];
  const MM: Vec = [];
  for (let i = 0; i < ms.length; i++) {
    const base = Number.isNaN(mm0[i]) ? 0 : mm0[i];
    const room = Math.max(Mmax - ms[i] - base, 0);
    for (const l of lam) {
      MS.push(ms[i]);
      MM0.push(mm0[i]);
      MM.push(base + l * room);
    }
  }
  const ev = MS.map((m, i) => model.evaluate(m, MM[i], 1, false));
  const score = ev.map((e) => (s.objective === "edge" ? e.snr : 1 / e.chi));
  const ok = MS.map((m, i) => Number.isFinite(MM0[i]) && Number.isFinite(score[i]) && m + MM[i] <= Mmax + 1e-9);
  const feas = ok.map((o, i) => o && ev[i].R_min >= s.min_retention);
  const anyFeas = feas.some(Boolean);
  const warnings: string[] = [];
  let pick: Vec;
  if (!anyFeas) {
    warnings.push(`No loading reaches the ${fmtG(s.min_retention)} retention target; showing the best retention achievable.`);
    pick = ev.map((e, i) => (ok[i] ? e.R_min : -Infinity));
  } else {
    pick = score.map((v, i) => (feas[i] ? v : -Infinity));
  }
  const b = argmax(pick);
  const mS = MS[b];
  const mM = MM[b];

  // best allowed score per sample mass, for the plot
  const bestPer: Vec = [];
  const ret: Vec = [];
  for (let i = 0; i < ms.length; i++) {
    let m = -Infinity;
    for (let j = 0; j < L; j++) {
      const q = i * L + j;
      if (feas[q] && !Number.isNaN(score[q]) && score[q] > m) m = score[q];
    }
    bestPer.push(Number.isFinite(m) ? m : NaN);
    ret.push(ev[i * L].R_min);
  }

  let binding = "SNR optimum";
  if (!anyFeas) binding = "retention target not reachable";
  else if (ev[b].R_min < s.min_retention + 0.01) binding = `retention target (${fmtG(s.min_retention)})`;
  else if (MS[b] + MM[b] >= Mmax * 0.999) binding = `maximum pellet mass (${fmtG(Mmax)} mg)`;

  const e1 = model.evaluate(mS, mM);
  const spectra = {
    E: model.E, k: model.E.map((e) => kOfE(e, s.E0)), dwell: model.dwell, R: e1.R, counts_f: e1.n_f,
    sigma_chi: e1.sig_chi, rate: model.rate.slice(0, -1),
  };
  if (s.background_per_photon === 0) warnings.push("Scatter background is set to zero; real SNR will be lower.");
  const dil = s.matrix ? dilutionRecipe(mS, mM, c.min_weighable_mg) : null;
  if (dil) {
    warnings.push(`Sample mass ${fmtFixed(mS, 2)} mg is below the ${fmtG(c.min_weighable_mg)} mg ` +
                  "weighing limit; use the dilution recipe.");
  }
  const present = [...new Set([...s.sample.elements, ...(s.matrix ? s.matrix.elements : [])])]
    .filter((el) => el !== s.absorber);
  const Emax = model.E[model.E.length - 1];
  for (const el of present) {
    for (const [name, e] of Object.entries(xrayEdges(el))) {
      if (model.E_f < e && e < Emax && e > s.E0 - 200 && (name === "K" || name === "L3")) {
        warnings.push(`${el} ${name} edge (${fmtFixed(e, 0)} eV) may add fluorescence or scatter background.`);
      }
    }
  }
  const total = mS + mM;
  return {
    m_sample_mg: mS, m_matrix_mg: mM,
    thickness_mm: s.matrix ? pelletThicknessMm(mS, mM, s, c.packing_fraction) : null,
    retention_ref: e1.R_ref, retention_min_k: e1.R_min, fluo_rate_ref: e1.rate_ref, snr_edge: e1.snr,
    chi_noise_rms: e1.chi, line_energy: model.E_f, binding, warnings, spectra, dilution: dil,
    curve: { m_sample: ms, best_score: bestPer, retention: ret, objective: s.objective },
    total_mg: total, sample_wt_pct: total ? (100 * mS) / total : 0,
  };
}

/** Counting time to reach rms sigma(k^w chi) = target at the given masses. */
export function timeForTarget(setup: FluorescenceSetup, mS: number, mM: number, target: number,
                              overheadS = 0): TimeEstimate {
  return estimate(new FluorescenceModel(setup), setup.scan, mS, mM, target, overheadS);
}
