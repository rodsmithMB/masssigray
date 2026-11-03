/**
 * Detector noise model and photon-transfer calibration.
 *
 * Variance of the signal in one energy point, in effective counts²:
 *   var(S) = n_frames read_var + dark_rate t + S + (instability S)²
 *
 * Calibration fits the same model in ADU to repeated I0 scans at several dwell
 * times, plus optional dark scans:
 *   var_ADU = R + D t + g m + eps² m²
 * g is ADU per effective count.
 */
import { interp, mean, median, nnls, Rng, variance, type Vec } from "./num";

export class DetectorModel {
  constructor(
    readonly read_var = 0,
    readonly dark_rate = 0,
    readonly instability = 0,
    readonly frame_time: number | null = null, // s; null means one frame per point
    readonly max_rate: number | null = null, // effective counts/s before saturation
    readonly label = "ideal Poisson",
  ) {}

  nFrames(t: number): number {
    if (!this.frame_time) return 1;
    return Math.max(1, Math.ceil(t / this.frame_time - 1e-9));
  }

  variance(counts: number, t: number): number {
    return this.nFrames(t) * this.read_var + this.dark_rate * t + counts + (this.instability * counts) ** 2;
  }
}

/**
 * Standard deviation of mu*t = ln(I0/I) for one energy point; I0 and I are
 * recorded in separate scans with dwell times tI0 and tSample.
 */
export function sigmaMut(rate: number, mut: number, tSample: number, tI0: number, det: DetectorModel): number {
  const n0 = rate * tI0;
  const n = n0 * (tSample / tI0) * Math.exp(-mut);
  if (!(n > 0 && n0 > 0)) return Infinity;
  return Math.sqrt(det.variance(n, tSample) / (n * n) + det.variance(n0, tI0) / (n0 * n0));
}

// ------------------------------------------------------------------ calibration

export interface Scan {
  dwell: number; // s per point
  energy: Vec;
  values: Vec; // ADU per point (ROI sum)
  dark: boolean;
}

export class CalibrationResult {
  constructor(
    readonly gain: number,
    readonly read_var_adu: number,
    readonly dark_var_rate_adu: number,
    readonly instability: number,
    readonly drift_rms: number,
    readonly dark_level_adu_per_s: number,
    readonly n_points: number,
    public r2: number,
    readonly points: { t: Vec; mean: Vec; var: Vec },
  ) {}

  toDetector(frameTime: number | null = null, maxRate: number | null = null): DetectorModel {
    const g2 = this.gain ** 2;
    return new DetectorModel(this.read_var_adu / g2, this.dark_var_rate_adu / g2, this.instability,
                             frameTime, maxRate, "calibrated");
  }

  predictedVar(meanAdu: number, t: number): number {
    return this.read_var_adu + this.dark_var_rate_adu * t + this.gain * meanAdu + (this.instability * meanAdu) ** 2;
  }

  summary(): Record<string, number> {
    return {
      "gain (ADU per effective count)": this.gain,
      "read-noise variance (ADU^2/point)": this.read_var_adu,
      "dark variance rate (ADU^2/s)": this.dark_var_rate_adu,
      "point-to-point flux instability (fraction)": this.instability,
      "scan-to-scan drift rms (fraction)": this.drift_rms,
      "dark level (ADU/s)": this.dark_level_adu_per_s,
      "points used": this.n_points,
      "observed / predicted variance (should be ~1)": this.r2,
    };
  }
}

function group(scans: Scan[]): Map<number, Scan[]> {
  const g = new Map<number, Scan[]>();
  for (const s of scans) {
    const key = Math.round(s.dwell * 1e9) / 1e9;
    if (!g.has(key)) g.set(key, []);
    g.get(key)!.push(s);
  }
  return g;
}

/** Values of repeated scans on the energy grid of the first one. */
function common(scans: Scan[]): Vec[] {
  const E = scans[0].energy;
  return scans.map((s) => interp(E, s.energy, s.values));
}

const column = (V: Vec[], j: number): Vec => V.map((row) => row[j]);

/**
 * Fit the noise model to repeated I0 scans (and optional dark scans). Needs at
 * least two repeats at one or more dwell times.
 */
export function calibrate(scans: Scan[], minCounts = 0): CalibrationResult {
  const light = group(scans.filter((s) => !s.dark));
  const darks = group(scans.filter((s) => s.dark));

  let darkLevel = 0;
  const darkRows: [number, number][] = [];
  if (darks.size) {
    const levels: Vec = [];
    for (const [t, grp] of darks) {
      const V = common(grp);
      levels.push(mean(V.flat()) / t);
      if (grp.length >= 2) {
        const vars = V[0].map((_, j) => variance(column(V, j), 1));
        darkRows.push([t, mean(vars)]);
      } else if (V[0].length > 3) {
        darkRows.push([t, variance(V[0], 1)]);
      }
    }
    darkLevel = mean(levels);
  }

  const rows: number[][] = [];
  const rhs: Vec = [];
  const pts = { t: [] as Vec, mean: [] as Vec, var: [] as Vec };
  const drifts: Vec = [];
  for (const [t, grp] of light) {
    if (grp.length < 2) continue;
    const V = common(grp).map((row) => row.map((v) => v - darkLevel * t));
    const npt = V[0].length;
    const ref = Array.from({ length: npt }, (_, j) => mean(column(V, j)));
    const good = ref.map((r) => r > Math.max(minCounts, 0) + 1e-12);
    const scale = V.map((v) => median(v.flatMap((x, j) => (good[j] ? [x / ref[j]] : []))));
    const sm = mean(scale);
    for (const s of scale) drifts.push(s / sm - 1);
    const Vc = V.map((row, i) => row.map((x) => x / scale[i]));
    for (let j = 0; j < npt; j++) {
      const col = column(Vc, j);
      const m = mean(col);
      const v = variance(col, 1);
      if (good[j] && v > 0) {
        rows.push([1, t, m, m * m]);
        rhs.push(v);
        pts.t.push(t);
        pts.mean.push(m);
        pts.var.push(v);
      }
    }
  }
  if (rows.length < 4) throw new Error("need at least two repeated I0 scans at one dwell time");
  for (const [t, v] of darkRows) {
    rows.push([1, t, 0, 0]);
    rhs.push(v);
  }
  const X = rows;
  const y = rhs;
  if (new Set([...pts.t, ...darkRows.map(([t]) => t)]).size < 2) {
    for (const r of X) r[1] = 0; // one dwell time: dark rate not separable from read noise
  }
  // Iteratively reweighted least squares, weighted by the predicted variance.
  let pred = y.map(() => mean(y));
  let coef = [0, 0, 0, 0];
  for (let it = 0; it < 6; it++) {
    const Aw = X.map((r, i) => r.map((v) => v / pred[i]));
    const norm = [0, 1, 2, 3].map((j) => Math.sqrt(Aw.reduce((s, r) => s + r[j] * r[j], 0)) || 1);
    const c = nnls(Aw.map((r) => r.map((v, j) => v / norm[j])), y.map((v, i) => v / pred[i]));
    coef = c.map((v, j) => v / norm[j]);
    pred = X.map((r) => Math.max(r.reduce((s, v, j) => s + v * coef[j], 0), 1e-12));
  }
  const [R, D, g, eps2] = coef;
  if (!(g > 0)) {
    throw new Error("fit gave no shot-noise term; check that scans are dark-subtracted " +
                    "and span a useful range of signal levels");
  }
  const res = new CalibrationResult(
    g, R, D, Math.sqrt(eps2),
    drifts.length ? Math.sqrt(mean(drifts.map((d) => d * d))) : 0,
    darkLevel, pts.t.length, 0, pts,
  );
  res.r2 = mean(pts.var.map((v, i) => v / res.predictedVar(pts.mean[i], pts.t[i])));
  return res;
}

/** Synthetic calibration data, for tests and demonstrations. */
export function simulateScans(rateFn: (E: Vec) => Vec, energy: Vec, opts: {
  dwells?: number[]; repeats?: number; gain?: number; readSdAdu?: number; darkAduPerS?: number;
  instability?: number; drift?: number; nDark?: number; seed?: number;
} = {}): Scan[] {
  const { dwells = [0.5, 1, 2, 5, 10], repeats = 3, gain = 2.5, readSdAdu = 30, darkAduPerS = 50,
          instability = 0.002, drift = 0.01, nDark = 2, seed = 0 } = opts;
  const rng = new Rng(seed);
  const rate = rateFn(energy);
  const scans: Scan[] = [];
  for (const t of dwells) {
    for (let r = 0; r < repeats; r++) {
      const d = 1 + drift * rng.normal();
      const values = rate.map((rt) => {
        const n = Math.max(rt * t * d * (1 + instability * rng.normal()), 0);
        return gain * rng.poisson(n) + rng.poisson(darkAduPerS * t) + readSdAdu * rng.normal();
      });
      scans.push({ dwell: t, energy, values, dark: false });
    }
    for (let r = 0; r < nDark; r++) {
      const values = energy.map(() => rng.poisson(darkAduPerS * t) + readSdAdu * rng.normal());
      scans.push({ dwell: t, energy, values, dark: true });
    }
  }
  return scans;
}
