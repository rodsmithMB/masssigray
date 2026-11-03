/**
 * Numerical helpers replacing the numpy/scipy calls of the Python version.
 * Each follows the numpy/scipy semantics it replaces (noted per function),
 * so results agree to rounding error.
 */

export type Vec = number[];

/** np.arange for floats: n = ceil((stop - start) / step), x[i] = start + i * delta. */
export function arange(start: number, stop: number, step: number): Vec {
  const n = Math.max(0, Math.ceil((stop - start) / step));
  const delta = start + step - start; // numpy fills with this difference
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = start + i * delta;
  return out;
}

/** np.linspace with endpoint. */
export function linspace(start: number, stop: number, num: number): Vec {
  if (num === 1) return [start];
  const step = (stop - start) / (num - 1);
  const out = new Array<number>(num);
  for (let i = 0; i < num; i++) out[i] = i * step + start;
  out[num - 1] = stop;
  return out;
}

/** np.geomspace: 10**linspace(log10 a, log10 b), endpoints exact. */
export function geomspace(start: number, stop: number, num: number): Vec {
  const out = linspace(Math.log10(start), Math.log10(stop), num).map((v) => 10 ** v);
  out[0] = start;
  out[num - 1] = stop;
  return out;
}

/** Index of the last xp[j] <= x (binary search); -1 if x < xp[0]. */
function searchRight(xp: Vec, x: number): number {
  let lo = 0;
  let hi = xp.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

/** np.interp for one value. */
export function interp1(x: number, xp: Vec, fp: Vec, left?: number, right?: number): number {
  const n = xp.length;
  if (Number.isNaN(x)) return x;
  if (x < xp[0]) return left ?? fp[0];
  if (x > xp[n - 1]) return right ?? fp[n - 1];
  if (x === xp[n - 1]) return fp[n - 1];
  const j = searchRight(xp, x);
  if (x === xp[j]) return fp[j];
  const slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  return slope * (x - xp[j]) + fp[j];
}

/** np.interp over an array. */
export function interp(x: Vec, xp: Vec, fp: Vec, left?: number, right?: number): Vec {
  return x.map((v) => interp1(v, xp, fp, left, right));
}

export const sum = (a: Vec): number => a.reduce((s, v) => s + v, 0);
export const mean = (a: Vec): number => sum(a) / a.length;

/** Maximum ignoring nothing: NaN propagates like np.max. */
export function max(a: Vec): number {
  let m = -Infinity;
  for (const v of a) {
    if (Number.isNaN(v)) return NaN;
    if (v > m) m = v;
  }
  return m;
}

export function min(a: Vec): number {
  let m = Infinity;
  for (const v of a) {
    if (Number.isNaN(v)) return NaN;
    if (v < m) m = v;
  }
  return m;
}

/** np.argmax: first maximum; a NaN counts as the maximum (first NaN wins). */
export function argmax(a: Vec): number {
  let best = 0;
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(a[i])) return i;
    if (a[i] > a[best]) best = i;
  }
  return best;
}

/** Sample variance with ddof. */
export function variance(a: Vec, ddof = 0): number {
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return s / (a.length - ddof);
}

/** np.median. */
export function median(a: Vec): number {
  const s = [...a].sort((x, y) => x - y);
  const n = s.length;
  if (n === 0) return NaN;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/** Indices that sort `a` ascending (stable). */
export function argsort(a: Vec): number[] {
  return a.map((_, i) => i).sort((i, j) => a[i] - a[j] || i - j);
}

/** np.gradient(y, x): second-order central differences inside, first-order at the ends. */
export function gradient(y: Vec, x: Vec): Vec {
  const n = y.length;
  const g = new Array<number>(n);
  for (let i = 1; i < n - 1; i++) {
    const d1 = x[i] - x[i - 1];
    const d2 = x[i + 1] - x[i];
    const a = -d2 / (d1 * (d1 + d2));
    const b = (d2 - d1) / (d1 * d2);
    const c = d1 / (d2 * (d1 + d2));
    g[i] = a * y[i - 1] + b * y[i] + c * y[i + 1];
  }
  g[0] = (y[1] - y[0]) / (x[1] - x[0]);
  g[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
  return g;
}

// ------------------------------------------------------------------ least squares

/**
 * Least-squares solution of A x = b by Householder QR (A is m x n, m >= n,
 * full column rank). Rows of A are arrays of length n.
 */
export function lstsq(A: number[][], b: Vec): Vec {
  const m = A.length;
  const n = A[0].length;
  const R = A.map((r) => [...r]);
  const y = [...b];
  for (let k = 0; k < n; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i][k] * R[i][k];
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = R[k][k] > 0 ? -norm : norm;
    const v = new Array<number>(m).fill(0);
    for (let i = k; i < m; i++) v[i] = R[i][k];
    v[k] -= alpha;
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i] * v[i];
    if (vv === 0) continue;
    for (let j = k; j < n; j++) {
      let s = 0;
      for (let i = k; i < m; i++) s += v[i] * R[i][j];
      const f = (2 * s) / vv;
      for (let i = k; i < m; i++) R[i][j] -= f * v[i];
    }
    let s = 0;
    for (let i = k; i < m; i++) s += v[i] * y[i];
    const f = (2 * s) / vv;
    for (let i = k; i < m; i++) y[i] -= f * v[i];
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= R[i][j] * x[j];
    x[i] = R[i][i] === 0 ? 0 : s / R[i][i];
  }
  return x;
}

/** A polynomial least-squares fit (np.polyfit + np.polyval), fitted in scaled x for stability. */
export function polyfit(x: Vec, y: Vec, deg: number): (t: number) => number {
  const xm = mean(x);
  let s = 0;
  for (const v of x) s = Math.max(s, Math.abs(v - xm));
  s = s || 1;
  const A = x.map((v) => {
    const u = (v - xm) / s;
    const row: number[] = [];
    for (let p = 0; p <= deg; p++) row.push(u ** p);
    return row;
  });
  const c = lstsq(A, y);
  return (t: number) => {
    const u = (t - xm) / s;
    let r = 0;
    for (let p = deg; p >= 0; p--) r = r * u + c[p];
    return r;
  };
}

/**
 * Non-negative least squares, min ||A x - b|| subject to x >= 0
 * (Lawson & Hanson active-set algorithm, as scipy.optimize.nnls).
 */
export function nnls(A: number[][], b: Vec, maxIter = 300): Vec {
  const m = A.length;
  const n = A[0].length;
  const x = new Array<number>(n).fill(0);
  const passive = new Array<boolean>(n).fill(false);
  const tol = 10 * Number.EPSILON * Math.max(m, n) * Math.max(1, ...A.flat().map(Math.abs));
  const grad = (): Vec => {
    const r = b.map((bi, i) => bi - A[i].reduce((s, a, j) => s + a * x[j], 0));
    return Array.from({ length: n }, (_, j) => A.reduce((s, row, i) => s + row[j] * r[i], 0));
  };
  const solveP = (): Vec => {
    const idx = passive.flatMap((p, j) => (p ? [j] : []));
    const z = new Array<number>(n).fill(0);
    if (!idx.length) return z;
    const sol = lstsq(A.map((row) => idx.map((j) => row[j])), b);
    idx.forEach((j, k) => (z[j] = sol[k]));
    return z;
  };
  let w = grad();
  for (let iter = 0; iter < maxIter; iter++) {
    let jmax = -1;
    for (let j = 0; j < n; j++) if (!passive[j] && w[j] > tol && (jmax < 0 || w[j] > w[jmax])) jmax = j;
    if (jmax < 0) break;
    passive[jmax] = true;
    let z = solveP();
    for (let inner = 0; inner < maxIter; inner++) {
      if (passive.every((p, j) => !p || z[j] > 0)) break;
      let alpha = Infinity;
      for (let j = 0; j < n; j++) {
        if (passive[j] && z[j] <= 0) alpha = Math.min(alpha, x[j] / (x[j] - z[j]));
      }
      for (let j = 0; j < n; j++) x[j] += alpha * (z[j] - x[j]);
      for (let j = 0; j < n; j++) if (passive[j] && Math.abs(x[j]) <= tol) passive[j] = false;
      z = solveP();
    }
    for (let j = 0; j < n; j++) x[j] = passive[j] ? z[j] : 0;
    w = grad();
  }
  return x;
}

/** Brent's root finder (scipy.optimize.brentq), f(a) and f(b) of opposite sign. */
export function brentq(f: (x: number) => number, a: number, b: number, xtol = 2e-12,
                       rtol = 4 * Number.EPSILON, maxIter = 100): number {
  let xpre = a, xcur = b;
  let fpre = f(xpre), fcur = f(xcur);
  if (fpre * fcur > 0) throw new Error("f(a) and f(b) must have different signs");
  if (fpre === 0) return xpre;
  if (fcur === 0) return xcur;
  let xblk = 0, fblk = 0, spre = 0, scur = 0;
  for (let i = 0; i < maxIter; i++) {
    if (fpre !== 0 && fcur !== 0 && Math.sign(fpre) !== Math.sign(fcur)) {
      xblk = xpre;
      fblk = fpre;
      spre = scur = xcur - xpre;
    }
    if (Math.abs(fblk) < Math.abs(fcur)) {
      xpre = xcur; xcur = xblk; xblk = xpre;
      fpre = fcur; fcur = fblk; fblk = fpre;
    }
    const delta = (xtol + rtol * Math.abs(xcur)) / 2;
    const sbis = (xblk - xcur) / 2;
    if (fcur === 0 || Math.abs(sbis) < delta) return xcur;
    if (Math.abs(spre) > delta && Math.abs(fcur) < Math.abs(fpre)) {
      let stry: number;
      if (xpre === xblk) {
        stry = (-fcur * (xcur - xpre)) / (fcur - fpre); // secant
      } else {
        const dpre = (fpre - fcur) / (xpre - xcur);
        const dblk = (fblk - fcur) / (xblk - xcur);
        stry = (-fcur * (fblk * dblk - fpre * dpre)) / (dblk * dpre * (fblk - fpre)); // inverse quadratic
      }
      if (2 * Math.abs(stry) < Math.min(Math.abs(spre), 3 * Math.abs(sbis) - delta)) {
        spre = scur;
        scur = stry;
      } else {
        spre = sbis;
        scur = sbis;
      }
    } else {
      spre = sbis;
      scur = sbis;
    }
    xpre = xcur;
    fpre = fcur;
    if (Math.abs(scur) > delta) xcur += scur;
    else xcur += sbis > 0 ? delta : -delta;
    fcur = f(xcur);
  }
  return xcur;
}

// ------------------------------------------------------------------ random numbers

/** Seeded generator with normal and Poisson draws (for simulated noise only). */
export class Rng {
  private s: number;
  private spare: number | null = null;

  constructor(seed = 0) {
    this.s = (seed * 2654435761) >>> 0 || 0x9e3779b9;
  }

  /** Uniform in (0, 1) (mulberry32). */
  uniform(): number {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  }

  normal(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    const r = Math.sqrt(-2 * Math.log(this.uniform()));
    const th = 2 * Math.PI * this.uniform();
    this.spare = r * Math.sin(th);
    return r * Math.cos(th);
  }

  /** Poisson draw: exact for small means, normal approximation above 50. */
  poisson(lam: number): number {
    if (!(lam > 0)) return 0;
    if (lam > 50) return Math.max(0, Math.round(lam + Math.sqrt(lam) * this.normal()));
    const L = Math.exp(-lam);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= this.uniform();
    } while (p > L);
    return k - 1;
  }
}

// ------------------------------------------------------------------ formatting (Python-style)

/** Python f"{x:.{d}f}" (round-half-even on exact ties). */
export function fmtFixed(x: number, d: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  if (d === 0) {
    const f = Math.floor(x);
    if (x - f === 0.5) return String(f % 2 === 0 ? f : f + 1);
  }
  const s = x.toFixed(d);
  return s === "-0" || /^-0\.0*$/.test(s) ? s.slice(1) : s;
}

/** Python f"{x:.{p}g}". */
export function fmtG(x: number, p = 6): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  if (x === 0) return "0";
  const exp = Number(x.toExponential(p - 1).split("e")[1]);
  if (exp < -4 || exp >= p) {
    const [mant, e] = x.toExponential(p - 1).split("e");
    const m = mant.includes(".") ? mant.replace(/\.?0+$/, "") : mant;
    const en = Number(e);
    return `${m}e${en < 0 ? "-" : "+"}${String(Math.abs(en)).padStart(2, "0")}`;
  }
  const s = x.toFixed(Math.max(0, p - 1 - exp));
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}

/** Python float(): accepts ints, decimals, exponents, nan/inf. Returns null if not a number. */
export function parseFloatPy(s: string): number | null {
  const t = s.trim().replace(/_/g, "");
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t)) return Number(t);
  const l = t.toLowerCase().replace(/^\+/, "");
  if (l === "nan" || l === "-nan") return NaN;
  if (l === "inf" || l === "infinity") return Infinity;
  if (l === "-inf" || l === "-infinity") return -Infinity;
  return null;
}
