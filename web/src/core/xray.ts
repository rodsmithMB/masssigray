/** X-ray cross sections and absorption edges. Mass attenuation coefficients in cm²/g. */
import type { Material } from "./materials";
import { edgeData, muElam, xrayEdges, type XsKind } from "./xraydb";

export const HC_K = 0.262468; // 2m/hbar² in eV⁻¹ Å⁻², k = sqrt(HC_K (E - E0))
export const EDGE_NAMES = ["K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"];

export const kOfE = (E: number, E0: number): number => Math.sqrt(Math.max(HC_K * (E - E0), 0));
export const EOfK = (k: number, E0: number): number => E0 + (k * k) / HC_K;

export const edgeEnergy = (element: string, edge: string): number => edgeData(element, edge).energy;

/** Edges of an element in [emin, emax], {edge: energy}. */
export function elementEdges(element: string, emin = 0, emax = Infinity): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [name, e] of Object.entries(xrayEdges(element))) {
    if (EDGE_NAMES.includes(name) && emin <= e && e <= emax) out[name] = e;
  }
  return out;
}

export const muRhoElement = (element: string, E: number[], kind: XsKind = "total"): number[] =>
  muElam(element, E, kind);

/** Mass attenuation coefficient of a material, cm²/g. */
export function muRho(material: Material, E: number[], kind: XsKind = "total"): number[] {
  const out = new Array<number>(E.length).fill(0);
  for (const [el, w] of material.massFractions) {
    const mu = muRhoElement(el, E, kind);
    for (let i = 0; i < E.length; i++) out[i] += w * mu[i];
  }
  return out;
}

const preEdgeCache = new Map<string, [number, number]>();

/** Log-log line through the element's photoabsorption just below the edge: [slope, intercept]. */
function preEdgeFit(element: string, edge: string): [number, number] {
  const key = `${element}/${edge}`;
  const hit = preEdgeCache.get(key);
  if (hit) return hit;
  const E0 = edgeEnergy(element, edge);
  const lower = Object.values(elementEdges(element, 0, E0 - 0.5));
  const floor = lower.length ? Math.max(...lower) + 2 : 0;
  const hi = E0 - 2;
  let lo = Math.max(E0 - 80, floor);
  if (lo >= hi - 1) lo = hi - 1;
  const [m0, m1] = muRhoElement(element, [lo, hi], "photo");
  const slope = Math.log(m1 / m0) / Math.log(hi / lo);
  const fit: [number, number] = [slope, Math.log(m1) - slope * Math.log(hi)];
  preEdgeCache.set(key, fit);
  return fit;
}

/**
 * Edge contribution of one element, cm² per gram of that element: the
 * photoabsorption above the edge minus the pre-edge extrapolation; zero below.
 */
export function edgeStepMuRho(element: string, edge: string, E: number[]): number[] {
  const E0 = edgeEnergy(element, edge);
  const [slope, icpt] = preEdgeFit(element, edge);
  const above = muRhoElement(element, E, "photo");
  return E.map((e, i) => {
    if (!(e >= E0)) return 0;
    const step = above[i] - Math.exp(icpt + slope * Math.log(e));
    return step > 0 ? step : step < 0 ? 0 : step; // clip at 0, NaN passes like np.clip
  });
}

/** Edges of any of `elements` inside [lo, hi], sorted by energy. */
export function interferingEdges(elements: Iterable<string>, lo: number, hi: number,
                                 exclude?: [string, string]): [string, string, number][] {
  const found: [string, string, number][] = [];
  for (const el of elements) {
    for (const [name, e] of Object.entries(elementEdges(el, lo, hi))) {
      if (exclude && el === exclude[0] && name === exclude[1]) continue;
      found.push([el, name, e]);
    }
  }
  return found.sort((a, b) => a[2] - b[2]);
}
