/**
 * Predicted spectra: tabulated absorption, optionally shaped by a reference
 * XANES/EXAFS, with counting noise. A reference spectrum of a similar compound
 * is normalized (flattened), aligned to the tabulated edge and scaled by the
 * predicted edge step, so white-line absorbance becomes realistic.
 */
import { sigmaMut, type DetectorModel } from "./detector";
import { readTable, type EnergyUnit } from "./flux";
import { argmax, argsort, gradient, interp, polyfit, Rng, type Vec } from "./num";

/** Flattened normalization: 0 before the edge, ~1 (plus oscillations) after. */
export function normalize(E: Vec, mu: Vec, e0: number | null = null, pre: [number, number] = [-150, -30],
                          post: [number, number | null] = [50, null]): { e0: number; norm: Vec } {
  const edge = e0 ?? E[argmax(gradient(mu, E))];
  const sel = (f: (e: number, i: number) => boolean) => E.flatMap((e, i) => (f(e, i) ? [i] : []));
  let pm = sel((e) => e >= edge + pre[0] && e <= edge + pre[1]);
  if (pm.length < 3) pm = sel((e) => e <= edge - 10);
  if (pm.length < 2) pm = sel((_, i) => i < 3);
  const preFit = polyfit(pm.map((i) => E[i]), pm.map((i) => mu[i]), 1);
  const hi = post[1] ?? E[E.length - 1] - edge;
  let qm = sel((e) => e >= edge + post[0] && e <= edge + hi);
  if (qm.length < 4) qm = sel((e) => e >= edge + 10);
  const postFit = polyfit(qm.map((i) => E[i]), qm.map((i) => mu[i]), qm.length > 6 ? 2 : 1);
  const jump0 = postFit(edge) - preFit(edge);
  const norm = E.map((e, i) => {
    const preLine = preFit(e);
    return e >= edge ? (mu[i] - preLine) / (postFit(e) - preLine) : (mu[i] - preLine) / jump0;
  });
  return { e0: edge, norm };
}

export interface Reference {
  E: Vec; // relative to the reference's edge
  norm: Vec;
}

/** Load a reference spectrum (energy, mu or I/I0) and return it normalized, relative to its edge. */
export function loadReference(text: string, energyCol = 0, valueCol = 1, energyUnit: EnergyUnit = "auto",
                              normalized = false, kind: "mu" | "transmission" = "mu"): Reference {
  const t = readTable(text);
  let E = t.column(energyCol);
  let y = t.column(valueCol);
  if (energyUnit === "keV" || (energyUnit === "auto" && Math.max(...E) < 200)) E = E.map((e) => e * 1000);
  const order = argsort(E);
  E = order.map((i) => E[i]);
  y = order.map((i) => y[i]);
  if (kind === "transmission") y = y.map((v) => -Math.log(Math.max(v, 1e-12)));
  if (normalized) {
    const e0 = E[argmax(gradient(y, E))];
    return { E: E.map((e) => e - e0), norm: y };
  }
  const { e0, norm } = normalize(E, y);
  return { E: E.map((e) => e - e0), norm };
}

/** mu t(E) on the scan grid, with the edge step shaped by `reference` (bare tables if null). */
export function modelMut(spectra: { E: Vec; mut: Vec; step: Vec }, E0: number, reference: Reference | null = null,
                         shift = 0): Vec {
  const { E, mut, step } = spectra;
  if (!reference) return [...mut];
  const above = E.flatMap((e, i) => (e >= E0 ? [i] : []));
  if (!above.length) return [...mut];
  const Ea = above.map((i) => E[i]);
  const Sa = above.map((i) => step[i]);
  const stepC = interp(E, Ea, Sa, Sa[0]);
  const shape = interp(E.map((e) => e - E0 - shift), reference.E, reference.norm, 0,
                       reference.norm[reference.norm.length - 1]);
  return mut.map((m, i) => m - step[i] + stepC[i] * shape[i]);
}

/** Simulated measurement: returns [noisy mu t, sigma] for the shaped model. */
export function addNoise(mutModel: Vec, spectra: { rate: Vec; dwell: Vec }, det: DetectorModel,
                         i0DwellFactor = 1, seed = 1): { noisy: Vec; sigma: Vec } {
  const rng = new Rng(seed);
  const sigma = mutModel.map((m, i) =>
    sigmaMut(spectra.rate[i], m, spectra.dwell[i], spectra.dwell[i] * i0DwellFactor, det));
  const noisy = mutModel.map((m, i) => m + rng.normal() * (Number.isFinite(sigma[i]) ? sigma[i] : 0));
  return { noisy, sigma };
}
