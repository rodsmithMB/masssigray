/**
 * Counting time needed to reach a target rms noise of k^w chi(k).
 *
 * Two ways to spend more time are compared:
 * - longer dwells: every dwell (sample and I0) is multiplied by f and the full
 *   detector model re-evaluated. Source instability does not average out, so
 *   the noise approaches a floor and a low target may be unreachable.
 * - repeated scans: N scan pairs averaged; every noise term falls as 1/sqrt(N).
 * Times are counting time plus a fixed overhead per point, paid once per point per scan.
 */
import { brentq } from "./num";
import type { ScanPlan } from "./scan";

const F_MIN = 1e-4;
const F_MAX = 1e6;

export interface TimeEstimate {
  target: number;
  sigma_one: number; // rms noise of one scan pair at the planned dwells
  floor: number; // noise limit for very long dwells (source instability)
  time_one_s: number; // one sample scan + one I0 scan, including overhead
  dwell_factor: number | null; // multiply every dwell by this; null if unreachable
  time_dwell_s: number | null;
  n_scans: number; // scan pairs to average at the planned dwells
  time_scans_s: number;
}

/** A planner model that can report the rms chi noise with all dwells scaled by `factor`. */
export interface ChiModel {
  nPoints: number;
  chiRms(mS: number, mM: number, factor: number): number;
}

export function estimate(model: ChiModel, scan: ScanPlan, mS: number, mM: number, target: number,
                         overheadS = 0): TimeEstimate {
  if (!(target > 0)) throw new Error("target must be positive");
  const chi = (f: number) => model.chiRms(mS, mM, f);
  const sigmaOne = chi(1);
  if (!Number.isFinite(sigmaOne)) {
    throw new Error("no noise estimate: the k range is outside the scan or there is no signal");
  }
  const [tSample, tI0] = scan.totalTime();
  const perScanOverhead = 2 * model.nPoints * overheadS;
  const timeOne = tSample + tI0 + perScanOverhead;

  const nScans = Math.max(1, Math.ceil((sigmaOne / target) ** 2 - 1e-9));
  // sigma²(f) ~ a/f + floor² at long dwells; extrapolate from two points
  const f1 = F_MAX / 100;
  const f2 = F_MAX;
  const s1 = chi(f1);
  const s2 = chi(f2);
  const floor = Math.sqrt(Math.max(0, (f2 * s2 ** 2 - f1 * s1 ** 2) / (f2 - f1)));

  let f: number | null;
  if (s2 > target) f = null;
  else if (chi(F_MIN) <= target) f = F_MIN;
  else f = 10 ** brentq((lf) => chi(10 ** lf) - target, Math.log10(F_MIN), Math.log10(F_MAX), 1e-4);
  return {
    target, sigma_one: sigmaOne, floor, time_one_s: timeOne, dwell_factor: f,
    time_dwell_s: f === null ? null : f * (tSample + tI0) + perScanOverhead,
    n_scans: nScans, time_scans_s: nScans * timeOne,
  };
}
