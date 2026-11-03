/** Energy grids and dwell times for an XAS scan. */
import { arange, sum, type Vec } from "./num";
import { EOfK, kOfE } from "./xray";

export interface ScanOptions {
  pre_start?: number;
  pre_step?: number;
  xanes_start?: number;
  xanes_end?: number;
  xanes_step?: number;
  kmax?: number;
  kstep?: number;
  dwell_pre?: number;
  dwell_xanes?: number;
  dwell_exafs?: number;
  k_time_weight?: number;
  i0_dwell_factor?: number;
}

/**
 * Three-region scan: pre-edge, XANES (energy steps), EXAFS (k steps). Energies
 * are relative to E0 in eV. EXAFS dwell grows as dwell_exafs (k / k_start)^k_time_weight.
 * i0_dwell_factor sets the I0 (blank) scan dwell relative to the sample scan.
 */
export class ScanPlan {
  readonly E0: number;
  readonly pre_start: number = -150;
  readonly pre_step: number = 5;
  readonly xanes_start: number = -30;
  readonly xanes_end: number = 40;
  readonly xanes_step: number = 0.5;
  readonly kmax: number = 12;
  readonly kstep: number = 0.05;
  readonly dwell_pre: number = 1;
  readonly dwell_xanes: number = 1;
  readonly dwell_exafs: number = 1;
  readonly k_time_weight: number = 0;
  readonly i0_dwell_factor: number = 1;
  private cache: { E: Vec; t: Vec } | null = null;

  constructor(E0: number, opts: ScanOptions = {}) {
    this.E0 = E0;
    Object.assign(this, Object.fromEntries(Object.entries(opts).filter(([, v]) => v !== undefined)));
  }

  /** Energies and dwell times. */
  grid(): { E: Vec; t: Vec } {
    if (this.cache) return this.cache;
    const pre = arange(this.pre_start, this.xanes_start, this.pre_step);
    const xan = arange(this.xanes_start, this.xanes_end, this.xanes_step);
    const k0 = kOfE(this.E0 + this.xanes_end, this.E0);
    const ks = this.kmax > k0 ? arange(k0, this.kmax + this.kstep / 2, this.kstep) : [];
    const E = [...pre.map((e) => this.E0 + e), ...xan.map((e) => this.E0 + e), ...ks.map((k) => EOfK(k, this.E0))];
    const t = [
      ...pre.map(() => this.dwell_pre),
      ...xan.map(() => this.dwell_xanes),
      ...ks.map((k) => this.dwell_exafs * (k / k0) ** this.k_time_weight),
    ];
    this.cache = { E, t };
    return this.cache;
  }

  get emin(): number {
    return this.E0 + this.pre_start;
  }

  get emax(): number {
    const { E } = this.grid();
    return E[E.length - 1];
  }

  /** [sample scan, I0 scan] counting time in s, excluding motor overhead. */
  totalTime(): [number, number] {
    const s = sum(this.grid().t);
    return [s, s * this.i0_dwell_factor];
  }
}
