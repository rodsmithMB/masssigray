/**
 * Measured incident flux, I0(E): the count rate with no sample in the beam, in
 * effective counts per second (detector units divided by the calibrated gain),
 * so that Poisson statistics apply directly.
 */
import referenceCsv from "../data/i0_reference.csv?raw";
import smoothCsv from "../data/i0_smooth.csv?raw";
import { argsort, arange, interp, parseFloatPy, type Vec } from "./num";

export class FluxSpectrum {
  readonly energy: Vec; // eV, increasing
  readonly rate: Vec; // effective counts / s
  readonly label: string;

  constructor(energy: Vec, rate: Vec, label = "") {
    const order = argsort(energy);
    this.energy = order.map((i) => energy[i]);
    this.rate = order.map((i) => Math.max(rate[i], 0));
    this.label = label;
  }

  static constant(rate: number, emin = 1000, emax = 40000): FluxSpectrum {
    return new FluxSpectrum([emin, emax], [rate, rate], `constant ${rate} /s`);
  }

  /** Interpolated rate; zero outside the measured range. */
  at(E: Vec): Vec {
    return interp(E, this.energy, this.rate, 0, 0);
  }

  covers(emin: number, emax: number): boolean {
    return this.energy[0] <= emin && emax <= this.energy[this.energy.length - 1];
  }

  scaled(factor: number): FluxSpectrum {
    return new FluxSpectrum(this.energy, this.rate.map((r) => r * factor), this.label);
  }

  get emin(): number {
    return this.energy[0];
  }

  get emax(): number {
    return this.energy[this.energy.length - 1];
  }
}

export interface Table {
  columns: string[];
  data: number[][]; // rows
  column(i: number): Vec;
}

/**
 * Read a whitespace/comma/tab/semicolon separated numeric table, skipping text
 * lines. Column names come from the last text line before the data if it has
 * the right number of fields, otherwise col0, col1, ...
 */
export function readTable(text: string): Table {
  let header: string[] | null = null;
  const rows: number[][] = [];
  for (const line of text.split(/\r\n|\r|\n/)) {
    const s = line.trim();
    if (!s) continue;
    const fields = s.replace(/[,;]/g, " ").split(/\s+/);
    const vals = fields.map(parseFloatPy);
    if (vals.every((v) => v !== null)) {
      rows.push(vals as number[]);
    } else if (!rows.length) {
      header = s.replace(/^#+/, "").replace(/[,;]/g, " ").split(/\s+/).filter(Boolean);
    }
  }
  if (!rows.length) throw new Error("no numeric data found");
  const width = Math.min(...rows.map((r) => r.length));
  const data = rows.map((r) => r.slice(0, width));
  const columns = header && header.length === width ? header : data[0].map((_, i) => `col${i}`);
  return { columns, data, column: (i: number) => data.map((r) => r[i]) };
}

export type ValueKind = "rate" | "counts";
export type EnergyUnit = "auto" | "eV" | "keV";

/** Load an I0 spectrum from table text. */
export function loadFlux(text: string, energyCol = 0, valueCol = 1, valueKind: ValueKind = "rate",
                         dwell: number | null = null, energyUnit: EnergyUnit = "auto", gain = 1,
                         label = ""): FluxSpectrum {
  const t = readTable(text);
  let E = t.column(energyCol);
  let v = t.column(valueCol);
  if (energyUnit === "keV" || (energyUnit === "auto" && Math.max(...E) < 200)) E = E.map((e) => e * 1000);
  if (valueKind === "counts") {
    if (!dwell) throw new Error("dwell time is needed to convert counts to a rate");
    v = v.map((x) => x / dwell);
  }
  return new FluxSpectrum(E, v.map((x) => x / gain), label || "measured I0");
}

// ------------------------------------------------------------------ spectrometer exports

const REF_PIXEL = /(?:^|\s)R\s+\d+$/; // "Algn R 1" ... "Algn R 100"; not "R I0"

/**
 * Energy (eV) and reference signal (ADU per point) from a spectrometer CSV export:
 * the sum of the per-pixel reference columns "Algn R 1" ... "Algn R n". Energy is
 * "Calibrated Energy" (else "Energy"). Rows with missing pixels or a zero sum are dropped.
 */
export function readReferenceScan(text: string): { energy: Vec; adu: Vec } {
  const lines = text.split(/\r\n|\r|\n/).filter((l) => l.length);
  const header = lines[0].split(",").map((h) => h.trim());
  const pix = header.flatMap((h, i) => (REF_PIXEL.test(h) ? [i] : []));
  if (!pix.length) throw new Error("no reference pixel columns ('Algn R 1', ...) in the header");
  const ecol = ["Calibrated Energy", "Energy"].map((n) => header.indexOf(n)).find((i) => i >= 0);
  if (ecol === undefined) throw new Error("no 'Calibrated Energy' or 'Energy' column in the header");
  const last = Math.max(...pix);
  const energy: Vec = [];
  const adu: Vec = [];
  for (const line of lines.slice(1)) {
    const r = line.split(",");
    if (r.length <= last || pix.some((i) => !r[i].trim())) continue;
    let s = 0;
    for (const i of pix) s += Number(r[i]);
    if (s > 0) {
      energy.push(Number(r[ecol]));
      adu.push(s);
    }
  }
  if (!energy.length) throw new Error("no complete reference rows");
  return { energy, adu };
}

/** I0 from a spectrometer export: summed reference pixels / dwell / gain. */
export function loadReferenceScan(text: string, dwell = 1, gain = 1, label = ""): FluxSpectrum {
  const { energy, adu } = readReferenceScan(text);
  return new FluxSpectrum(energy, adu.map((a) => a / dwell / gain), label || "reference scan");
}

// ------------------------------------------------------------------ built-in instrument I0

function dataRows(text: string): string[][] {
  return text.split(/\r?\n/).filter((l) => l && !l.startsWith("#")).slice(1).map((l) => l.split(","));
}

let segmentsCache: Map<string, { energy: Vec; adu: Vec }> | null = null;

/** Measured windows of the lab spectrometer, ordered by energy: {name: (energy, ADU per point)}. */
export function instrumentSegments(): Map<string, { energy: Vec; adu: Vec }> {
  if (segmentsCache) return segmentsCache;
  const out = new Map<string, { energy: Vec; adu: Vec }>();
  for (const [name, e, a] of dataRows(referenceCsv)) {
    if (!out.has(name)) out.set(name, { energy: [], adu: [] });
    out.get(name)!.energy.push(Number(e));
    out.get(name)!.adu.push(Number(a));
  }
  segmentsCache = new Map([...out].sort((x, y) => x[1].energy[0] - y[1].energy[0]));
  return segmentsCache;
}

/** Smoothed full-span I0 (energy, ADU per point). */
export function instrumentSmooth(): { energy: Vec; adu: Vec } {
  const rows = dataRows(smoothCsv);
  return { energy: rows.map((r) => Number(r[0])), adu: rows.map((r) => Number(r[1])) };
}

/** Name of the measured window that contains E0, or null. */
export function instrumentWindow(E0: number): string | null {
  for (const [name, { energy }] of instrumentSegments()) {
    if (energy[0] <= E0 && E0 <= energy[energy.length - 1]) return name;
  }
  return null;
}

/** Built-in I0: the smoothed curve (window null) or one measured window. */
export function instrumentFlux(window: string | null = null, dwell = 1, gain = 1): FluxSpectrum {
  const seg = window === null ? instrumentSmooth() : instrumentSegments().get(window);
  if (!seg) throw new Error(`no I0 window '${window}'`);
  const label = window === null ? "instrument I0, smoothed" : `instrument I0, ${window} window`;
  return new FluxSpectrum(seg.energy, seg.adu.map((a) => a / dwell / gain), label);
}

/** A synthetic I0 file for demos (CSV text, eV and counts/s). */
export function exampleFluxCsv(emin = 4000, emax = 25000, step = 10, peak = 2e5): string {
  const E = arange(emin, emax + step, step);
  let shape = E.map((e) => ((emax * 1.2 - e) / e) * Math.exp(-((4500 / e) ** 3)));
  const top = Math.max(...shape);
  shape = shape.map((s, i) => (s / top) * (1 - 0.35 * Math.exp(-0.5 * ((E[i] - 11000) / 300) ** 2)));
  return "# energy_eV, rate_cps\n" + E.map((e, i) => `${e.toFixed(1)}, ${(peak * shape[i]).toFixed(2)}`).join("\n") + "\n";
}
