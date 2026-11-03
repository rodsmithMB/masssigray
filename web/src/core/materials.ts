/**
 * Materials: chemical formulas, weight-percent mixtures, densities, presets.
 *
 * Every material is reduced to element mass fractions, which is all the
 * absorption calculations need. A material is a single formula ("Fe2O3",
 * "Ni0.5Fe0.5(OH)2") or a weight-percent mixture ("Pt:5, Al2O3:95").
 */
import { fmtG, parseFloatPy } from "./num";
import { atomicMass, atomicNumber, chemparse } from "./xraydb";

/** Matrix presets: [formula, nominal bulk density g/cm³]. Pressed pellets are less dense. */
export const MATRIX_PRESETS: Record<string, [string, number]> = {
  "Boron nitride (BN)": ["BN", 2.1],
  Cellulose: ["C6H10O5", 1.5],
  "Polyethylene (PE)": ["C2H4", 0.94],
  Graphite: ["C", 2.2],
  Sucrose: ["C12H22O11", 1.59],
  PVP: ["C6H9NO", 1.2],
  Starch: ["C6H10O5", 1.5],
  "Lithium carbonate": ["Li2CO3", 2.11],
  "Silica (SiO2)": ["SiO2", 2.2],
  "Alumina (Al2O3)": ["Al2O3", 3.95],
};

/** {element: moles}; parentheses and decimals allowed. */
export function parseFormula(formula: string): Map<string, number> {
  const f = formula.trim();
  if (!f) throw new Error("empty formula");
  let comp: Map<string, number>;
  try {
    comp = chemparse(f);
  } catch (err) {
    throw new Error(`cannot parse formula '${f}': ${(err as Error).message}`);
  }
  if (!comp.size) throw new Error(`cannot parse formula '${f}'`);
  return new Map([...comp].filter(([, n]) => n > 0));
}

/** Molar mass in g/mol. */
export function molarMass(formula: string): number {
  let m = 0;
  for (const [el, n] of parseFormula(formula)) m += n * atomicMass(el);
  return m;
}

export function formulaMassFractions(formula: string): Map<string, number> {
  const masses = new Map<string, number>();
  for (const [el, n] of parseFormula(formula)) masses.set(el, n * atomicMass(el));
  let total = 0;
  for (const m of masses.values()) total += m;
  return new Map([...masses].map(([el, m]) => [el, m / total]));
}

/**
 * Parse "Pt:5, Al2O3:95" (wt%) into {formula: wt%}. A single component without
 * a weight gets 100 wt%; one component may omit its weight and takes the balance.
 */
export function parseMixture(text: string): Map<string, number> {
  const parts = text.replace(/;/g, ",").split(",").map((p) => p.trim()).filter(Boolean);
  if (!parts.length) throw new Error("empty composition");
  const comps = new Map<string, number | null>();
  for (const p of parts) {
    const i = p.indexOf(":");
    if (i >= 0) {
      const w = parseFloatPy(p.slice(i + 1).trim().replace(/%+$/, ""));
      if (w === null) throw new Error(`could not convert '${p.slice(i + 1).trim()}' to a weight`);
      comps.set(p.slice(0, i).trim(), w);
    } else {
      comps.set(p, null);
    }
  }
  const missing = [...comps].filter(([, w]) => w === null).map(([f]) => f);
  if (missing.length > 1) throw new Error("only one component may omit its wt% (the balance)");
  let given = 0;
  for (const w of comps.values()) if (w !== null) given += w;
  if (missing.length) {
    if (given > 100) throw new Error("weights exceed 100 wt%");
    comps.set(missing[0], 100 - given);
  }
  return comps as Map<string, number>;
}

/** A material defined by wt% of formula components, with optional density (g/cm³). */
export class Material {
  readonly components: Map<string, number>;
  readonly density: number | null;
  readonly name: string;
  private readonly wfrac: Map<string, number>;

  constructor(components: Map<string, number>, density: number | null = null, name = "") {
    if (!components.size) throw new Error("material has no components");
    let total = 0;
    for (const w of components.values()) total += w;
    if (total <= 0) throw new Error("component weights must be positive");
    const wf = new Map<string, number>();
    for (const [formula, w] of components) {
      for (const [el, f] of formulaMassFractions(formula)) wf.set(el, (wf.get(el) ?? 0) + (f * w) / total);
    }
    this.components = components;
    this.density = density || null;
    this.wfrac = wf;
    this.name = name || this.label;
  }

  static fromFormula(formula: string, density: number | null = null, name = ""): Material {
    return new Material(new Map([[formula, 100]]), density, name);
  }

  /** Formula or wt% mixture text, e.g. "CuO" or "Pt:5, Al2O3". */
  static fromText(text: string, density: number | null = null, name = ""): Material {
    return new Material(parseMixture(text), density, name);
  }

  get label(): string {
    if (this.components.size === 1) return [...this.components.keys()][0];
    return [...this.components].map(([f, w]) => `${fmtG(w)}% ${f}`).join(" + ");
  }

  get massFractions(): Map<string, number> {
    return new Map(this.wfrac);
  }

  massFraction(el: string): number {
    return this.wfrac.get(el) ?? 0;
  }

  /** Elements sorted by atomic number. */
  get elements(): string[] {
    return [...this.wfrac.keys()].sort((a, b) => atomicNumber(a) - atomicNumber(b));
  }
}

export function matrixPreset(name: string): Material {
  const [formula, rho] = MATRIX_PRESETS[name];
  return Material.fromFormula(formula, rho, name);
}
