/**
 * The parts of xraydb the planner uses, from tables exported by
 * scripts/export_web_data.py: Elam cross sections (cm²/g), absorption edges,
 * fluorescence lines, atomic masses and densities, and the formula parser.
 */
import db from "../data/xraydb.json";

type Spline = [number[], number[]]; // log values, spline second derivatives

interface ElementData {
  z: number;
  mass: number;
  density: number;
  edges: Record<string, [number, number]>; // edge: [energy eV, fluorescence yield]
  lines: Record<string, [number, number]>; // edge: [mean line energy eV, net probability]
  photo: [number[], number[], number[]]; // log energy, log cross section, spline
  scatter_lne: number[];
  coh: Spline;
  incoh: Spline;
}

const ELEMENTS = db.elements as unknown as Record<string, ElementData>;
export const DATA_SOURCE: string = db.meta.source;

export function element(sym: string): ElementData {
  const e = ELEMENTS[sym];
  if (!e) throw new Error(`'${sym}' is not an element symbol`);
  return e;
}

export const isElement = (sym: string): boolean => sym in ELEMENTS;
export const atomicNumber = (sym: string): number => element(sym).z;
export const atomicMass = (sym: string): number => element(sym).mass;
export const atomicDensity = (sym: string): number => element(sym).density;

/** Edges in xraydb order: {edge: energy eV}. */
export function xrayEdges(sym: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [name, [e]] of Object.entries(element(sym).edges)) out[name] = e;
  return out;
}

export function edgeData(sym: string, edge: string): { energy: number; fyield: number } {
  const d = element(sym).edges[edge];
  if (!d) throw new Error(`no ${edge} edge for ${sym}`);
  return { energy: d[0], fyield: d[1] };
}

/** Main fluorescence line of an edge: (mean energy, net probability), or null. */
export function fluorLine(sym: string, edge: string): { energy: number; prob: number } | null {
  const d = element(sym).lines[edge];
  return d ? { energy: d[0], prob: d[1] } : null;
}

/** Elam spline interpolation (xraydb.utils.elam_spline) at one log-energy. */
function elamSpline(xin: number[], yin: number[], yspl: number[], x: number): number {
  const n = xin.length;
  // lo: last index with xin < x (else 0); hi: first index with xin > x (else n - 1)
  let lo = 0;
  let a = 0;
  let b = n;
  while (a < b) {
    const mid = (a + b) >> 1;
    if (xin[mid] < x) a = mid + 1;
    else b = mid;
  }
  lo = a > 0 ? a - 1 : 0;
  a = 0;
  b = n;
  while (a < b) {
    const mid = (a + b) >> 1;
    if (xin[mid] <= x) a = mid + 1;
    else b = mid;
  }
  const hi = a < n ? a : n - 1;
  const diff = xin[hi] - xin[lo];
  const A = (xin[hi] - x) / diff;
  const B = (x - xin[lo]) / diff;
  return A * yin[lo] + B * yin[hi] +
    ((diff * diff) / 6) * ((A * A - 1) * A * yspl[lo] + (B * B - 1) * B * yspl[hi]);
}

export type XsKind = "total" | "photo";

/** Mass attenuation coefficient of an element, cm²/g (xraydb.mu_elam). */
export function muElam(sym: string, E: number[], kind: XsKind = "total"): number[] {
  const el = element(sym);
  const [plne, pval, pspl] = el.photo;
  return E.map((e) => {
    const lx = Math.log(Math.min(Math.max(e, 100.0), 8e5));
    let mu = Math.exp(elamSpline(plne, pval, pspl, lx));
    if (kind === "total") {
      mu += Math.exp(elamSpline(el.scatter_lne, el.coh[0], el.coh[1], lx));
      mu += Math.exp(elamSpline(el.scatter_lne, el.incoh[0], el.incoh[1], lx));
    }
    return mu;
  });
}

// ------------------------------------------------------------------ formula parser (xraydb.chemparse)

const PARSER_ALIASES: Record<string, string> = { D: "H" };

/** {element: moles} for a formula; parentheses, decimals and exponents allowed. */
export function chemparse(formula: string): Map<string, number> {
  // numbers starting with '.' get a 0: 'Fe.7Mg.3O' -> 'Fe0.7Mg0.3O'
  let f = "";
  for (let i = 0; i < formula.length; i++) {
    const ch = formula[i];
    if (ch === "." && !/[0-9]/.test(formula[i - 1] ?? "")) f += "0";
    f += ch;
  }
  f = f.replace(/ /g, "");
  const lexer = /[A-Z][a-z]*|[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?|[()]/y;
  let pos = 0;
  let tok: string | null = null;
  const next = () => {
    if (pos >= f.length) {
      tok = null;
      return;
    }
    lexer.lastIndex = pos;
    const m = lexer.exec(f);
    if (!m) throw new Error(`unrecognized element or number at '${f.slice(pos)}'`);
    pos = lexer.lastIndex;
    tok = m[0];
  };
  type Node = { items: (Node | string)[]; count: number };
  const isNum = (t: string | null) => t !== null && t[0] >= "0" && t[0] <= "9";
  const isName = (t: string | null) => t !== null && t[0] >= "A" && t[0] <= "Z";
  const sequence = (): Node => {
    const seq: Node = { items: [], count: 1 };
    while (tok === "(" || isName(tok)) {
      let thing: Node;
      if (tok === "(") {
        next();
        thing = sequence();
        if ((tok as string | null) !== ")") throw new Error("expected right paren");
        next();
      } else {
        const sym = PARSER_ALIASES[tok!] ?? tok!;
        if (!isElement(sym)) throw new Error(`'${tok}' is not an element symbol`);
        thing = { items: [sym], count: 1 };
        next();
      }
      if (isNum(tok)) {
        thing.count = Number(tok);
        next();
      }
      seq.items.push(thing);
    }
    return seq;
  };
  next();
  const root = sequence();
  if (tok !== null) throw new Error("expected end of input");
  const out = new Map<string, number>();
  const add = (node: Node, weight: number) => {
    const w = weight * node.count;
    for (const it of node.items) {
      if (typeof it === "string") out.set(it, (out.get(it) ?? 0) + w);
      else add(it, w);
    }
  };
  add(root, 1);
  return out;
}
