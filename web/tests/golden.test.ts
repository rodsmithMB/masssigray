/**
 * The TypeScript port against reference results from the frozen Python code
 * (tests/golden.json, written by scripts/make_golden.py).
 */
import { describe, expect, it } from "vitest";
import { calibrate, DetectorModel, type Scan } from "../src/core/detector";
import * as fl from "../src/core/fluorescence";
import { FluxSpectrum, instrumentFlux, readTable } from "../src/core/flux";
import { Material, matrixPreset, molarMass, parseFormula } from "../src/core/materials";
import { ScanPlan } from "../src/core/scan";
import { loadReference, modelMut, normalize } from "../src/core/simulate";
import type { TimeEstimate } from "../src/core/timing";
import {
  dilutionRecipe, evaluateMasses, foilSetup, optimize, pelletConstraint, timeForTarget, TransmissionSetup,
  type PelletConstraint,
} from "../src/core/transmission";
import { edgeEnergy, edgeStepMuRho, elementEdges, muRho } from "../src/core/xray";
import goldenRaw from "./golden.json";

/* eslint-disable @typescript-eslint/no-explicit-any */
const golden = goldenRaw as any;

const decode = (v: unknown): number | null =>
  v === null ? null : v === "nan" ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : (v as number);
const dec = (a: unknown[]): number[] => a.map((v) => decode(v) as number);

/** Relative closeness, treating NaN == NaN and inf == inf. */
function close(a: number | null, b: number | null, rel = 1e-9, abs = 1e-12): boolean {
  if (a === null || b === null) return a === b;
  if (Number.isNaN(a) || Number.isNaN(b)) return Number.isNaN(a) && Number.isNaN(b);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  return Math.abs(a - b) <= Math.max(abs, rel * Math.max(Math.abs(a), Math.abs(b)));
}

function expectClose(actual: number | null, expected: unknown, label: string, rel = 1e-9, abs = 1e-12) {
  const e = decode(expected);
  if (!close(actual, e, rel, abs)) {
    expect.fail(`${label}: got ${actual}, expected ${e} (rel ${rel})`);
  }
}

function expectArray(actual: number[], expected: unknown[], label: string, rel = 1e-9, abs = 1e-12) {
  expect(actual.length, `${label} length`).toBe(expected.length);
  const e = dec(expected);
  for (let i = 0; i < e.length; i++) {
    if (!close(actual[i], e[i], rel, abs)) {
      expect.fail(`${label}[${i}]: got ${actual[i]}, expected ${e[i]}`);
    }
  }
}

// ------------------------------------------------------------------ builders mirroring make_golden.py

function material(spec: any): Material | null {
  if (!spec) return null;
  if (spec.preset) return matrixPreset(spec.preset);
  return Material.fromText(spec.text, spec.density ?? null, spec.name ?? "");
}

function flux(spec: any): FluxSpectrum {
  if (spec.kind === "constant") return FluxSpectrum.constant(spec.rate);
  return instrumentFlux(spec.window ?? null, spec.dwell ?? 1, spec.gain ?? 1);
}

function detector(spec: any): DetectorModel {
  const s = spec ?? {};
  return new DetectorModel(s.read_var ?? 0, s.dark_rate ?? 0, s.instability ?? 0, s.frame_time ?? null,
                           s.max_rate ?? null);
}

const constraint = (spec: any): PelletConstraint => pelletConstraint(spec ?? {});

function tsetup(spec: any): TransmissionSetup {
  const E0 = edgeEnergy(spec.absorber, spec.edge);
  return new TransmissionSetup({
    sample: material(spec.sample)!, matrix: material(spec.matrix), area_cm2: spec.area,
    absorber: spec.absorber, edge: spec.edge, flux: flux(spec.flux), detector: detector(spec.detector),
    scan: new ScanPlan(E0, spec.scan ?? {}), objective: spec.objective ?? "edge",
    k_range: spec.k_range ?? [3, 10], k_weight: spec.k_weight ?? 2,
  });
}

function fsetup(spec: any): fl.FluorescenceSetup {
  const E0 = edgeEnergy(spec.absorber, spec.edge);
  const f = spec.fluo ?? {};
  return new fl.FluorescenceSetup({
    sample: material(spec.sample)!, matrix: material(spec.matrix), area_cm2: spec.area,
    absorber: spec.absorber, edge: spec.edge, flux: flux(spec.flux), scan: new ScanPlan(E0, spec.scan ?? {}),
    theta_in: f.theta_in, theta_out: f.theta_out, solid_angle_frac: f.omega, det_efficiency: f.eff,
    flux_scale: f.flux_scale, background_per_photon: f.bg, i0_detector: detector(spec.detector),
    objective: spec.objective ?? "exafs", k_range: spec.k_range ?? [3, 10], k_weight: spec.k_weight ?? 2,
    min_retention: f.min_retention, max_total_mass_mg: f.max_mass,
  });
}

const PLAN_SCALARS = ["m_sample_mg", "m_matrix_mg", "edge_step", "mut_below", "mut_above", "mut_max",
                      "snr_edge", "chi_noise_rms", "absorber_mass_mg", "thickness_mm"];

function comparePlan(p: any, g: any, label: string, rel = 1e-9) {
  for (const k of PLAN_SCALARS) expectClose(p[k], g[k], `${label} ${k}`, rel);
  expect(p.binding, `${label} binding`).toBe(g.binding);
  expect(p.feasible, `${label} feasible`).toBe(g.feasible);
  expect(p.warnings, `${label} warnings`).toEqual(g.warnings);
  if (g.dilution) {
    for (const [k, v] of Object.entries(g.dilution)) {
      if (typeof v === "number" || typeof v === "string" && /nan|inf/.test(v)) {
        expectClose(p.dilution[k], v, `${label} dilution ${k}`, rel);
      } else {
        expect(p.dilution[k]).toEqual(v);
      }
    }
  } else {
    expect(p.dilution).toBeNull();
  }
  for (const [k, v] of Object.entries(g.spectra)) expectArray(p.spectra[k], v as unknown[], `${label} spectra.${k}`, rel);
}

function compareTime(t: TimeEstimate, g: any, label: string) {
  expectClose(t.sigma_one, g.sigma_one, `${label} sigma_one`);
  expectClose(t.floor, g.floor, `${label} floor`, 1e-6, 1e-9);
  expectClose(t.time_one_s, g.time_one_s, `${label} time_one_s`);
  expect(t.n_scans, `${label} n_scans`).toBe(g.n_scans);
  expectClose(t.time_scans_s, g.time_scans_s, `${label} time_scans_s`);
  // brentq stops within xtol = 1e-4 in log10(f): about 2.3e-4 relative in f
  expectClose(t.dwell_factor, g.dwell_factor, `${label} dwell_factor`, 1e-3);
  expectClose(t.time_dwell_s, g.time_dwell_s, `${label} time_dwell_s`, 1e-3);
}

// ------------------------------------------------------------------ tests

describe("materials and cross sections", () => {
  it("parses formulas like xraydb.chemparse", () => {
    for (const g of golden.formulas) {
      const comp = parseFormula(g.formula);
      expect(Object.fromEntries(comp), g.formula).toEqual(
        Object.fromEntries(Object.entries(g.comp).filter(([, n]) => (n as number) > 0)));
      expectClose(molarMass(g.formula), g.molar_mass, `${g.formula} molar mass`, 1e-12);
    }
  });

  it("builds mixtures", () => {
    for (const g of golden.mixtures) {
      const m = Material.fromText(g.text);
      expect(m.elements).toEqual(g.elements);
      expect(m.label).toBe(g.label);
      for (const [el, w] of Object.entries(g.fractions)) expectClose(m.massFraction(el), w, `${g.text} ${el}`, 1e-12);
    }
  });

  it("matches Elam cross sections", () => {
    const E = dec(golden.energies);
    for (const g of golden.mu_rho) {
      const m = Material.fromText(g.text);
      expectArray(muRho(m, E), g.total, `${g.text} total`, 1e-11);
      expectArray(muRho(m, E, "photo"), g.photo, `${g.text} photo`, 1e-11);
    }
  });

  it("matches edge steps and edge lists", () => {
    const E = dec(golden.energies);
    for (const g of golden.edge_step) {
      expectArray(edgeStepMuRho(g.element, g.edge, E), g.values, `${g.element} ${g.edge}`, 1e-9, 1e-9);
    }
    for (const [el, edges] of Object.entries(golden.edges)) {
      expect(elementEdges(el, 1000, 120000)).toEqual(edges);
    }
  });
});

describe("transmission", () => {
  for (const g of golden.transmission) {
    it(g.spec.name, () => {
      const s = tsetup(g.spec);
      const c = constraint(g.spec.constraint);
      const p = optimize(s, c);
      comparePlan(p, g.plan, "optimize");
      const curve = p.curve as any;
      for (const [k, v] of Object.entries(g.plan.curve)) {
        if (k === "feasible") expect(curve.feasible).toEqual((v as number[]).map((x) => x === 1));
        else expectArray(curve[k], v as unknown[], `curve.${k}`);
      }
      comparePlan(evaluateMasses(s, g.evaluate.m_s, g.evaluate.m_m, c), g.evaluate.plan, "evaluate");
      if (g.time.error) {
        expect(() => timeForTarget(s, p.m_sample_mg, p.m_matrix_mg, g.time.target, g.time.overhead))
          .toThrow(g.time.error);
      } else {
        compareTime(timeForTarget(s, p.m_sample_mg, p.m_matrix_mg, g.time.target, g.time.overhead),
                    g.time.result, "time");
      }
    });
  }

  it("foil", () => {
    const s = foilSetup(Material.fromFormula("Cu", 8.96), "Cu", "K", FluxSpectrum.constant(1e5));
    comparePlan(optimize(s, pelletConstraint({ mode: "none" })), golden.foil.plan, "foil");
  });

  it("dilution recipes", () => {
    for (const g of golden.dilution) {
      const [ms, mm, w] = g.args;
      const r = dilutionRecipe(ms, mm, w);
      if (g.result === null) expect(r).toBeNull();
      else for (const [k, v] of Object.entries(g.result)) {
        if (typeof v === "number") expectClose((r as any)[k], v, k);
        else expect((r as any)[k]).toEqual(v);
      }
    }
  });
});

describe("fluorescence", () => {
  for (const g of golden.fluorescence) {
    it(g.spec.name, () => {
      const s = fsetup(g.spec);
      const p = fl.optimize(s, constraint(g.spec.constraint));
      const gp = g.plan;
      for (const k of ["m_sample_mg", "m_matrix_mg", "retention_ref", "retention_min_k", "fluo_rate_ref",
                       "snr_edge", "chi_noise_rms", "line_energy", "thickness_mm"]) {
        expectClose((p as any)[k], gp[k], k);
      }
      expect(p.binding).toBe(gp.binding);
      // element order in Python's set iteration is arbitrary: compare as sets
      expect([...p.warnings].sort()).toEqual([...gp.warnings].sort());
      for (const [k, v] of Object.entries(gp.curve)) expectArray((p.curve as any)[k], v as unknown[], `curve.${k}`);
      for (const [k, v] of Object.entries(gp.spectra)) expectArray((p.spectra as any)[k], v as unknown[], `spectra.${k}`);
      if (g.time.error) {
        expect(() => fl.timeForTarget(s, p.m_sample_mg, p.m_matrix_mg, g.time.target, g.time.overhead)).toThrow();
      } else {
        compareTime(fl.timeForTarget(s, p.m_sample_mg, p.m_matrix_mg, g.time.target, g.time.overhead),
                    g.time.result, "time");
      }
    });
  }
});

describe("detector calibration", () => {
  const scans: Scan[] = golden.calibration.scans.map((s: any) => ({
    dwell: s.dwell, dark: s.dark, energy: dec(s.energy), values: dec(s.values),
  }));

  it("fits the photon-transfer model like scipy nnls", () => {
    const summary = calibrate(scans).summary();
    // six reweighted nnls passes amplify rounding differences to ~1e-7
    for (const [k, v] of Object.entries(golden.calibration.summary)) expectClose(summary[k], v, k, 1e-6, 1e-9);
  });

  it("handles a single dwell time", () => {
    const summary = calibrate(scans.filter((s) => s.dwell === 1 && !s.dark)).summary();
    for (const [k, v] of Object.entries(golden.calibration_one_dwell)) expectClose(summary[k], v, k, 1e-6, 1e-9);
  });
});

describe("reference spectra and tables", () => {
  const g = golden.simulate;

  it("normalizes like the Python version", () => {
    const rows = readTable(g.text);
    const { e0, norm } = normalize(rows.column(0), rows.column(1));
    expectClose(e0, g.e0, "e0");
    // np.polyfit on raw energies (~9 keV) loses ~1e-8; the port fits in scaled x
    expectArray(norm, g.norm, "norm", 1e-6, 1e-6);
    const ref = loadReference(g.text);
    expectArray(ref.E, g.ref_E, "ref E");
    expectArray(ref.norm, g.ref_norm, "ref norm", 1e-6, 1e-6);
  });

  it("shapes the model spectrum", () => {
    const t = golden.transmission[0];
    const p = optimize(tsetup(t.spec), constraint(t.spec.constraint));
    const ref = loadReference(g.text);
    expectArray(modelMut(p.spectra, edgeEnergy("Cu", "K"), ref, 2), g.model_shift2, "model", 1e-6, 1e-6);
  });

  it("reads tables with headers and mixed separators", () => {
    const t = readTable("# energy, rate, extra\n1 2 3\n4,5,6\nbad line\n7\t8\t9\n");
    expect(t.columns).toEqual(golden.read_table.columns);
    expect(t.data).toEqual(golden.read_table.values);
  });
});
