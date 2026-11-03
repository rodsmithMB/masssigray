import { describe, expect, it } from "vitest";
import cuExport from "../../source/Cu.csv?raw";
import { calibrate, DetectorModel, sigmaMut, simulateScans } from "../src/core/detector";
import {
  exampleFluxCsv, instrumentFlux, instrumentSegments, loadFlux, readReferenceScan,
} from "../src/core/flux";
import { brentq, fmtFixed, fmtG, nnls, parseFloatPy } from "../src/core/num";
import { edgeEnergy } from "../src/core/xray";

const exportText = (rows: string[]) => [
  "Number, Bragg Angle, Energy, Calibrated Angle, Calibrated Energy, Aln. Sum Reference, " +
  "Aln. Sum Signal, Absorption, Algn R 1, Algn R 2, Algn R 3, Algn S 1, Algn S 2, R I0, S I0",
  ...rows,
].join("\n");

describe("instrument I0", () => {
  it("sums the reference pixels of an export", () => {
    const text = exportText([
      "1, 30, 9000, 30, 9000.5, 0, 0, 0,,,,,, 0, 0",
      "2, 30, 9001, 30, 9001.5, 6, 9, 0.1, 1, 2, 3, 4, 5, 0, 0",
      "3, 30, 9002, 30, 9002.5, 15, 9, 0.1, 4, 5, 6, 4, 5, 0, 0",
      "4, 30, 9003, 30, 9003.5, 0, 0, 0,,,,,, 0, 0",
    ]);
    const { energy, adu } = readReferenceScan(text);
    expect(energy).toEqual([9001.5, 9002.5]);
    expect(adu).toEqual([6, 15]);
  });

  it("reads a real export like the Python build script", () => {
    const { energy, adu } = readReferenceScan(cuExport);
    const win = instrumentSegments().get("Cu")!;
    expect(energy.length).toBe(win.energy.length);
    energy.forEach((e, i) => expect(e).toBeCloseTo(win.energy[i], 3));
    adu.forEach((a, i) => expect(a).toBeCloseTo(win.adu[i], 2));
  });

  it("gives the smoothed curve and single windows", () => {
    const segs = instrumentSegments();
    expect([...segs.keys()].slice(0, 3)).toEqual(["V", "Cr", "Mn"]);
    const cu = instrumentFlux("Cu", 2);
    const w = segs.get("Cu")!;
    expect(cu.at([w.energy[10]])[0]).toBeCloseTo(w.adu[10] / 2, 6);
    expect(cu.at([w.energy[w.energy.length - 1] + 50])[0]).toBe(0);
    const sm = instrumentFlux(null, 2);
    expect(sm.covers(5300, 14600)).toBe(true);
    for (const [name, { energy, adu }] of segs) {
      const ratio = sm.at(energy).reduce((a, b) => a + b) / adu.reduce((a, b) => a + b / 2, 0);
      expect(ratio, name).toBeGreaterThan(0.7);
      expect(ratio, name).toBeLessThan(1.4);
    }
    expect(edgeEnergy("Cu", "K")).toBe(8979);
  });

  it("loads I0 tables in keV and counts", () => {
    const f = loadFlux(exampleFluxCsv());
    expect(f.covers(5000, 20000)).toBe(true);
    const keV = Array.from({ length: 100 }, (_, i) => `${(8 + i * 0.01).toFixed(3)} ${10 * (8000 + i * 10)}`).join("\n");
    const g = loadFlux(keV, 0, 1, "counts", 2);
    expect(g.energy[0]).toBeCloseTo(8000, 6);
    expect(g.at([8500])[0]).toBeCloseTo(5 * 8500, 0);
  });
});

describe("numerics", () => {
  it("nnls solves a non-negative problem", () => {
    const A = [[1, 0], [0, 1], [1, 1]];
    expect(nnls(A, [1, -1, 0])).toEqual([0.5, 0].map((v) => expect.closeTo(v, 12)));
    const x = nnls([[2, 1], [1, 3], [1, 1]], [5, 10, 4]);
    expect(x[0]).toBeGreaterThanOrEqual(0);
    expect(x[1]).toBeGreaterThanOrEqual(0);
  });

  it("brentq finds roots", () => {
    expect(brentq((x) => x * x - 2, 0, 2)).toBeCloseTo(Math.SQRT2, 10);
    expect(brentq((x) => Math.cos(x) - x, 0, 1)).toBeCloseTo(0.7390851332, 9);
  });

  it("formats like Python", () => {
    expect(fmtG(1.5)).toBe("1.5");
    expect(fmtG(100)).toBe("100");
    expect(fmtG(1e-5)).toBe("1e-05");
    expect(fmtG(1234567)).toBe("1.23457e+06");
    expect(fmtG(0.0001)).toBe("0.0001");
    expect(fmtFixed(8978.5, 0)).toBe("8978");
    expect(fmtFixed(8979.5, 0)).toBe("8980");
    expect(fmtFixed(0.0004, 3)).toBe("0.000");
    expect(parseFloatPy("1e3")).toBe(1000);
    expect(parseFloatPy("abc")).toBeNull();
  });
});

describe("detector", () => {
  it("sigma(mu t) is Poisson for an ideal detector", () => {
    const s = sigmaMut(1e4, 1.0, 1, 1, new DetectorModel());
    const n = 1e4 * Math.exp(-1);
    expect(s).toBeCloseTo(Math.sqrt(1 / n + 1 / 1e4), 12);
  });

  it("calibration recovers simulated parameters", () => {
    const E = Array.from({ length: 200 }, (_, i) => 8000 + 10 * i);
    const scans = simulateScans((x) => x.map((e) => 2e4 * (1 + 0.3 * Math.sin(e / 150))), E, { seed: 5 });
    const r = calibrate(scans);
    expect(r.gain).toBeGreaterThan(2.3);
    expect(r.gain).toBeLessThan(2.7);
    expect(r.instability).toBeGreaterThan(0.0012);
    expect(r.instability).toBeLessThan(0.0028);
    expect(r.dark_level_adu_per_s).toBeCloseTo(50, -1);
    expect(r.r2).toBeCloseTo(1, 1);
  });
});
