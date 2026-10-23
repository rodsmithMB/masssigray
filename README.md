# massSigray

Sample planning for lab-based XAS. Given a sample composition, a host matrix, the
absorption edge and the pellet area, it finds the sample and matrix masses that give
the best signal-to-noise. It uses tabulated cross sections (xraydb, Elam tables) and
your measured incident flux I0(E), so edge steps, counts and noise are close to
quantitative for your instrument.

## Run

    conda activate masssigray
    streamlit run masssigray/app.py        # or: masssigray-app

Install elsewhere with `pip install -e ".[test]"`; run tests with `pytest`.

## What it does

| Tab | Purpose |
|---|---|
| Transmission pellet | Optimal mg of sample and matrix under a pellet rule (minimum total mass, fixed total mass, minimum thickness, or minimum matrix mass to bind the pellet), with caps on edge step and total mu*t. Dilution recipe when the sample mass is too small to weigh. Check masses you actually weighed. Counting time (longer dwells vs repeated scans) to reach a target rms σ(k^w chi). |
| Fluorescence | Signal, self-absorption (EXAFS amplitude retention) and noise; searches sample and matrix mass together; counting time for a target noise. |
| Foil / film | Optimal thickness for matrix-free samples of known density. |
| Detector calibration | Photon-transfer fit to repeated I0 scans at several dwell times: gain, read noise, dark current, source instability, scan-to-scan drift. |
| Method | Equations and limitations. |

Outputs include predicted mu*t(E) with simulated noise, transmitted counts, sigma of
k^w chi(k), and a CSV export. Upload a reference spectrum of a similar compound to get
realistic white-line (peak) absorbance.

## Inputs

* **I0 file**: text/CSV with an energy column (eV or keV) and a rate (counts/s) or counts
  per point (give the dwell). Divide by the calibrated gain to get effective counts.
* **Calibration scans**: one file per scan, energy + ROI-summed ADU; set dwell and
  dark/light per file in the table.

## Library use

```python
import numpy as np
from masssigray import Material, matrix_preset, TransmissionSetup, PelletConstraint, optimize, load_flux

flux = load_flux("I0.csv")                         # energy, counts/s
setup = TransmissionSetup(Material.from_formula("CuO", 6.31), matrix_preset("Cellulose"),
                          area_cm2=np.pi * 0.65**2, absorber="Cu", edge="K", flux=flux)
plan = optimize(setup, PelletConstraint(mode="min_total_mass", total_mass_mg=100))
print(plan.m_sample_mg, plan.m_matrix_mg, plan.edge_step, plan.snr_edge, plan.binding)
```

Modules: `materials` (formulas, wt% mixtures, presets), `xray` (cross sections, edge
steps), `flux`, `detector` (noise model, calibration), `scan` (energy grid and dwell),
`transmission`, `fluorescence`, `simulate`.
