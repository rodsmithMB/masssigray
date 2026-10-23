import io
import math

import numpy as np
import pytest

from masssigray import fluorescence as fl
from masssigray import simulate
from masssigray.detector import DetectorModel, calibrate, sigma_mut, simulate_scans
from masssigray.flux import FluxSpectrum, example_flux_csv, load_flux
from masssigray.materials import Material, matrix_preset, parse_mixture
from masssigray.scan import ScanPlan
from masssigray.transmission import (PelletConstraint, TransmissionSetup, dilution_recipe,
                                     evaluate_masses, foil_setup, matrix_mass, optimize,
                                     time_for_target)
from masssigray.xray import edge_energy, edge_step_mu_rho, mu_rho

AREA = np.pi * 0.65**2  # 13 mm die
FLAT = FluxSpectrum.constant(1e5)


def test_mass_fractions_and_mixtures():
    cuo = Material.from_formula("CuO")
    assert cuo.mass_fraction("Cu") == pytest.approx(63.546 / 79.545, rel=1e-3)
    assert parse_mixture("Pt:5, Al2O3") == {"Pt": 5.0, "Al2O3": 95.0}
    mix = Material.from_text("Pt:5, Al2O3")
    assert mix.mass_fraction("Pt") == pytest.approx(0.05)
    assert sum(mix.mass_fractions.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        parse_mixture("Pt, Al2O3")


def test_cross_sections_match_tables():
    # NIST XCOM: Cu total mu/rho at 10 keV is about 216 cm2/g
    assert mu_rho(Material.from_formula("Cu"), 10000.0)[0] == pytest.approx(216, rel=0.03)
    E0 = edge_energy("Cu", "K")
    step = edge_step_mu_rho("Cu", "K", [E0 - 10, E0 + 50])
    assert step[0] == 0
    assert 200 < step[1] < 260


def test_cuo_pellet_hits_step_cap():
    s = TransmissionSetup(Material.from_formula("CuO", 6.31), matrix_preset("Cellulose"), AREA,
                          "Cu", "K", FLAT)
    p = optimize(s, PelletConstraint(total_mass_mg=100, max_edge_step=1.0))
    assert p.edge_step == pytest.approx(1.0, rel=0.01)
    assert p.binding.startswith("edge-step")
    assert p.total_mg == pytest.approx(100, rel=1e-6)
    # hand check: dmu = m_Cu * dmu/rho / A
    dmu = edge_step_mu_rho("Cu", "K", s.E0 + 50)[0]
    m_cu = p.m_sample_mg * 1e-3 * Material.from_formula("CuO").mass_fraction("Cu")
    assert m_cu * dmu / AREA == pytest.approx(p.edge_step, rel=1e-6)


def test_counting_optimum_is_mut_2():
    # Pure absorber, no matrix, negligible I0 noise: SNR max at mu*t ~ 2 (above the edge,
    # nearly all absorption from the absorber element itself is not true here, so check range)
    s = TransmissionSetup(Material.from_formula("Fe"), None, 1.0, "Fe", "K", FLAT,
                          scan=ScanPlan(edge_energy("Fe", "K"), i0_dwell_factor=1e6))
    p = optimize(s, PelletConstraint(mode="none", max_edge_step=None, max_mut=None), n_grid=4000)
    # analytic: maximize m*d*exp(-m*mu/2) -> m*mu = 2 at the reference energy
    assert p.mut_above == pytest.approx(2.0, rel=0.01)


def test_matrix_rules():
    s = TransmissionSetup(Material.from_formula("CuO", 6.31), matrix_preset("Cellulose"), AREA,
                          "Cu", "K", FLAT)
    ms = np.array([10.0, 150.0])
    assert np.allclose(matrix_mass(ms, s, PelletConstraint("min_total_mass", 100)), [90, 0])
    fixed = matrix_mass(ms, s, PelletConstraint("fixed_total_mass", 100))
    assert fixed[0] == 90 and np.isnan(fixed[1])
    thick = matrix_mass(np.array([0.0]), s, PelletConstraint("min_thickness", min_thickness_mm=1.0,
                                                             packing_fraction=0.7))
    assert thick[0] == pytest.approx(1.5e3 * 0.7 * AREA * 0.1, rel=1e-6)
    binder = matrix_mass(ms, s, PelletConstraint("min_matrix_mass", min_matrix_mg=40))
    assert np.allclose(binder, [40, 40])
    binder = matrix_mass(ms, s, PelletConstraint("min_matrix_mass", min_matrix_mg=40,
                                                 min_matrix_fraction=0.5))
    assert np.allclose(binder, [40, 150])


def test_min_matrix_mass_optimize():
    s = TransmissionSetup(Material.from_formula("CuO", 6.31), matrix_preset("Cellulose"), AREA,
                          "Cu", "K", FLAT)
    p = optimize(s, PelletConstraint("min_matrix_mass", min_matrix_mg=60))
    assert p.m_matrix_mg == pytest.approx(60)


def test_dilute_sample_not_matrix_limited():
    s = TransmissionSetup(Material.from_text("Pt:1, Al2O3", 3.95), matrix_preset("Boron nitride (BN)"),
                          AREA, "Pt", "L3", FLAT)
    p = optimize(s, PelletConstraint(total_mass_mg=100))
    assert p.m_matrix_mg == 0
    assert 1.8 < p.mut_above < 3.0
    assert p.edge_step < 0.5


def test_evaluate_matches_optimize():
    s = TransmissionSetup(Material.from_formula("CuO", 6.31), matrix_preset("Cellulose"), AREA,
                          "Cu", "K", FLAT)
    p = optimize(s)
    u = evaluate_masses(s, p.m_sample_mg, p.m_matrix_mg)
    assert u.snr_edge == pytest.approx(p.snr_edge)
    assert u.edge_step == pytest.approx(p.edge_step)


def test_dilution_recipe():
    d = dilution_recipe(0.3, 99.7, 2.0)
    assert d["possible"] and d["factor"] == 10
    assert d["stock_for_pellet_mg"] == pytest.approx(3.0)
    assert d["extra_matrix_mg"] == pytest.approx(99.7 - 2.7)
    assert dilution_recipe(5.0, 95, 2.0) is None


def test_foil_thickness():
    s = foil_setup(Material.from_formula("Cu", 8.96), "Cu", "K", FLAT)
    p = optimize(s, PelletConstraint(mode="none", max_edge_step=1.0, max_mut=None))
    # ~ 8 um Cu gives a step of ~1.5; a step of 1 needs ~ 5 um
    t_um = p.m_sample_mg * 1e-3 / 8.96 * 1e4
    assert 3 < t_um < 7


def test_sigma_mut_poisson():
    det = DetectorModel()
    s = sigma_mut(1e4, 0.0, 1.0, 1.0, det)
    assert s == pytest.approx(np.sqrt(2e-4))
    s2 = sigma_mut(1e4, 2.0, 1.0, 1e9, det)
    assert s2 == pytest.approx(np.sqrt(np.exp(2) / 1e4), rel=1e-6)


def test_calibration_recovers_parameters():
    E = np.linspace(8000, 9000, 400)
    scans = simulate_scans(lambda e: np.full_like(e, 2e3) * (1 + 0.5 * np.sin(e / 50)), E,
                           dwells=(0.5, 1, 2, 5, 10), repeats=4, gain=2.5, read_sd_adu=30,
                           dark_adu_per_s=50, instability=0.003, drift=0.02, rng=3)
    r = calibrate(scans)
    assert r.gain == pytest.approx(2.5, rel=0.1)
    assert r.instability == pytest.approx(0.003, rel=0.3)
    assert r.read_var_adu == pytest.approx(900, rel=0.5)
    assert r.drift_rms > 0.005
    det = r.to_detector()
    assert det.read_var == pytest.approx(r.read_var_adu / r.gain**2)


def test_fluorescence_limits():
    setup = fl.FluorescenceSetup(Material.from_text("Cu:0.01, SiO2"), matrix_preset("Cellulose"),
                                 AREA, "Cu", "K", FLAT, min_retention=0.0)
    m = fl._FModel(setup)
    dilute = m.evaluate([100.0], [0.0])
    assert dilute["R_ref"][0] == pytest.approx(1.0, abs=1e-3)
    # thick concentrated limit: R = (mu_b/sin + mu_f/sin) / (mu_T/sin + mu_f/sin)
    conc = fl.FluorescenceSetup(Material.from_formula("Cu"), None, 1.0, "Cu", "K", FLAT)
    mc = fl._FModel(conc)
    ev = mc.evaluate([1000.0], [0.0])
    a = mc.dmu_s[-1]
    mu_T, mu_f = mc.mu_s[-1], mc.muf_s
    assert ev["R_ref"][0] == pytest.approx((mu_T - a + mu_f) / (mu_T + mu_f), rel=1e-3)


def test_fluorescence_optimizer_respects_retention():
    setup = fl.FluorescenceSetup(Material.from_formula("CuO", 6.3), matrix_preset("Cellulose"), AREA,
                                 "Cu", "K", FLAT, min_retention=0.9)
    p = fl.optimize(setup)
    assert p.retention_min_k >= 0.9 - 1e-9
    assert p.binding.startswith("retention")


def test_flux_loader_and_reference():
    f = load_flux(io.StringIO(example_flux_csv()))
    assert f.covers(5000, 20000)
    keV = "\n".join(f"{e/1000:.3f} {10*e:.1f}" for e in np.arange(8000, 9000, 10.0))
    g = load_flux(io.StringIO(keV), value_kind="counts", dwell=2.0)
    assert g.energy[0] == pytest.approx(8000)
    assert g(8500.0) == pytest.approx(5 * 8500, rel=1e-3)
    E = np.linspace(8800, 9600, 801)
    mu = 0.1 + 1 / (1 + np.exp(-(E - 8980) / 2)) + 0.5 * np.exp(-0.5 * ((E - 8995) / 4) ** 2)
    e0, norm = simulate.normalize(E, mu)
    assert e0 == pytest.approx(8980, abs=3)
    assert norm[-1] == pytest.approx(1.0, abs=0.05)
    assert norm.max() > 1.3


def test_time_for_target_poisson_and_floor():
    s = TransmissionSetup(Material.from_formula("CuO", 6.31), matrix_preset("Cellulose"), AREA,
                          "Cu", "K", FLAT, objective="exafs", k_weight=3)
    p = optimize(s)
    target = p.chi_noise_rms / 3
    te = time_for_target(s, p.m_sample_mg, p.m_matrix_mg, target)
    assert te.sigma_one == pytest.approx(p.chi_noise_rms)
    assert te.n_scans == 9  # pure counting noise: sigma ~ 1/sqrt(N)
    assert te.dwell_factor == pytest.approx(9, rel=1e-3)
    assert te.time_dwell_s == pytest.approx(te.time_scans_s, rel=1e-3)
    assert te.time_one_s == pytest.approx(sum(s.scan.total_time()))

    s.detector = DetectorModel(instability=0.05)  # floor ~ sqrt(2)*0.05 in mu*t per point
    te = time_for_target(s, p.m_sample_mg, p.m_matrix_mg, target)
    assert te.floor > target and te.dwell_factor is None and te.time_dwell_s is None
    assert te.n_scans == math.ceil((te.sigma_one / target) ** 2)

    # overhead is paid per point per scan, so repeats cost more than longer dwells
    te = time_for_target(s, p.m_sample_mg, p.m_matrix_mg, te.sigma_one / 2, overhead_s=0.5)
    n_pts = len(s.scan.grid()[0])
    assert te.time_one_s == pytest.approx(sum(s.scan.total_time()) + 2 * n_pts * 0.5)


def test_fluorescence_time_for_target():
    setup = fl.FluorescenceSetup(Material.from_text("Pt:1, Al2O3", 3.95), matrix_preset("Cellulose"),
                                 AREA, "Pt", "L3", FLAT, k_weight=3)
    p = fl.optimize(setup)
    te = fl.time_for_target(setup, p.m_sample_mg, p.m_matrix_mg, p.chi_noise_rms / 2)
    assert te.n_scans == 4
    assert te.dwell_factor == pytest.approx(4, rel=1e-3)
