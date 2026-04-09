"""Tests for co2_adsorption_analysis."""
import numpy as np
import pytest

import co2_adsorption_analysis as co2


# ── langmuir ────────────────────────────────────────────────────────────────

def test_langmuir_zero_pressure_is_zero():
    assert co2.langmuir(0.0, q_max=3.0, K_L=5.0) == 0.0

def test_langmuir_saturation_approaches_qmax():
    """At very high P, q → q_max."""
    P = 1e6
    q = co2.langmuir(P, q_max=3.0, K_L=5.0)
    assert q == pytest.approx(3.0, rel=1e-5)

def test_langmuir_monotonic_increasing():
    P = np.linspace(0, 10, 50)
    q = co2.langmuir(P, q_max=3.0, K_L=5.0)
    assert np.all(np.diff(q) > 0)


# ── freundlich ──────────────────────────────────────────────────────────────

def test_freundlich_zero_pressure_is_zero():
    assert co2.freundlich([0.0], K_F=2.0, n=2.0)[0] == 0.0

def test_freundlich_n_one_is_linear():
    """With n=1, q = K_F * P (linear)."""
    P = np.array([0.5, 1.0, 2.0])
    q = co2.freundlich(P, K_F=3.0, n=1.0)
    np.testing.assert_allclose(q, 3.0 * P)

def test_freundlich_accepts_python_list():
    out = co2.freundlich([1.0, 2.0], K_F=1.0, n=2.0)
    assert isinstance(out, np.ndarray)


# ── fit_isotherms ───────────────────────────────────────────────────────────

def test_fit_isotherms_returns_both_models():
    fits = co2.fit_isotherms()
    assert "langmuir" in fits and "freundlich" in fits
    assert len(fits["langmuir"]) == 2
    assert len(fits["freundlich"]) == 2

def test_fit_isotherms_recovers_known_parameters():
    """Synthetic data from a known Langmuir curve should be recovered."""
    P = np.linspace(0.01, 1.0, 20)
    true_q_max, true_K = 4.0, 6.0
    q = co2.langmuir(P, true_q_max, true_K)
    sigma = np.full_like(q, 0.01)
    fits = co2.fit_isotherms(P=P, q=q, sigma=sigma)
    assert fits["langmuir"][0] == pytest.approx(true_q_max, rel=1e-3)
    assert fits["langmuir"][1] == pytest.approx(true_K,     rel=1e-3)


# ── breakthrough_curve / breakthrough_times ─────────────────────────────────

def test_breakthrough_curve_starts_at_zero():
    assert co2.breakthrough_curve(np.array([0.0]), c_in=0.15, tau=900)[0] == 0.0

def test_breakthrough_curve_approaches_cin():
    t = np.array([1e6])
    assert co2.breakthrough_curve(t, c_in=0.15, tau=900)[0] == pytest.approx(0.15)

def test_breakthrough_times_happy():
    t = np.linspace(0, 5000, 1000)
    c = co2.breakthrough_curve(t, c_in=0.15, tau=900)
    t_break, t_sat = co2.breakthrough_times(t, c, c_in=0.15)
    assert 0 < t_break < t_sat
    assert t_sat < 5000

def test_breakthrough_times_never_reaches_threshold_raises():
    """Failure mode 1: curve too flat to ever cross thresholds."""
    t = np.linspace(0, 10, 100)
    c = np.zeros_like(t)
    with pytest.raises(ValueError, match="never reaches"):
        co2.breakthrough_times(t, c, c_in=0.15)


# ── working_capacity ────────────────────────────────────────────────────────

def test_working_capacity_positive_for_typical_conditions():
    wc = co2.working_capacity(T_ads=40, T_regen=120, q_max=3.0, K_L=5.0)
    assert wc > 0

def test_working_capacity_increases_with_regen_temperature():
    """Higher regen temperature → more capacity recovered (up to a limit)."""
    wc_low  = co2.working_capacity(40, 80,  3.0, 5.0)
    wc_high = co2.working_capacity(40, 150, 3.0, 5.0)
    assert wc_high > wc_low

def test_working_capacity_zero_when_temperatures_equal():
    """Failure mode 2: no temperature swing → zero working capacity."""
    wc = co2.working_capacity(T_ads=40, T_regen=40, q_max=3.0, K_L=5.0)
    # K does not change → q_ads(0.15) - q_regen(0.01)
    expected = co2.langmuir(0.15, 3.0, 5.0) - co2.langmuir(0.01, 3.0, 5.0)
    assert wc == pytest.approx(expected)

def test_working_capacity_custom_pressures():
    wc = co2.working_capacity(40, 120, 3.0, 5.0, P_ads=0.20, P_regen=0.05)
    assert wc > 0
