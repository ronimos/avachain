"""
tests/test_arrest_indices.py — Unit tests for arrest_indices.py.

Run with:  uv run pytest tests/test_arrest_indices.py -v

Covers:
  TestMeloche2025Calibration — three published calibration runs (Fig8a, Fig8b, JBC)
  TestPrimitiveFunctions     — individual function correctness
  TestJan18Sanity            — aggregate physical-plausibility smoke test on real data
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from arrest_indices import (
    R_FIT,
    arrest_length,
    arrest_length_energy,
    critical_strength,
    dynamic_gradient,
    elastic_length,
    energy_ratio,
    evaluate,
    gravitational_shear,
    residual_shear,
    slab_energy_cap,
    tension_gradient,
    tensile_length,
    weak_layer_fracture_energy,
)

# ---------------------------------------------------------------------------
# Shared calibration inputs (Meloche et al. 2025, Table 1 / §3)
# ---------------------------------------------------------------------------
_BASE = dict(
    rho=250.0, h=0.5, psi_deg=35.0, E=4e6, sigma_t=6e3,
    G_wl=0.2e6, D_wl=0.04, nu=0.3, phi_deg=27.0,
    tau_p0=1e3,
)
# Run-specific overrides
_FIG8A = dict(theta=30.0,  delta=0.0)
_FIG8B = dict(theta=150.0, delta=0.0)
_JBC   = dict(theta=15.0,  delta=1.0)

_OBSERVED = {'Fig8a': 74.0, 'Fig8b': 14.0, 'JBC': 74.0}


def _ev(**kw):
    p = {**_BASE, **kw}
    return evaluate(
        rho=p['rho'], h=p['h'], psi_deg=p['psi_deg'], E=p['E'],
        sigma_t=p['sigma_t'], D_wl=p['D_wl'], G_wl=p['G_wl'],
        theta=p['theta'], nu=p['nu'], phi_deg=p['phi_deg'], delta=p['delta'],
        tau_p0=p.get('tau_p0'), C=0.045, R=R_FIT,
    )


# ---------------------------------------------------------------------------
# Group 1 — Meloche 2025 calibration runs
# ---------------------------------------------------------------------------

class TestMeloche2025Calibration:

    def test_evaluate_returns_required_keys(self):
        out = _ev(**_FIG8A)
        for key in ('Lambda', 'K_wl', 'tau_g', 'tau_r', 'k_f', 'k_x',
                    'L_t', 'L_dyn', 'sustained', 'G_slab', 'tau_p_star', 'R_used'):
            assert key in out, f"missing key: {key}"

    def test_energy_keys_present_when_tau_p0_given(self):
        out = _ev(**_FIG8A)
        for key in ('G_c0', 'R0', 'A_ca_energy'):
            assert key in out, f"missing key: {key}"

    def test_fig8a_A_ca_energy_within_15m(self):
        out = _ev(**_FIG8A)
        assert abs(out['A_ca_energy'] - _OBSERVED['Fig8a']) <= 15.0, (
            f"Fig8a A_ca_energy={out['A_ca_energy']:.1f} m, expected ~{_OBSERVED['Fig8a']} m"
        )

    def test_fig8b_A_ca_energy_within_15m(self):
        out = _ev(**_FIG8B)
        assert abs(out['A_ca_energy'] - _OBSERVED['Fig8b']) <= 15.0, (
            f"Fig8b A_ca_energy={out['A_ca_energy']:.1f} m, expected ~{_OBSERVED['Fig8b']} m"
        )

    def test_jbc_A_ca_energy_within_20m(self):
        # JBC is out-of-sample; doc predicts 81 m vs observed ~74 m
        out = _ev(**_JBC)
        assert abs(out['A_ca_energy'] - _OBSERVED['JBC']) <= 20.0, (
            f"JBC A_ca_energy={out['A_ca_energy']:.1f} m, expected ~{_OBSERVED['JBC']} m"
        )

    def test_tau_p_star_similar_for_fig8a_and_fig8b(self):
        # Both runs share sigma_t, h, E, K_wl, delta=0 → tau_p* is identical by construction
        # (threshold-strength property: arrest happens at nearly constant WL strength)
        a = _ev(**_FIG8A)['tau_p_star']
        b = _ev(**_FIG8B)['tau_p_star']
        assert abs(a - b) <= 100.0, (
            f"Fig8a tau_p_star={a:.0f} Pa, Fig8b tau_p_star={b:.0f} Pa; "
            f"expected within 100 Pa"
        )

    def test_tau_p_star_value_fig8a(self):
        # Arithmetic: K_wl=5e6, G_slab≈2.047 J/m², tau_p*=sqrt(2*5e6*0.48*2.047)≈3135 Pa
        out = _ev(**_FIG8A)
        assert out['tau_p_star'] == pytest.approx(3135.0, abs=50.0)

    def test_sustained_true_above_friction_angle(self):
        out = _ev(**_FIG8A)
        assert out['sustained'] is True

    def test_sustained_false_below_friction_angle(self):
        # psi=25 < phi=27
        out = evaluate(
            rho=250, h=0.5, psi_deg=25.0, E=4e6, sigma_t=6e3,
            D_wl=0.04, G_wl=0.2e6, theta=30.0, phi_deg=27.0,
        )
        assert out['sustained'] is False

    def test_k_f_zero_below_friction_angle(self):
        out = evaluate(
            rho=250, h=0.5, psi_deg=25.0, E=4e6, sigma_t=6e3,
            D_wl=0.04, G_wl=0.2e6, theta=30.0, phi_deg=27.0,
        )
        assert out['k_f'] == pytest.approx(0.0)

    def test_L_t_infinite_below_friction_angle(self):
        out = evaluate(
            rho=250, h=0.5, psi_deg=25.0, E=4e6, sigma_t=6e3,
            D_wl=0.04, G_wl=0.2e6, theta=30.0, phi_deg=27.0,
        )
        assert math.isinf(out['L_t'])

    def test_A_ca_energy_infinite_when_theta_zero(self):
        # theta=0 means uniform WL strength → crack never encounters rising resistance
        out = evaluate(
            rho=250, h=0.5, psi_deg=35.0, E=4e6, sigma_t=6e3,
            D_wl=0.04, G_wl=0.2e6, theta=0.0, phi_deg=27.0, tau_p0=1e3,
        )
        # A_ca_energy should be inf (or absent — arrest_length_energy returns inf for theta<=0)
        val = out.get('A_ca_energy', np.inf)
        assert math.isinf(val)

    def test_A_ca_brittle_fig8a_within_15m(self):
        # Eq. 20 scaling (C=0.045) should also be close
        out = _ev(**_FIG8A)
        assert 'A_ca' in out
        assert abs(out['A_ca'] - _OBSERVED['Fig8a']) <= 15.0, (
            f"Fig8a A_ca (brittle)={out['A_ca']:.1f} m, expected ~{_OBSERVED['Fig8a']} m"
        )

    def test_A_ca_brittle_fig8b_within_15m(self):
        out = _ev(**_FIG8B)
        assert 'A_ca' in out
        assert abs(out['A_ca'] - _OBSERVED['Fig8b']) <= 15.0, (
            f"Fig8b A_ca (brittle)={out['A_ca']:.1f} m, expected ~{_OBSERVED['Fig8b']} m"
        )


# ---------------------------------------------------------------------------
# Group 2 — primitive function checks
# ---------------------------------------------------------------------------

class TestPrimitiveFunctions:

    def test_elastic_length_formula(self):
        # Lambda = sqrt(E * h * D_wl / ((1-nu^2) * G_wl))
        E, h, D, G, nu = 4e6, 0.5, 0.04, 0.2e6, 0.3
        expected = math.sqrt(E * h * D / ((1 - nu**2) * G))
        assert elastic_length(E, h, D, G, nu) == pytest.approx(expected, rel=1e-9)

    def test_elastic_length_value(self):
        # Concrete value for the calibration inputs
        val = elastic_length(4e6, 0.5, 0.04, 0.2e6, nu=0.3)
        expected = math.sqrt(4e6 * 0.5 * 0.04 / (0.91 * 0.2e6))
        assert val == pytest.approx(expected, rel=1e-6)

    def test_gravitational_shear_formula(self):
        # tau_g = rho * g * h * sin(psi)
        val = gravitational_shear(250, 0.5, 35.0)
        expected = 250 * 9.81 * 0.5 * math.sin(math.radians(35))
        assert val == pytest.approx(expected, rel=1e-6)

    def test_gravitational_shear_approx_702(self):
        val = gravitational_shear(250, 0.5, 35.0)
        assert 695 < val < 715

    def test_tension_gradient_positive_above_friction(self):
        k = tension_gradient(250, 35.0, 27.0)
        assert k > 0

    def test_tension_gradient_zero_at_friction_angle(self):
        k = tension_gradient(250, 27.0, 27.0)
        assert k == pytest.approx(0.0, abs=1e-9)

    def test_slab_energy_cap_formula(self):
        # G_slab = sigma_t^2 * h / (2 * E')
        sigma_t, h, E, nu = 6e3, 0.5, 4e6, 0.3
        E_prime = E / (1 - nu**2)
        expected = sigma_t**2 * h / (2 * E_prime)
        assert slab_energy_cap(sigma_t, h, E, nu) == pytest.approx(expected, rel=1e-9)

    def test_weak_layer_fracture_energy_formula(self):
        # G_c = tau_p^2 * (1+delta) / (2 * K_wl)
        tau_p, K_wl, delta = 3e3, 5e6, 0.0
        expected = tau_p**2 * (1 + delta) / (2 * K_wl)
        assert weak_layer_fracture_energy(tau_p, K_wl, delta=delta) == pytest.approx(expected, rel=1e-9)

    def test_energy_ratio_at_tau_p_star_equals_R_FIT(self):
        # By construction, energy_ratio(tau_p_star, ...) = R_FIT
        sigma_t, h, E, nu = 6e3, 0.5, 4e6, 0.3
        K_wl = 0.2e6 / 0.04  # G_wl / D_wl = 5e6 Pa/m
        delta = 0.0
        tau_p_star = critical_strength(sigma_t, h, E, K_wl, delta=delta, R=R_FIT, nu=nu)
        R = energy_ratio(tau_p_star, K_wl, sigma_t, h, E, delta=delta, nu=nu)
        assert R == pytest.approx(R_FIT, rel=1e-6)

    def test_tensile_length_positive_for_positive_k(self):
        val = tensile_length(6e3, 300.0)
        assert val > 0

    def test_tensile_length_infinite_for_zero_k(self):
        val = tensile_length(6e3, 0.0)
        assert math.isinf(val)

    def test_dynamic_gradient_less_than_static(self):
        k_f = tension_gradient(250, 35.0, 27.0)
        k_x = dynamic_gradient(k_f, 4e6, 250.0, nu=0.3)
        # k_x < k_f (dynamic cracks build tension more slowly)
        assert k_x < k_f

    def test_residual_shear_formula(self):
        rho, h, psi, phi = 250, 0.5, 35.0, 27.0
        expected = rho * 9.81 * h * math.cos(math.radians(psi)) * math.tan(math.radians(phi))
        assert residual_shear(rho, h, psi, phi) == pytest.approx(expected, rel=1e-6)

    def test_arrest_length_energy_zero_when_tau_p0_above_star(self):
        # If WL is already strong enough, arrest length is 0
        val = arrest_length_energy(tau_p0=5e3, tau_p_star=3e3, theta=50.0)
        assert val == pytest.approx(0.0)

    def test_arrest_length_energy_formula_when_positive(self):
        # (tau_p_star - tau_p0) / theta
        val = arrest_length_energy(tau_p0=1e3, tau_p_star=3e3, theta=50.0)
        assert val == pytest.approx((3e3 - 1e3) / 50.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Group 3 — Jan 18 aggregate sanity smoke test
# ---------------------------------------------------------------------------

_CSV = Path(__file__).resolve().parent.parent / (
    "outputs/little_prof/analysis/meloche_features_2026-01-18.csv"
)

_RHO_FIXED   = 280.0
_H_FIXED     = 0.5
_E_FIXED     = (_RHO_FIXED / 300.0)**2.5 * 4e6
_SIGMA_T_FIXED = (_RHO_FIXED / 300.0)**1.4 * 5e3
_D_WL_FIXED  = 0.04
_G_WL_FIXED  = 0.2e6


@pytest.fixture(scope="class")
def jan18_results(request):
    if not _CSV.exists():
        pytest.skip(f"Jan 18 data not found: {_CSV}")

    import pandas as pd
    df = pd.read_csv(_CSV)
    needed = ['tau_g', 'theta', 'Lambda', 'L_t', 'A_ca_brittle', 'slope_angle']
    df = df.dropna(subset=needed)
    df = df[(df['tau_g'] > 0) & (df['theta'] > 0) & (df['Lambda'] > 0)
            & (df['L_t'] > 0) & (df['A_ca_brittle'] > 0) & (df['slope_angle'] > 28.0)]

    if df.empty:
        pytest.skip("No valid rows after filtering")

    results = []
    for _, row in df.iterrows():
        out = evaluate(
            rho=_RHO_FIXED, h=_H_FIXED, psi_deg=float(row['slope_angle']),
            E=_E_FIXED, sigma_t=_SIGMA_T_FIXED,
            D_wl=_D_WL_FIXED, G_wl=_G_WL_FIXED,
            theta=float(row['theta']),
            phi_deg=27.0, tau_p0=None,
        )
        results.append(out)

    request.cls.results = results
    return results


@pytest.mark.usefixtures("jan18_results")
class TestJan18Sanity:

    def test_all_lambda_positive(self):
        assert all(r['Lambda'] > 0 for r in self.results)

    def test_all_tau_g_positive(self):
        assert all(r['tau_g'] > 0 for r in self.results)

    def test_all_A_ca_present_and_positive(self):
        for r in self.results:
            assert 'A_ca' in r, "A_ca missing — L_t not finite or theta=0?"
            assert r['A_ca'] > 0

    def test_median_A_ca_in_physical_range(self):
        vals = [r['A_ca'] for r in self.results if 'A_ca' in r]
        median = float(np.median(vals))
        assert 10.0 <= median <= 1000.0, (
            f"Median A_ca={median:.1f} m outside expected 10–1000 m range"
        )

    def test_all_sustained(self):
        # All filtered rows have slope_angle > 28 > phi=27
        assert all(r['sustained'] is True for r in self.results)
