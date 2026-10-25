"""Tests for energy_path.py, the per-cell path-energy quantities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'avachain'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

import energy_path as ep
import release_geometry as rg
from arrest_indices import (
    R_FIT,
    arrest_length_energy,
    critical_strength,
    gravitational_shear,
    residual_shear,
    slab_energy_cap,
    weak_layer_stiffness,
)
from energy_path import (
    cell_energy,
    e_column,
    g_avail,
    g_c,
    g_slab,
    n_max,
    tau_net,
)

# Meloche et al. (2025) Table 1 slab, as in test_arrest_indices._BASE.
RHO, H, PSI, E, SIG_T, G_WL, D_WL, NU = 250.0, 0.5, 35.0, 4e6, 6e3, 0.2e6, 0.04, 0.3
TAU_P0 = 1e3
K_WL = weak_layer_stiffness(G_WL, D_WL)


class TestJGRCasesThroughEnergyPath:
    """Saturated form (N = N_max) on a linear tau_p ramp lands at 71, 14, 81 m."""

    @pytest.mark.parametrize('theta, delta, expected', [
        (30.0, 0.0, 71.17),   # Fig. 8a
        (150.0, 0.0, 14.23),  # Fig. 8b
        (15.0, 1.0, 81.12),   # JBC
    ])
    def test_arrest_point_on_ramp(self, theta, delta, expected):
        x = np.arange(0.0, 200.0, 0.001)
        G_av = g_avail(n_max(SIG_T, H), E, H, NU, R_FIT)
        stopped = g_c(TAU_P0 + theta * x, K_WL, delta) > G_av
        x_arrest = x[np.argmax(stopped)]
        assert x_arrest == pytest.approx(expected, abs=0.01)
        tau_star = critical_strength(SIG_T, H, E, K_WL, delta, 0.0, R_FIT, NU)
        assert x_arrest == pytest.approx(
            arrest_length_energy(TAU_P0, tau_star, theta), abs=0.002)


class TestPrimitives:

    def test_tau_net_exactly_zero_at_friction_angle(self):
        assert tau_net(RHO, H, ep.PHI_DEG, ep.PHI_DEG) == 0.0

    @pytest.mark.parametrize('psi', [0.0, 10.0, 26.9])
    def test_tau_net_negative_below_friction_angle(self, psi):
        assert tau_net(RHO, H, psi, ep.PHI_DEG) < 0.0

    def test_tau_net_is_tau_g_minus_tau_r(self):
        expected = (gravitational_shear(RHO, H, PSI)
                    - residual_shear(RHO, H, PSI, ep.PHI_DEG))
        assert tau_net(RHO, H, PSI) == pytest.approx(expected, rel=1e-12)

    def test_g_avail_at_n_max_is_r_times_g_slab(self):
        for R in (0.3, R_FIT, 0.7):
            assert g_avail(n_max(SIG_T, H), E, H, NU, R) == pytest.approx(
                R * g_slab(SIG_T, H, E, NU), rel=1e-12)

    def test_g_avail_zero_tension_is_zero(self):
        assert g_avail(0.0, E, H) == 0.0

    def test_g_slab_is_existing_cap(self):
        assert g_slab(SIG_T, H, E, NU) == slab_energy_cap(SIG_T, H, E, NU)

    def test_g_c_residual_reduces_it(self):
        assert g_c(2000.0, K_WL, 0.0, tau_r=500.0) < g_c(2000.0, K_WL, 0.0)


def _row(**kw):
    base = dict(slab_density=RHO, slab_thickness=H, wl_shear_strength=2000.0,
                wl_thickness=D_WL, E_eff=E, E_eff__project_fit=2e6,
                sigma_t_mean=SIG_T, sigma_t_wl=3e3)
    return pd.Series({**base, **kw})


class TestCellEnergy:

    def test_values_from_row(self):
        out = cell_energy(_row(), PSI, delta=0.0)
        assert out['tau_net'] == pytest.approx(tau_net(RHO, H, PSI))
        assert out['G_c'] == pytest.approx(g_c(2000.0, G_WL / D_WL, 0.0))
        assert out['G_slab_path'] == pytest.approx(g_slab(SIG_T, H, E))
        assert out['N_max'] == pytest.approx(SIG_T * H)

    def test_sigma_t_column_switch(self):
        out = cell_energy(_row(), PSI, sigma_t_col='sigma_t_wl')
        assert out['N_max'] == pytest.approx(3e3 * H)

    def test_e_relation_switch(self):
        assert e_column(ep.E_RELATION) == 'E_eff'
        out = cell_energy(_row(), PSI, e_relation='project_fit')
        assert out['G_slab_path'] == pytest.approx(g_slab(SIG_T, H, 2e6))

    def test_g_wl_scales_g_c_inversely(self):
        a = cell_energy(_row(), PSI, G_wl=0.1e6)['G_c']
        b = cell_energy(_row(), PSI, G_wl=0.4e6)['G_c']
        assert a / b == pytest.approx(4.0)

    def test_missing_wl_thickness_uses_fallback(self):
        out = cell_energy(_row(wl_thickness=np.nan), PSI, delta=0.0)
        assert out['G_c'] == pytest.approx(
            g_c(2000.0, G_WL / ep.D_WL_FALLBACK, 0.0))

    def test_no_weak_layer_gives_nan_g_c(self):
        out = cell_energy(_row(wl_shear_strength=np.nan), PSI)
        assert np.isnan(out['G_c'])
        assert np.isfinite(out['tau_net'])

    def test_residual_option(self):
        tr = residual_shear(RHO, H, PSI, ep.PHI_DEG)
        out = cell_energy(_row(), PSI, delta=0.0, use_residual=True)
        assert out['G_c'] == pytest.approx(g_c(2000.0, G_WL / D_WL, 0.0, tr))


def test_compute_meloche_features_emits_energy_columns_without_theta():
    """Four clusters are too few for a theta fit, yet every row gets the columns."""
    from snowpack_analysis import compute_meloche_features
    n = 20
    cmap = np.zeros((n, n), int)
    cmap[:10, :10], cmap[:10, 10:], cmap[10:, :10], cmap[10:, 10:] = 1, 2, 3, 4
    rows, _ = np.mgrid[0:n, 0:n]
    dem = -rows * np.tan(np.radians(PSI))   # 1 m pixels, falls to the south
    feats = pd.DataFrame([_row(Lambda=1.0, E_slab=E, sigma_t=SIG_T)] * 4,
                         index=[1, 2, 3, 4])
    out = compute_meloche_features({'all': feats}, cmap, dem,
                                   from_origin(0, n, 1.0, 1.0))
    for col in ('tau_net', 'G_c', 'N_max', 'G_slab_path'):
        assert out[col].notna().all(), col
    assert out['tau_net'].iloc[0] == pytest.approx(tau_net(RHO, H, PSI), rel=1e-3)


class TestAvachainConstants:
    """energy_path keeps its own copies of constants defined elsewhere in avachain."""

    def test_copies_match_their_sources(self):
        import release_geometry as rg
        import snowpack_analysis as sa
        assert ep.G_WL == sa.G_WL
        assert ep.NU == sa.NU == rg.NU
        assert ep.E_RELATION == sa.E_RELATION
        assert ep.PHI_DEG == rg.FRICTION_DEG

    def test_energy_delta_is_zero_and_separate_from_eq20_delta(self):
        import release_geometry as rg
        assert ep.ENERGY_DELTA == 0.0
        assert rg.MELOCHE_DELTA == 1.0

    def test_cell_energy_uses_energy_delta(self, monkeypatch):
        g0 = cell_energy(_row(), PSI)['G_c']
        monkeypatch.setattr(ep, 'ENERGY_DELTA', 1.0)
        assert cell_energy(_row(), PSI)['G_c'] == pytest.approx(2.0 * g0)
