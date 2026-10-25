"""Path-energy brief, Step 1 options: sigma_t(rho) relations, KWL_THICKNESS, ln_Pi,
and the weakest-WL-element thickness behind KWL_THICKNESS = 'weakest_element'."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'avachain'))

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arrest_indices as ai
import energy_path as ep
from snowpack_analysis import profile_features


class TestJamieson1990:

    def test_published_coefficients(self):
        """Table III group I: A = 79.7 kN m-2, a = 2.39, rho_ice = 917."""
        assert ai.slab_tensile_strength_jamieson1990(917.0) == pytest.approx(79.7e3)
        assert ai.slab_tensile_strength_jamieson1990(250.0) == pytest.approx(
            79.7e3 * (250.0 / 917.0) ** 2.39, rel=1e-12)

    def test_value_in_their_density_range(self):
        # 79.7 * 0.27263^2.39 = 3.568 kPa at 250 kg m-3, inside the 100-345 range of the fit
        assert ai.slab_tensile_strength_jamieson1990(250.0) == pytest.approx(3568.4, abs=0.5)

    def test_aggregate_slab_emits_suffixed_bounds_only_when_asked(self):
        rho, dz = np.array([200.0, 300.0]), np.array([0.3, 0.7])
        sp, dd = np.array([0.8, 0.9]), np.array([0.0, 0.0])
        base = ai.aggregate_slab(rho, dz, sp, dd)
        out = ai.aggregate_slab(rho, dz, sp, dd, st_relations_extra=('jamieson1990',))
        assert 'sigma_t_mean__jamieson1990' not in base
        assert out['sigma_t_mean'] == base['sigma_t_mean']
        f = ai.facetedness(sp, dd)
        st = ai.slab_tensile_strength_jamieson1990(rho) * (1 - ai.FACET_STRENGTH_FACTOR * f)
        assert out['sigma_t_mean__jamieson1990'] == pytest.approx(
            ai.tensile_strength_mean(st, dz))
        assert np.isfinite(out['sigma_t_wl__jamieson1990'])

    def test_sigma_t_column_selects_it(self):
        row = pd.Series(dict(slab_density=250.0, slab_thickness=1.0, wl_shear_strength=2000.0,
                             wl_thickness=0.05, E_eff=5e6, sigma_t_mean=6e3,
                             sigma_t_mean__jamieson1990=3e3))
        out = ep.cell_energy(row, 35.0, sigma_t_col='sigma_t_mean__jamieson1990')
        assert out['N_max'] == pytest.approx(3e3)


class TestKwlThickness:
    ROW = pd.Series(dict(slab_density=250.0, slab_thickness=1.0, wl_shear_strength=2000.0,
                         wl_thickness=0.08, wl_weakest_thickness=0.02, E_eff=5e6,
                         sigma_t_mean=6e3))

    def test_choices(self):
        assert ep.wl_thickness(self.ROW, 'sequence') == 0.08
        assert ep.wl_thickness(self.ROW, 'weakest_element') == 0.02
        assert ep.wl_thickness(self.ROW, 0.04) == 0.04

    def test_g_c_scales_with_d_wl(self):
        g = {k: ep.cell_energy(self.ROW, 35.0, kwl_thickness=k)['G_c']
             for k in ('sequence', 'weakest_element', 0.04)}
        assert g['sequence'] / g['weakest_element'] == pytest.approx(4.0)
        assert g['sequence'] / g[0.04] == pytest.approx(2.0)

    def test_missing_column_falls_back(self):
        row = self.ROW.drop('wl_weakest_thickness')
        assert ep.wl_thickness(row, 'weakest_element') == ep.D_WL_FALLBACK

    def test_unknown_choice_raises(self):
        with pytest.raises(ValueError):
            ep.wl_thickness(self.ROW, 'nonsense')

    def test_default_is_sequence(self):
        assert ep.KWL_THICKNESS == 'sequence'


def test_ln_pi_is_the_saturated_admit_test():
    row = TestKwlThickness.ROW
    out = ep.cell_energy(row, 35.0)
    assert out['ln_Pi'] == pytest.approx(np.log(ep.energy_r() * out['G_slab_path'] / out['G_c']))
    assert np.isnan(ep.cell_energy(row.drop('wl_shear_strength'), 35.0)['ln_Pi'])


def test_profile_features_weakest_wl_element_thickness():
    """WL = the five DH elements at the base; z is the element top, so the
    lowest-tau element (z = -0.95) spans -1.0 to -0.95, i.e. 0.05 m."""
    z = np.array([-1.2, -1.1, -1.0, -0.95, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1])
    gt = np.array([520] * 5 + [320] * 8, dtype=float)
    tau = np.array([900, 800, 700, 400, 950] + [3000] * 8, dtype=float)
    ds = xr.Dataset({'z': ('e', z), 'grain_type': ('e', gt), 'shear_strength': ('e', tau),
                     'HS': ((), 1.2)})
    feat = profile_features(ds, 30.0)
    assert feat['wl_shear_strength'] == pytest.approx(tau[:5].mean())
    assert feat['wl_weakest_thickness'] == pytest.approx(0.05)
    assert feat['wl_thickness'] > feat['wl_weakest_thickness']
