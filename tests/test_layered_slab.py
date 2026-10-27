"""Tests for the layered-slab sigma_t / E aggregation feeding Eq. 20.

Ported from ~/avalanche-release-areas/tests/test_layered_slab.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import arrest_indices as ai  # noqa: E402

RHO, DZ_N = 300.0, 5
SP_RG, SP_FC = 0.9, 0.1


def _arrest(sigma_t, E, h=1.0, D_wl=0.05, theta=50.0, psi=35.0, rho=RHO):
    """Eq. 20 with a given sigma_t / E, everything else fixed."""
    Lam = ai.elastic_length(E, h, D_wl, 0.2e6)
    tau_g = ai.gravitational_shear(rho, h, psi)
    k_f = ai.tension_gradient(rho, psi)
    L_t = ai.tensile_length(sigma_t, k_f)
    return ai.arrest_length(tau_g, theta, Lam, sigma_t, 1.0, L_t)[0]


class TestFacetedness:
    def test_non_dendritic_is_one_minus_sphericity(self):
        f = ai.facetedness([0.9, 0.1], [0.0, 0.0])
        assert f == pytest.approx([0.1, 0.9])

    def test_dendritic_layers_are_zero(self):
        f = ai.facetedness([0.1, 0.1], [0.0, 0.5])
        assert f == pytest.approx([0.9, 0.0])

    def test_clipped_to_unit_interval(self):
        f = ai.facetedness([-0.2, 1.4], [0.0, 0.0])
        assert np.all((f >= 0.0) & (f <= 1.0))

    def test_sp_ref_anchors_rounded_snow_to_zero(self):
        f = ai.facetedness([0.86, 0.2], [0.0, 0.0], sp_ref=0.86)
        assert f[0] == pytest.approx(0.0)
        assert f[1] > 0.5

    def test_facet_factor_halves_strength_at_f_one(self):
        st = ai.layer_tensile_strength(RHO, 1.0, a=0.5)
        assert st == pytest.approx(0.5 * ai.slab_tensile_strength(RHO))


class TestFacetednessGrainClass:
    """f by primary grain class (Jamieson & Johnston 1990 grouped by class)."""

    def test_fc_dh_fcxr_are_faceted(self):
        f = ai.facetedness_grain_class([440, 450, 550, 990, 942])
        assert f == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])

    def test_other_classes_are_not(self):
        # PP, DF, RG, SH, MF, IF, and RG with a faceted secondary (34x)
        f = ai.facetedness_grain_class([110, 220, 330, 660, 772, 880, 340])
        assert f == pytest.approx(np.zeros(7))

    def test_missing_codes_are_zero(self):
        f = ai.facetedness_grain_class([np.nan, 0.0, -1.0, 440.0])
        assert f == pytest.approx([0.0, 0.0, 0.0, 1.0])

    def test_rounding_facets_differ_from_sphericity(self):
        # The reason for the switch: FCxr with sphericity 0.6 gets f = 0.4 from
        # 1 - sphericity but is fully faceted by class.
        assert ai.facetedness([0.6], [0.0]) == pytest.approx([0.4])
        assert ai.facetedness_grain_class([990]) == pytest.approx([1.0])


class TestAggregateFacetSource:
    rho = np.array([250.0, 250.0])
    dz = np.array([0.5, 0.5])
    sp = np.array([0.9, 0.6])
    dd = np.zeros(2)
    gt = np.array([330.0, 990.0])

    def test_grain_class_knocks_down_the_fcxr_layer_fully(self):
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd,
                                grain_type_i=self.gt, facet_source='grain_class')
        st_rg = ai.slab_tensile_strength(250.0)
        assert agg['f_mean_weighted'] == pytest.approx(0.5)
        assert agg['sigma_t_mean'] == pytest.approx(st_rg * (0.5 * 1.0 + 0.5 * 0.5))

    def test_sphericity_ignores_grain_type(self):
        a = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd)
        b = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd,
                              grain_type_i=self.gt, facet_source='sphericity')
        assert a['sigma_t_mean'] == b['sigma_t_mean']
        assert a['f_mean_weighted'] == pytest.approx(0.25)   # (0.1 + 0.4) / 2

    def test_grain_class_needs_grain_type(self):
        with pytest.raises(ValueError, match='grain_type_i'):
            ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd,
                              facet_source='grain_class')

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError, match='facet_source'):
            ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd,
                              grain_type_i=self.gt, facet_source='dendricity')

    def test_extra_relations_use_the_same_f(self):
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd,
                                grain_type_i=self.gt, facet_source='grain_class',
                                st_relations_extra=('jamieson1990',))
        st = ai.slab_tensile_strength_jamieson1990(self.rho) * (1 - 0.5 * np.array([0.0, 1.0]))
        assert agg['sigma_t_mean__jamieson1990'] == pytest.approx(
            ai.tensile_strength_mean(st, self.dz))


class TestHomogeneousSlab:
    """A uniform slab must give both bounds equal, and equal to current Eq. 20."""

    rho = np.full(DZ_N, RHO)
    dz = np.full(DZ_N, 0.2)
    sp = np.full(DZ_N, 1.0)       # f = 0 -> sigma_t is the unmodified relation
    dd = np.zeros(DZ_N)

    def test_both_bounds_identical(self):
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd)
        assert agg['sigma_t_wl'] == pytest.approx(agg['sigma_t_mean'])

    def test_bounds_equal_the_bulk_relation(self):
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd)
        assert agg['sigma_t_mean'] == pytest.approx(ai.slab_tensile_strength(RHO))
        assert agg['E_eff'] == pytest.approx(ai.slab_modulus(RHO))

    def test_arrest_length_matches_current_eq20(self):
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd)
        h = float(self.dz.sum())
        current = _arrest(ai.slab_tensile_strength(RHO), ai.slab_modulus(RHO), h=h)
        for st in (agg['sigma_t_mean'], agg['sigma_t_wl']):
            assert _arrest(st, agg['E_eff'], h=h) == pytest.approx(current)

    def test_f_zero_everywhere_reproduces_current_sigma_t(self):
        # Note: synthetic only. Real profiles have RG sphericity ~0.86, so
        # f is never 0 there; see ~/avalanche-release-areas/docs/release_area_methods.md §4(c).
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd)
        assert agg['f_mean_weighted'] == pytest.approx(0.0)
        assert agg['sigma_t_mean'] == pytest.approx(ai.slab_tensile_strength(RHO))


class TestThinFacetedLayer:
    """One thin FC layer must drive the weakest-link bound below the mean."""

    rho = np.array([RHO, RHO, RHO, RHO, RHO])
    dz = np.array([0.20, 0.20, 0.02, 0.20, 0.20])     # 2 cm FC layer, index 2
    sp = np.array([SP_RG, SP_RG, SP_FC, SP_RG, SP_RG])
    dd = np.zeros(5)

    def test_weakest_link_below_mean(self):
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd)
        assert agg['sigma_t_wl'] < agg['sigma_t_mean']

    def test_controlling_layer_is_the_fc_layer(self):
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd,
                                depth_i=np.cumsum(self.dz))
        assert agg['wl_ctrl_index'] == 2
        assert agg['wl_ctrl_thickness'] == pytest.approx(0.02)

    def test_thin_layer_barely_moves_the_mean_bound(self):
        # 2 cm of 82 cm: the mean bound is diluted, the weakest link is not.
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd)
        no_fc = ai.aggregate_slab(self.rho, self.dz, np.full(5, SP_RG), self.dd)
        assert agg['sigma_t_mean'] / no_fc['sigma_t_mean'] > 0.98
        assert agg['sigma_t_wl'] / no_fc['sigma_t_wl'] < 0.95

    def test_weakest_link_gives_shorter_arrest_length(self):
        agg = ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd)
        h = float(self.dz.sum())
        a_mean = _arrest(agg['sigma_t_mean'], agg['E_eff'], h=h)
        a_wl = _arrest(agg['sigma_t_wl'], agg['E_eff'], h=h)
        assert a_wl < a_mean
        # A_ca ~ sigma_t^1.5 at fixed E.
        assert a_mean / a_wl == pytest.approx(
            (agg['sigma_t_mean'] / agg['sigma_t_wl']) ** 1.5, rel=1e-9)


class TestBoundOrdering:
    """sigma_t_wl <= sigma_t_mean for any layering, by construction."""

    @pytest.mark.parametrize('seed', range(8))
    def test_invariant_on_random_slabs(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 40))
        rho = rng.uniform(120.0, 450.0, n)
        dz = rng.uniform(0.003, 0.04, n)
        sp = rng.uniform(0.0, 1.0, n)
        dd = np.where(rng.random(n) < 0.1, rng.random(n), 0.0)
        agg = ai.aggregate_slab(rho, dz, sp, dd)
        assert agg['sigma_t_wl'] <= agg['sigma_t_mean'] * (1 + 1e-12)

    def test_equality_only_when_failure_strain_is_uniform(self):
        # sigma_t_i/E_i constant requires uniform rho and uniform f.
        rho = np.array([200.0, 400.0])
        dz = np.array([0.1, 0.1])
        agg = ai.aggregate_slab(rho, dz, np.ones(2), np.zeros(2))
        assert agg['sigma_t_wl'] < agg['sigma_t_mean']


class TestEffectiveModulus:
    def test_iso_strain_weighting(self):
        E = np.array([1.0e6, 5.0e6])
        dz = np.array([0.3, 0.1])
        assert ai.effective_modulus(E, dz) == pytest.approx(
            (1.0e6 * 0.3 + 5.0e6 * 0.1) / 0.4)

    def test_nan_for_zero_thickness(self):
        assert np.isnan(ai.effective_modulus([1.0e6], [0.0]))


class TestLigamentBound:
    """Hypothesis branch: off unless K_Ic is supplied."""

    def test_absent_without_k_ic(self):
        agg = ai.aggregate_slab(np.full(3, RHO), np.full(3, 0.1),
                                np.full(3, 0.5), np.zeros(3))
        assert np.isnan(agg['sigma_t_ligament'])

    def test_sent_factor_at_small_crack(self):
        assert ai.sent_geometry_factor(0.0) == pytest.approx(1.122)

    def test_sent_factor_outside_validity_is_nan(self):
        assert np.isnan(ai.sent_geometry_factor(0.8))

    def test_embedded_factor_grows_with_crack(self):
        assert ai.embedded_geometry_factor(0.5) > ai.embedded_geometry_factor(0.1)

    def test_edge_crack_strength_scales_as_inverse_sqrt_a(self):
        s1 = ai.edge_crack_strength(1.0e3, 0.01, 1.0)
        s2 = ai.edge_crack_strength(1.0e3, 0.04, 1.0)
        assert s1 / s2 == pytest.approx(2.0, rel=0.05)

    def test_edge_crack_rejects_bad_geometry(self):
        assert np.isnan(ai.edge_crack_strength(1.0e3, 1.0, 1.0))
        assert np.isnan(ai.edge_crack_strength(1.0e3, -0.1, 1.0))


class TestDegenerateInputs:
    def test_all_nan_when_no_usable_layer(self):
        agg = ai.aggregate_slab([np.nan], [np.nan], [np.nan], [np.nan])
        assert agg['n_slab_layers'] == 0
        assert np.isnan(agg['sigma_t_mean']) and np.isnan(agg['sigma_t_wl'])

    def test_nonpositive_thickness_dropped(self):
        agg = ai.aggregate_slab([RHO, RHO], [0.1, 0.0], [1.0, 0.0], [0.0, 0.0])
        assert agg['n_slab_layers'] == 1
        assert agg['sigma_t_mean'] == pytest.approx(ai.slab_tensile_strength(RHO))


class TestKIcRelations:
    def test_kirchner2000_value_and_units(self):
        # 7.84 (rho/917)^2.3 kPa m^0.5, returned in Pa m^0.5
        assert ai.k_ic(300.0, 'kirchner2000') == pytest.approx(
            7.84e3 * (300.0 / 917.0) ** 2.3)

    def test_nan_outside_validity_when_clamp_disabled(self):
        assert np.isnan(ai.k_ic(90.0, 'kirchner2000', clamp=False))
        assert np.isnan(ai.k_ic(600.0, 'kirchner2000', clamp=False))
        assert np.isfinite(ai.k_ic(100.0, 'kirchner2000', clamp=False))
        assert np.isfinite(ai.k_ic(540.0, 'kirchner2000', clamp=False))

    def test_clamping_is_the_default_and_pins_to_the_range(self):
        lo, hi = ai.k_ic_valid_range('kirchner2000')
        assert (lo, hi) == (100.0, 540.0)
        assert ai.k_ic(50.0, 'kirchner2000') == pytest.approx(
            ai.k_ic(lo, 'kirchner2000'))
        assert ai.k_ic(900.0, 'kirchner2000') == pytest.approx(
            ai.k_ic(hi, 'kirchner2000'))

    def test_clamped_count(self):
        assert ai.k_ic_n_clamped([50.0, 300.0, 600.0], 'kirchner2000') == 2
        assert ai.k_ic_n_clamped([150.0, 250.0], 'kirchner2000') == 0
        # schweizer2004 tops out at 300, so typical slab layers are clamped
        assert ai.k_ic_n_clamped([320.0, 400.0], 'schweizer2004') == 2

    def test_schweizer2004_eq8_uses_the_printed_exponent(self):
        # Default is the paper's printed 1.9 since 2026-10-09.
        # rho = 300, d_max = 1 mm -> ~1.32 kPa m^0.5
        got = ai.k_ic(300.0, 'schweizer2004', d_max=1e-3)
        assert got == pytest.approx(
            350.0 * (300.0 / 917.0) ** 1.9 / np.sqrt(1e-3))
        assert got / 1e3 == pytest.approx(1.32, abs=0.01)

    def test_schweizer2004_printed_exponent_is_the_default(self):
        assert ai.SCH2004_EXP == ai.SCH2004_EXP_PAPER == 1.9
        assert ai.SCH2004_EXP_ROUND == 2.0

    def test_schweizer2004_rounded_exponent_reproduces_older_output(self):
        # What the default gave before 2026-10-09: 11% lower at rho = 300.
        got = ai.k_ic_schweizer2004(300.0, d_max=1e-3,
                                    exp=ai.SCH2004_EXP_ROUND)
        assert got == pytest.approx(350.0 * (300.0 / 917.0) ** 2 / np.sqrt(1e-3))
        assert got / 1e3 == pytest.approx(1.18, abs=0.01)
        assert got < ai.k_ic(300.0, 'schweizer2004', d_max=1e-3)

    def test_schweizer2004_scales_as_inverse_sqrt_dmax(self):
        a = ai.k_ic(250.0, 'schweizer2004', d_max=1e-3)
        b = ai.k_ic(250.0, 'schweizer2004', d_max=4e-3)
        assert a / b == pytest.approx(2.0)

    def test_schweizer2004_valid_range(self):
        assert ai.k_ic_valid_range('schweizer2004') == (80.0, 300.0)

    def test_schweizer2004_needs_dmax(self):
        assert np.isnan(ai.k_ic(250.0, 'schweizer2004'))

    def test_schweizer2004_is_the_default(self):
        assert ai.K_IC_DEFAULT == 'schweizer2004'

    def test_kirchner_is_a_lower_bound_vs_schweizer(self):
        # at d_max = 1 mm, within both validity windows
        assert ai.k_ic(250.0, 'kirchner2000') < ai.k_ic(
            250.0, 'schweizer2004', d_max=1e-3)

    def test_only_borstad_remains_gated(self):
        with pytest.raises(NotImplementedError):
            ai.k_ic(300.0, 'borstad2013')
        assert ai.K_IC_PENDING == ('borstad2013',)

    def test_unknown_relation_raises(self):
        with pytest.raises(ValueError):
            ai.k_ic(300.0, 'nope')

    def test_characteristic_length(self):
        assert ai.characteristic_length(600.0, 3000.0) == pytest.approx(0.04)


class TestElHaddad:
    K, H, ST = 600.0, 1.0, 3000.0

    def test_zero_crack_recovers_ligament_strength_exactly(self):
        # a0 uses the same F(a/h) as sigma_c, so the a -> 0 limit is exact.
        s = ai.edge_crack_strength(self.K, 1e-12, self.H, sigma_t_lig=self.ST)
        assert s == pytest.approx(self.ST, rel=1e-9)

    def test_zero_crack_exact_at_several_scales(self):
        for K in (200.0, 600.0, 2000.0):
            for st in (1000.0, 3000.0, 6000.0):
                a0 = ai.intrinsic_crack_length(K, st, self.H, a=1e-14)
                if not np.isfinite(a0):
                    continue        # a0 >= h, covered by the test below
                got = ai.edge_crack_strength(K, 1e-14, self.H, sigma_t_lig=st)
                assert got == pytest.approx(st, rel=1e-9)

    def test_nan_when_a0_exceeds_slab_thickness(self):
        # K/sigma_t = 2 gives l_ch = 4 m: the intrinsic flaw size exceeds a 1 m
        # slab, so the LEFM geometry is meaningless and the bound is withheld.
        assert np.isnan(ai.intrinsic_crack_length(2000.0, 1000.0, 1.0, a=1e-14))
        assert np.isnan(ai.edge_crack_strength(2000.0, 1e-14, 1.0,
                                               sigma_t_lig=1000.0))

    def test_self_consistent_option_is_only_approximate(self):
        # Kept as an option; its a0 is a property of material+geometry alone,
        # at the cost of the exact a -> 0 limit.
        s = ai.edge_crack_strength(self.K, 1e-12, self.H, sigma_t_lig=self.ST,
                                   self_consistent_a0=True)
        assert s == pytest.approx(self.ST, rel=2e-3)
        assert s != pytest.approx(self.ST, rel=1e-9)

    def test_large_crack_matches_lefm(self):
        a = 0.5   # >> a0 = (1/pi)(K/(F sigma_t))^2 ~ 1e-2 m here
        eh = ai.edge_crack_strength(self.K, a, self.H, sigma_t_lig=self.ST)
        lefm = ai.edge_crack_strength(self.K, a, self.H, model='lefm')
        assert eh == pytest.approx(lefm, rel=0.05)
        assert eh < lefm        # the correction is always softening

    def test_elhaddad_never_exceeds_ligament_strength(self):
        for a in (1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5):
            s = ai.edge_crack_strength(self.K, a, self.H, sigma_t_lig=self.ST)
            assert s <= self.ST * (1 + 1e-9)

    def test_monotonic_decreasing_in_crack_length(self):
        s = [ai.edge_crack_strength(self.K, a, self.H, sigma_t_lig=self.ST)
             for a in (1e-4, 1e-3, 1e-2, 0.1, 0.4)]
        assert all(b < a for a, b in zip(s, s[1:]))

    def test_lefm_option_is_the_uncorrected_form(self):
        a = 0.02
        got = ai.edge_crack_strength(self.K, a, self.H, model='lefm')
        F = float(ai.sent_geometry_factor(a / self.H))
        assert got == pytest.approx(self.K / (F * np.sqrt(np.pi * a)))

    def test_lefm_used_when_no_ligament_strength_given(self):
        a = 0.02
        assert ai.edge_crack_strength(self.K, a, self.H) == pytest.approx(
            ai.edge_crack_strength(self.K, a, self.H, model='lefm'))

    def test_a0_uses_F_at_the_actual_crack_length(self):
        a = 0.05
        a0 = ai.intrinsic_crack_length(self.K, self.ST, self.H, a=a)
        F = float(ai.sent_geometry_factor(a / self.H))
        assert a0 == pytest.approx((self.K / (F * self.ST)) ** 2 / np.pi, rel=1e-12)

    def test_a0_requires_a_unless_self_consistent(self):
        with pytest.raises(ValueError):
            ai.intrinsic_crack_length(self.K, self.ST, self.H)
        assert np.isfinite(ai.intrinsic_crack_length(
            self.K, self.ST, self.H, self_consistent=True))

    def test_a0_self_consistent_matches_F_at_a0(self):
        a0 = ai.intrinsic_crack_length(self.K, self.ST, self.H,
                                       self_consistent=True)
        F = float(ai.sent_geometry_factor(a0 / self.H))
        assert a0 == pytest.approx((self.K / (F * self.ST)) ** 2 / np.pi, rel=1e-9)

    def test_bad_model_raises(self):
        with pytest.raises(ValueError):
            ai.edge_crack_strength(self.K, 0.01, self.H, model='bogus')

    def test_embedded_branch_also_corrected(self):
        a = 0.02
        eh = ai.edge_crack_strength(self.K, a, self.H, embedded=True,
                                    sigma_t_lig=self.ST)
        lefm = ai.edge_crack_strength(self.K, a, self.H, embedded=True,
                                      model='lefm')
        assert eh < lefm


class TestLigamentFromIntactLayers:
    """K_Ic and sigma_t_lig must come from the intact layers, not the crack."""

    rho = np.array([320.0, 320.0, 255.0, 320.0])   # index 2 = weak FC layer
    dz = np.array([0.2, 0.2, 0.01, 0.2])
    sp = np.array([0.95, 0.95, 0.05, 0.95])
    dd = np.zeros(4)

    def _agg(self, **kw):
        return ai.aggregate_slab(self.rho, self.dz, self.sp, self.dd,
                                 depth_i=np.cumsum(self.dz),
                                 k_ic_relation='kirchner2000', **kw)

    def test_ligament_excludes_the_controlling_layer(self):
        agg = self._agg()
        assert agg['wl_ctrl_index'] == 2
        # Intact layers are all rho=320, so both weighted means are that layer's.
        assert agg['K_Ic_ligament'] == pytest.approx(
            float(ai.k_ic(320.0, 'kirchner2000')))
        assert agg['sigma_t_lig_intact'] == pytest.approx(
            float(ai.layer_tensile_strength(320.0, ai.facetedness(0.95, 0.0))))

    def test_ligament_strength_above_controlling_layer_strength(self):
        agg = self._agg()
        assert agg['sigma_t_lig_intact'] > agg['sigma_t_wl']

    def test_a0_and_l_ch_emitted(self):
        agg = self._agg()
        assert np.isfinite(agg['a0_ligament'])
        assert agg['l_ch_ligament'] == pytest.approx(
            (agg['K_Ic_ligament'] / agg['sigma_t_lig_intact']) ** 2)

    def test_elhaddad_softer_than_lefm(self):
        # Adding a0 to a lengthens the denominator, so El Haddad always sits
        # below LEFM; the gap is what stops LEFM diverging at short cracks.
        assert self._agg(ligament_model='elhaddad')['sigma_t_ligament'] < \
            self._agg(ligament_model='lefm')['sigma_t_ligament']

    def test_lefm_overshoots_the_ligament_strength_here(self):
        # The failure El Haddad exists to fix: for a 10 mm crack, plain LEFM
        # predicts the slab is stronger than its own intact snow.
        agg_l = self._agg(ligament_model='lefm')
        agg_e = self._agg(ligament_model='elhaddad')
        assert agg_l['sigma_t_ligament'] > agg_l['sigma_t_lig_intact']
        assert agg_e['sigma_t_ligament'] < agg_e['sigma_t_lig_intact']


class TestEModulusRelations:
    def test_vanherwijnen2016_is_the_default(self):
        assert ai.E_DEFAULT == 'vanherwijnen2016'
        assert ai.slab_modulus(250.0) == pytest.approx(0.93 * 250.0 ** 2.8)

    def test_project_fit_still_available_unchanged(self):
        assert ai.slab_modulus(300.0, 'project_fit') == pytest.approx(4.0e6)
        assert ai.slab_modulus(250.0, 'project_fit') == pytest.approx(
            (250.0 / 300.0) ** 2.5 * 4.0e6)

    def test_vanherwijnen_near_the_meloche_calibration_point(self):
        # C_FIT was fitted at E = 4 MPa, rho = 250.
        assert ai.slab_modulus(250.0) / 4.0e6 == pytest.approx(1.204, abs=0.01)
        assert ai.slab_modulus(250.0, 'project_fit') / 4.0e6 == pytest.approx(
            0.634, abs=0.01)

    def test_unknown_relation_raises(self):
        with pytest.raises(ValueError):
            ai.slab_modulus(300.0, 'nope')

    def test_homogeneous_E_eff_follows_the_selected_relation(self):
        rho, dz = np.full(4, 280.0), np.full(4, 0.1)
        sp, dd = np.ones(4), np.zeros(4)
        for rel in ('vanherwijnen2016', 'project_fit'):
            agg = ai.aggregate_slab(rho, dz, sp, dd, e_relation=rel)
            assert agg['E_eff'] == pytest.approx(float(ai.slab_modulus(280.0, rel)))

    def test_extra_relations_emitted_alongside(self):
        agg = ai.aggregate_slab(np.full(3, 300.0), np.full(3, 0.1),
                                np.ones(3), np.zeros(3),
                                e_relations_extra=('project_fit',))
        assert agg['E_eff__project_fit'] == pytest.approx(4.0e6)
        assert agg['E_eff'] != pytest.approx(4.0e6)


class TestKIcVariants:
    rho = np.array([290.0, 290.0, 240.0, 290.0])
    dz = np.array([0.2, 0.2, 0.01, 0.2])
    sp = np.array([0.95, 0.95, 0.05, 0.95])
    dd = np.zeros(4)
    gsz = np.full(4, 6e-4)

    def test_variants_suffixed_and_dmax_factor_applied(self):
        agg = ai.aggregate_slab(
            self.rho, self.dz, self.sp, self.dd,
            depth_i=np.cumsum(self.dz), grain_size_i=self.gsz,
            k_ic_relation='schweizer2004',
            k_ic_variants=[('d1', 'schweizer2004', 1.0),
                           ('d2', 'schweizer2004', 2.0)])
        # K_Ic ~ 1/sqrt(d_max), so doubling d_max divides K_Ic by sqrt(2)
        assert agg['K_Ic_ligament__d1'] / agg['K_Ic_ligament__d2'] == \
            pytest.approx(np.sqrt(2.0))
        assert agg['K_Ic_ligament__d1'] == pytest.approx(agg['K_Ic_ligament'])

    def test_clamped_counts_per_variant(self):
        agg = ai.aggregate_slab(
            np.array([320.0, 320.0, 240.0, 320.0]), self.dz, self.sp, self.dd,
            depth_i=np.cumsum(self.dz), grain_size_i=self.gsz,
            k_ic_variants=[('kir', 'kirchner2000', 1.0),
                           ('sch', 'schweizer2004', 1.0)])
        # intact layers are rho=320: inside Kirchner (100-540), above
        # Schweizer's 300 ceiling
        assert agg['n_rho_clamped__kir'] == 0
        assert agg['n_rho_clamped__sch'] == 3
