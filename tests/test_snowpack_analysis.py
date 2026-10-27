"""
tests/test_snowpack_analysis.py — Unit tests for snowpack_analysis.py.

Run with:  pytest tests/test_snowpack_analysis.py -v

Covers:
  - split_wl_slab()       : basal WL/slab interface detection
  - profile_features()    : min_depth_cm shallow-WL guard
  - assign_cluster_groups(): release / adjacent / reference assignment
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from snowpack_analysis import (split_wl_slab, assign_cluster_groups,
                               profile_features)


# ---------------------------------------------------------------------------
# split_wl_slab
# ---------------------------------------------------------------------------
#
# SNOWPACK grain type codes used in tests:
#   FC 4xx (faceted crystals)  — WL
#   DH 5xx (depth hoar)        — WL
#   RG 3xx (rounded grains)    — slab
#   DF 2xx (decomposing)       — slab
#
# z convention: negative = depth below surface; most-negative = deepest layer.

class TestSplitWlSlab:

    def test_clean_basal_wl_block(self):
        # Bottom two layers FC, top layer RG
        gt = np.array([450.0, 450.0, 310.0])
        z  = np.array([-1.0,  -0.5,  -0.2])
        slab, wl, iface = split_wl_slab(gt, z)
        assert slab is not None
        # WL = bottom two layers
        assert wl[0] and wl[1]
        assert not wl[2]
        # Slab = top layer
        assert slab[2]
        assert not slab[0] and not slab[1]
        # Interface is at the top of the basal WL block
        assert iface == pytest.approx(-0.5)

    def test_single_basal_wl_layer(self):
        gt = np.array([450.0, 310.0, 310.0, 310.0])
        z  = np.array([-2.0,  -1.5,  -1.0,  -0.5])
        slab, wl, iface = split_wl_slab(gt, z)
        assert wl[0]
        assert not wl[1] and not wl[2] and not wl[3]
        assert slab[1] and slab[2] and slab[3]
        assert iface == pytest.approx(-2.0)

    def test_wl_with_df_interlayer_stops_at_first_slab(self):
        # Basal FC, then a DF interlayer, then more FC — only initial FC block is WL
        gt = np.array([450.0, 250.0, 450.0, 310.0])
        z  = np.array([-2.0,  -1.5,  -1.0,  -0.5])
        slab, wl, iface = split_wl_slab(gt, z)
        # Only the deepest layer is WL (breaks at DF)
        assert wl[0]
        assert not wl[1] and not wl[2] and not wl[3]
        assert iface == pytest.approx(-2.0)

    def test_dh_at_base_also_counts_as_wl(self):
        gt = np.array([520.0, 520.0, 310.0])
        z  = np.array([-1.5,  -0.8,  -0.3])
        slab, wl, iface = split_wl_slab(gt, z)
        assert wl[0] and wl[1]
        assert slab[2]

    def test_first_layer_is_slab_returns_none(self):
        # Bottom layer is RG (slab) → no basal WL
        gt = np.array([310.0, 450.0, 450.0])
        z  = np.array([-1.0,  -0.5,  -0.2])
        slab, wl, iface = split_wl_slab(gt, z)
        assert slab is None
        assert wl is None
        assert iface is None

    def test_near_surface_facets_only_returns_none(self):
        # Slab at base, facets only near surface
        gt = np.array([310.0, 310.0, 450.0])
        z  = np.array([-1.0,  -0.5,  -0.1])
        slab, wl, iface = split_wl_slab(gt, z)
        assert slab is None

    def test_basal_fcxr_under_fc_starts_the_wl(self):
        # The Jan 18 crown case: 990 at the ground, FC above, RG slab
        gt = np.array([990.0, 990.0, 450.0, 310.0])
        z  = np.array([-1.9,  -1.85, -1.8,  -0.5])
        slab, wl, iface = split_wl_slab(gt, z)
        assert list(wl) == [True, True, True, False]
        assert slab[3]
        assert iface == pytest.approx(-1.8)

    def test_basal_fcxr_alone_is_the_wl(self):
        gt = np.array([990.0, 310.0, 310.0])
        z  = np.array([-1.5,  -1.0,  -0.5])
        slab, wl, iface = split_wl_slab(gt, z)
        assert list(wl) == [True, False, False]

    def test_fcxr_above_basal_fc_still_ends_the_block(self):
        # Unchanged behaviour: FCxr is admitted only at the ground
        gt = np.array([450.0, 990.0, 450.0, 310.0])
        z  = np.array([-2.0,  -1.5,  -1.0,  -0.5])
        slab, wl, iface = split_wl_slab(gt, z)
        assert list(wl) == [True, False, False, False]
        assert iface == pytest.approx(-2.0)

    def test_basal_fcxr_switch_off_restores_the_old_rule(self, monkeypatch):
        import snowpack_analysis as sa
        monkeypatch.setattr(sa, 'BASAL_FCXR_IS_WL', False)
        slab, wl, iface = split_wl_slab(np.array([990.0, 450.0, 310.0]),
                                        np.array([-1.5, -1.0, -0.5]))
        assert slab is None

    def test_all_wl_layers(self):
        gt = np.array([450.0, 450.0, 450.0])
        z  = np.array([-1.0,  -0.5,  -0.2])
        slab, wl, iface = split_wl_slab(gt, z)
        # All WL, no slab
        assert wl is not None
        assert wl.all()
        assert not slab.any()

    def test_too_few_valid_layers_returns_none(self):
        gt = np.array([450.0, 450.0])
        z  = np.array([-1.0,  -0.5])
        slab, wl, iface = split_wl_slab(gt, z)
        assert slab is None

    def test_nan_layers_excluded(self):
        gt = np.array([float("nan"), 450.0, 450.0, 310.0])
        z  = np.array([float("nan"), -1.0,  -0.5,  -0.2])
        slab, wl, iface = split_wl_slab(gt, z)
        # NaN row excluded; result still valid from remaining 3 rows
        assert slab is not None
        assert wl is not None

    def test_zero_grain_type_excluded(self):
        gt = np.array([0.0, 450.0, 450.0, 310.0])
        z  = np.array([-2.0, -1.0,  -0.5,  -0.2])
        slab, wl, iface = split_wl_slab(gt, z)
        # zero gt excluded; valid result from remaining 3 rows
        assert slab is not None

    def test_slab_and_wl_masks_disjoint(self):
        gt = np.array([450.0, 450.0, 310.0, 310.0])
        z  = np.array([-2.0,  -1.5,  -0.8,  -0.3])
        slab, wl, iface = split_wl_slab(gt, z)
        # No layer can be both slab and WL
        assert not (slab & wl).any()

    def test_unsorted_input_handled(self):
        # Input deliberately in random order; function should sort by z
        gt = np.array([310.0, 450.0, 450.0, 310.0])
        z  = np.array([-0.2,  -2.0,  -1.5,  -0.8])
        slab, wl, iface = split_wl_slab(gt, z)
        # After sorting: z=-2.0 (FC), z=-1.5 (FC), z=-0.8 (RG), z=-0.2 (RG)
        # WL = bottom two; slab = top two
        assert slab is not None
        # The two original FC layers (indices 1 and 2) should be WL
        assert wl[1] and wl[2]
        assert slab[0] and slab[3]


# ---------------------------------------------------------------------------
# assign_cluster_groups
# ---------------------------------------------------------------------------

def _make_inputs(grid_size=10):
    """
    Build a synthetic 10×10 terrain for cluster group assignment.

    Layout:
      cluster 1: rows 0-2, cols 0-2  (overlaps release_mask ≥ 30%)
      cluster 2: rows 0-2, cols 7-9  (overlaps start_zone_mask ≥ 30%)
      cluster 3: rows 5-7, cols 0-3  (terrain-matched → reference)
      cluster 4: rows 8-9, cols 8-9  (low slope, far from release → unassigned)
    """
    n = grid_size
    cluster_map = np.zeros((n, n), dtype=int)
    # cluster 1
    cluster_map[0:3, 0:3] = 1
    # cluster 2
    cluster_map[0:3, 7:10] = 2
    # cluster 3
    cluster_map[5:8, 0:4] = 3
    # cluster 4
    cluster_map[8:10, 8:10] = 4

    # Flat DEM with a small gradient so slope can be computed
    # Release zone mean elevation ~3300m; match cluster 3 to be close
    dem = np.full((n, n), 3300.0)
    # cluster 4 area: much lower elevation → won't terrain-match
    dem[8:10, 8:10] = 2800.0
    # Add tiny gradient to produce a non-zero slope everywhere
    for i in range(n):
        dem[i, :] += i * 2.0   # gentle 2m/cell grade

    domain_mask = cluster_map > 0

    # release_mask covers cluster 1 cells (≥30% overlap guaranteed)
    release_mask = np.zeros((n, n), dtype=bool)
    release_mask[0:3, 0:3] = True

    # start_zone_mask covers cluster 2 cells
    start_zone_mask = np.zeros((n, n), dtype=bool)
    start_zone_mask[0:3, 7:10] = True

    return cluster_map, release_mask, start_zone_mask, dem, domain_mask


class TestAssignClusterGroups:

    def test_release_cluster_identified(self):
        args = _make_inputs()
        groups = assign_cluster_groups(*args)
        assert 1 in groups['release']

    def test_adjacent_cluster_identified(self):
        args = _make_inputs()
        groups = assign_cluster_groups(*args)
        assert 2 in groups['adjacent']

    def test_release_not_in_adjacent_or_reference(self):
        args = _make_inputs()
        groups = assign_cluster_groups(*args)
        assert 1 not in groups['adjacent']
        assert 1 not in groups['reference']

    def test_adjacent_not_in_release(self):
        args = _make_inputs()
        groups = assign_cluster_groups(*args)
        assert 2 not in groups['release']

    def test_groups_are_mutually_exclusive(self):
        args = _make_inputs()
        groups = assign_cluster_groups(*args)
        rel = groups['release']
        adj = groups['adjacent']
        ref = groups['reference']
        assert len(rel & adj) == 0
        assert len(rel & ref) == 0
        assert len(adj & ref) == 0

    def test_no_release_mask_skips_reference(self):
        cluster_map, _, start_zone_mask, dem, domain_mask = _make_inputs()
        # Empty release mask
        empty_release = np.zeros_like(cluster_map, dtype=bool)
        groups = assign_cluster_groups(
            cluster_map, empty_release, start_zone_mask, dem, domain_mask
        )
        # With no release zone, reference group must be empty
        assert len(groups['reference']) == 0

    def test_no_release_mask_still_assigns_adjacent(self):
        cluster_map, _, start_zone_mask, dem, domain_mask = _make_inputs()
        empty_release = np.zeros_like(cluster_map, dtype=bool)
        groups = assign_cluster_groups(
            cluster_map, empty_release, start_zone_mask, dem, domain_mask
        )
        assert 2 in groups['adjacent']

    def test_returns_dict_with_three_keys(self):
        args = _make_inputs()
        groups = assign_cluster_groups(*args)
        assert set(groups.keys()) == {'release', 'adjacent', 'reference'}

    def test_below_threshold_cluster_not_release(self):
        """A cluster with < 30% overlap with release mask should not be release."""
        n = 10
        cluster_map = np.zeros((n, n), dtype=int)
        # Cluster 5: 10 cells, only 2 overlap release (20% < 30%)
        cluster_map[0, 0:10] = 5        # 10 cells total
        release_mask = np.zeros((n, n), dtype=bool)
        release_mask[0, 0:2] = True     # only 2 cells overlap

        start_zone_mask = np.zeros((n, n), dtype=bool)
        dem = np.full((n, n), 3300.0)
        domain_mask = cluster_map > 0

        groups = assign_cluster_groups(cluster_map, release_mask, start_zone_mask,
                                       dem, domain_mask)
        assert 5 not in groups['release']

    def test_exactly_at_threshold_is_release(self):
        """A cluster with exactly 30% overlap should be assigned to release."""
        n = 10
        cluster_map = np.zeros((n, n), dtype=int)
        cluster_map[0, 0:10] = 6        # 10 cells total
        release_mask = np.zeros((n, n), dtype=bool)
        release_mask[0, 0:3] = True     # exactly 3/10 = 30%

        start_zone_mask = np.zeros((n, n), dtype=bool)
        dem = np.full((n, n), 3300.0)
        domain_mask = cluster_map > 0

        groups = assign_cluster_groups(cluster_map, release_mask, start_zone_mask,
                                       dem, domain_mask)
        assert 6 in groups['release']


# ---------------------------------------------------------------------------
# profile_features — min_depth_cm guard
# ---------------------------------------------------------------------------
#
# The parameter was in the signature and documented but never referenced in
# the body until 2026-10-13, so a shallow weak layer produced a full feature
# set instead of being rejected. ~/avalanche-release-areas implements it; this
# pins the behaviour so the two repos cannot drift apart again.

_Z_GRID = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6,
           -0.5, -0.4, -0.3, -0.2, -0.1]


def _profile(grain_types, hs=1.2):
    """One cluster at one timestep: DH 5xx = weak layer, RG 3xx = slab."""
    return xr.Dataset({
        'z':          ('e', np.array(_Z_GRID, dtype=float)),
        'grain_type': ('e', np.array(grain_types, dtype=float)),
        'HS':         ((), hs),
    })


# WL from the base up to z=-0.2 leaves a 0.2 m slab; WL in the bottom two
# elements only leaves a 1.1 m slab.
_THIN_SLAB  = [520] * 11 + [320]
_THICK_SLAB = [520] * 2 + [320] * 10


def test_shallow_wl_is_rejected_above_min_depth():
    """A 0.2 m slab is shallower than the 0.3 m floor — no slab features."""
    feat = profile_features(_profile(_THIN_SLAB), 30.0)
    assert 'slab_thickness' not in feat
    assert feat == {'hs': pytest.approx(1.2)}


def test_deep_wl_is_kept_at_same_min_depth():
    """A 1.1 m slab clears the same floor and yields the full feature set."""
    feat = profile_features(_profile(_THICK_SLAB), 30.0)
    assert feat['slab_thickness'] == pytest.approx(1.1)
    assert 'E_slab' in feat


def test_min_depth_zero_disables_the_guard():
    """min_depth_cm=0 must not reject anything — guard is opt-in by value."""
    feat = profile_features(_profile(_THIN_SLAB), 0.0)
    assert feat['slab_thickness'] == pytest.approx(0.2)


def test_guard_boundary_is_inclusive_of_equal_depth():
    """-interface_z == min_depth_cm/100 is kept; the test is strictly less."""
    feat = profile_features(_profile(_THIN_SLAB), 20.0)
    assert feat['slab_thickness'] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# tau_g thresholds — parity with ~/avalanche-release-areas
# ---------------------------------------------------------------------------
#
# These are two deliberately different constants, which is easy to mistake for
# a bug and "fix" by collapsing them: commit 8de10d4 did exactly that, lowering
# the feature floor to 40 to match the trigger gate. The standalone repo keeps
# them apart, so avachain must too. Both values are pinned here because the
# divergence is invisible on Jan 18 (no cluster falls in the 40-50 Pa band) and
# so cannot be caught by the acceptance test.

def test_tau_g_feature_floor_matches_standalone():
    from snowpack_analysis import TAU_G_FEATURE_FLOOR
    assert TAU_G_FEATURE_FLOOR == 50.0


def test_feature_floor_sits_above_the_trigger_gate():
    """The 40-50 Pa dead band is intentional, not an oversight.

    If someone closes it, that is a modelling decision that has to be made in
    ~/avalanche-release-areas first — not a local cleanup.
    """
    from snowpack_analysis import TAU_G_FEATURE_FLOOR
    trigger_gate = 40.0
    assert TAU_G_FEATURE_FLOOR > trigger_gate


# ---------------------------------------------------------------------------
# profile_features — critical cut length r_c at the weak layer
# ---------------------------------------------------------------------------
#
# Until 2026-10-20 rc_wl was NaN for every cluster: RC_VAR named the variable
# 'critical_cut_length', but xsnow stores SNOWPACK code 0606 as 'rc_flat' (the
# old name is only an alias). The KeyError was swallowed by a bare except, so
# nothing reported it. SNOWPACK also leaves the basal element undefined (-999),
# and writes r_c <= 0 where the WL cannot carry the slab (tau_p < tau_g in
# StabilityAlgorithms::CriticalCutLength) — that is the MOST unstable case, so
# it must count as 0 m, not be dropped as bad data.

_WL3_SLAB = [520] * 3 + [320] * 9      # WL = bottom three elements


def _rc_profile(rc_wl_elements):
    ds = _profile(_WL3_SLAB)
    rc = np.full(len(_Z_GRID), 0.9)    # slab values, must never be read
    rc[:3] = rc_wl_elements
    return ds.assign(rc_flat=('e', rc))


def test_rc_wl_reads_the_rc_flat_variable():
    feat = profile_features(_rc_profile([-999.0, 0.4, 0.2]), 0.0)
    assert feat['rc_wl'] == pytest.approx(0.3)


def test_rc_wl_excludes_the_undefined_basal_element():
    """-999 is SNOWPACK's undefined marker, not a length."""
    feat = profile_features(_rc_profile([-999.0, 0.5, 0.5]), 0.0)
    assert feat['rc_wl'] == pytest.approx(0.5)


def test_rc_wl_counts_non_positive_values_as_zero():
    """r_c <= 0 means tau_p < tau_g: self-propagating, i.e. r_c = 0 m."""
    feat = profile_features(_rc_profile([-999.0, -0.13, 0.6]), 0.0)
    assert feat['rc_wl'] == pytest.approx(0.3)


def test_rc_wl_is_nan_when_every_wl_element_is_undefined():
    feat = profile_features(_rc_profile([-999.0, -999.0, -999.0]), 0.0)
    assert np.isnan(feat['rc_wl'])


def test_rc_wl_missing_variable_warns_instead_of_silent_nan():
    with pytest.warns(UserWarning, match='rc_flat'):
        feat = profile_features(_profile(_WL3_SLAB), 0.0)
    assert np.isnan(feat['rc_wl'])


def test_snapshot_gap_refuses_dates_past_the_store():
    """analyze must not read the last step for a date the store does not reach."""
    import pandas as pd
    from analyze_release_zone import MAX_SNAPSHOT_GAP, snapshot_gap
    times = pd.date_range('2025-11-26 18:00', '2026-01-18 18:00', freq='6h').values
    assert snapshot_gap(times, '2026-01-18') <= MAX_SNAPSHOT_GAP
    assert snapshot_gap(times, '2026-03-07') > MAX_SNAPSHOT_GAP


# ---------------------------------------------------------------------------
# Facet knockdown source (grain class since 2026-10-27)
# ---------------------------------------------------------------------------

class TestFacetSource:
    """The feature path must pass grain_type through, or 'grain_class' would
    fail (ValueError) or silently fall back. Jan 18 cannot see a wiring slip
    that leaves every sigma_t column plausible, so it is pinned here."""

    def _ds(self):
        n = 2
        return xr.Dataset({
            'density':    ('layer', np.array([250.0, 250.0])),
            'sphericity': ('layer', np.array([0.9, 0.6])),
            'dendricity': ('layer', np.zeros(n)),
            'grain_type': ('layer', np.array([330.0, 990.0])),
        })

    def test_default_is_grain_class(self):
        import snowpack_analysis as sa
        assert sa.FACET_SOURCE == 'grain_class'

    def test_grain_type_reaches_the_aggregation(self, monkeypatch):
        import snowpack_analysis as sa
        dz, z, sel = np.array([0.5, 0.5]), np.array([-0.25, -0.75]), np.ones(2, bool)
        monkeypatch.setattr(sa, 'FACET_SOURCE', 'grain_class')
        by_class = sa._slab_layer_aggregate(self._ds(), sel, dz, z)
        monkeypatch.setattr(sa, 'FACET_SOURCE', 'sphericity')
        by_sph = sa._slab_layer_aggregate(self._ds(), sel, dz, z)
        assert by_class['f_mean_weighted'] == pytest.approx(0.5)
        assert by_sph['f_mean_weighted'] == pytest.approx(0.25)
        assert by_class['sigma_t_mean__jamieson1990'] < by_sph['sigma_t_mean__jamieson1990']
