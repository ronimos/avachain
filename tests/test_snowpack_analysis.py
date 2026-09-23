"""
tests/test_snowpack_analysis.py — Unit tests for snowpack_analysis.py.

Run with:  pytest tests/test_snowpack_analysis.py -v

Covers:
  - split_wl_slab()       : basal WL/slab interface detection
  - assign_cluster_groups(): release / adjacent / reference assignment
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from snowpack_analysis import split_wl_slab, assign_cluster_groups


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
