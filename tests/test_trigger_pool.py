"""
tests/test_trigger_pool.py — which clusters may be triggers.

A trigger must lie wholly inside the start-zone KML. Until 2026-10-26 a
cluster qualified if any one pixel touched the KML, while the BFS gate in
release_geometry admits a cluster by its centroid pixel — so a cluster the
BFS rejects as 'outside_start_zone' could still seed the BFS as its trigger
(56 such clusters on 2026-01-18, 125 straddling the boundary in all).
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from analysis_pipeline import _start_zone_clusters


def _fixture():
    # Start zone = left three columns.
    #   1: wholly inside        2: straddles, centroid inside
    #   3: straddles, centroid outside   4: wholly outside   0: nodata
    cluster_map = np.array([
        [1, 1, 2, 2, 4],
        [1, 1, 2, 3, 3],
        [0, 1, 3, 3, 3],
    ])
    mask = np.zeros(cluster_map.shape, bool)
    mask[:, :3] = True
    return cluster_map, mask


def test_only_wholly_inside_clusters_are_eligible():
    cluster_map, mask = _fixture()
    assert _start_zone_clusters(cluster_map, mask) == {1}


def test_straddling_cluster_with_centroid_inside_is_still_excluded():
    """Stricter than the BFS centroid gate, on purpose: no partial triggers."""
    cluster_map, mask = _fixture()
    px = np.argwhere(cluster_map == 2).mean(axis=0).astype(int)
    assert mask[px[0], px[1]]
    assert 2 not in _start_zone_clusters(cluster_map, mask)


def test_nodata_pixels_inside_the_zone_do_not_count():
    cluster_map, mask = _fixture()
    assert 0 not in _start_zone_clusters(cluster_map, mask)
