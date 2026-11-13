"""POLYGON_PIECE: which piece of a multi-part release is published (2026-11-13).
Jan 18 threshold-sum WL at tau_ratio_floor 0.35 published a 3,897 m2 piece that
did not contain the trigger; 'trigger' keeps the trigger's piece."""
import inspect, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'avachain'))
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import MultiPolygon, box
import model_params as mp
import release_geometry as rg

T = Affine(1, 0, 0, 0, -1, 100)          # 1 m pixels, north-up, top at y=100
CMAP = np.zeros((100, 100), int)
CMAP[10:14, 10:14] = 7                    # trigger cluster: x 10-14, y 86-90
SMALL = box(8, 84, 16, 92)               # holds the trigger, 64 m2
BIG = box(50, 20, 90, 60)                # 1600 m2, far away
MP = MultiPolygon([SMALL, BIG])


def test_largest_keeps_the_big_piece():
    assert rg.pick_release_piece(MP, CMAP, T, 7, 'largest').equals(BIG)


def test_trigger_keeps_the_trigger_piece():
    assert rg.pick_release_piece(MP, CMAP, T, 7, 'trigger').equals(SMALL)


def test_trigger_falls_back_to_the_nearest_piece():
    near = box(20, 80, 26, 94)            # 4 m east of the trigger, holds no pixel centre
    got = rg.pick_release_piece(MultiPolygon([near, BIG]), CMAP, T, 7, 'trigger')
    assert got.equals(near)


def test_single_polygon_passes_through_and_bad_mode_raises():
    assert rg.pick_release_piece(BIG, CMAP, T, 7, 'trigger') is BIG
    with pytest.raises(ValueError):
        rg.pick_release_piece(MP, CMAP, T, 7, 'centroid')


def test_default_and_wiring():
    assert mp.POLYGON_PIECE == 'largest'
    for f in (rg.propagate_release, rg.make_release_polygon_2d):
        src = inspect.getsource(f)
        assert 'pick_release_piece(' in src
        assert 'max(poly.geoms' not in src and 'max(polygon.geoms' not in src


# ---- BFS_ADJACENCY = 'pixel' -------------------------------------------------

def test_pixel_adjacency_is_edge_sharing_only():
    m = np.array([[1, 1, 2, 0],
                  [3, 3, 2, 0],
                  [0, 0, 0, 4]])
    adj = rg.pixel_adjacency(m)
    assert adj == {1: [2, 3], 2: [1, 3], 3: [1, 2], 4: []}   # 4 touches 2 only diagonally


def test_pixel_adjacency_default_and_wiring():
    assert mp.BFS_ADJACENCY == 'knn'
    src = inspect.getsource(rg.propagate_release)
    assert "mp.BFS_ADJACENCY == 'pixel'" in src and 'pixel_adjacency(cluster_map)' in src
