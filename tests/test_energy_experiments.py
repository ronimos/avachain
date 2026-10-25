"""Geometry behind the Step 7 metrics: fall-line frame, crown/flank offsets, boundary segments.

Directions are checked in the map frame (x east, y north) at known bearings,
as test_terrain_direction.py does for the BFS helpers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'avachain'))

import numpy as np
import pytest
from shapely.geometry import box

from energy_experiments import SlopeFrame, boundary_segments, crown_flank_offsets


def test_frame_axes_at_known_bearings():
    s = SlopeFrame(180.0, (0, 0))                       # falls to the south
    assert np.allclose(s.f, [0, -1]) and np.allclose(s.r, [-1, 0])   # right looking south = west
    e = SlopeFrame(90.0, (0, 0))                        # falls to the east
    assert np.allclose(e.f, [1, 0]) and np.allclose(e.r, [0, -1])    # right looking east = south
    u, v = e.uv((10.0, -4.0))
    assert (u, v) == pytest.approx((10.0, 4.0))


def test_crown_offset_sign_and_flanks_south_facing():
    frame = SlopeFrame(180.0, (0, 0))
    obs = box(-20, -60, 20, 0)                          # crown at y = 0 (top), toe at y = -60
    pred = box(-23, -70, 20, -5)                        # crown 5 m lower, 3 m wider to the west
    o = crown_flank_offsets(pred, obs, frame)
    assert o['crown_offsets_m'] == pytest.approx([5.0] * 5)
    assert o['crown_offset_median_m'] == pytest.approx(5.0)
    assert o['flank_right_m'] == pytest.approx(3.0)     # west is right looking downslope
    assert o['flank_left_m'] == pytest.approx(0.0)


def test_crown_offset_negative_when_predicted_crown_higher():
    frame = SlopeFrame(90.0, (0, 0))                    # falls east: crown is the west edge
    obs = box(0, -20, 60, 20)
    pred = box(-8, -20, 60, 20)
    assert crown_flank_offsets(pred, obs, frame)['crown_offset_median_m'] == pytest.approx(-8.0)


def test_boundary_segments_label_edges_by_outward_normal():
    frame = SlopeFrame(180.0, (0, 0))
    obs = box(-20, -60, 20, 0)
    pts, lab = boundary_segments(obs, frame)
    top = [l for p, l in zip(pts, lab) if p[1] == pytest.approx(0) and abs(p[0]) < 15]
    bot = [l for p, l in zip(pts, lab) if p[1] == pytest.approx(-60) and abs(p[0]) < 15]
    west = [l for p, l in zip(pts, lab) if p[0] == pytest.approx(-20) and -50 < p[1] < -10]
    east = [l for p, l in zip(pts, lab) if p[0] == pytest.approx(20) and -50 < p[1] < -10]
    assert set(top) == {'crown'} and set(bot) == {'stauchwall'}
    assert set(west) == {'flank_right'} and set(east) == {'flank_left'}
