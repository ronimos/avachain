"""_scour_depth_for_cluster works in metres, the units of HS and element
height in the Zarr store. It treated them as cm until 2026-10-27: with a slab
every scour depth came out 0, without one 100x too small, and scour_depth.tif
(com1DFA's entrainable depth) carried that.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from analysis_pipeline import _scour_depth_for_cluster  # noqa: E402


def _profile():
    # bottom to top, element top heights (m) and hand hardness (xsnow, negative)
    height = np.array([0.20, 0.50, 0.80, 1.10, 1.40])
    hh = np.array([-5.0, -4.0, -2.0, -2.0, -1.0])     # hard below 0.50 m
    return xr.Dataset({'HS': ((), 1.40),
                       'height': (('layer',), height),
                       'hand_hardness': (('layer',), hh)})


def test_slab_scour_runs_from_failure_plane_to_first_hard_layer():
    # failure plane at 1.40 - 0.70 = 0.70 m; first layer at or below it is
    # 0.50 m (hh -4, harder than 1F)
    d, d_mid, v = _scour_depth_for_cluster(_profile(), 0.70)
    assert d == pytest.approx(0.20)
    assert d_mid == pytest.approx(1.40 - 0.35 - 0.50)
    assert v == -4.0


def test_no_slab_scans_from_surface():
    d, d_mid, v = _scour_depth_for_cluster(_profile(), None)
    assert d == pytest.approx(1.40 - 0.50)
    assert d_mid == d and v == -4.0
