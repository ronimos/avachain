"""
Layer-of-concern weak-layer selection (model_params.WL_SELECTION).

'basal' (default) must stay exactly split_wl_slab; 'threshold_sum' picks the
interface with the most threshold-sum lemons (Schweizer & Jamieson 2007).
The Jan 18 replay cannot pin these: its observed bed is noisy per cluster, so
the rule is pinned on synthetic profiles here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))
import model_params as mp
from snowpack_analysis import (split_wl_slab, split_wl_slab_tsa, select_wl_slab,
                               tsa_lemons, profile_features)

# Element tops, bottom -> top (z negative down). 12 elements of 0.1 m.
Z = np.array([-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1])
# Basal DH (2 elements), hard RG, a soft coarse FC at 0.5 m under a hard RG
# layer, soft RG near the surface.
GT = np.array([550, 550, 330, 330, 330, 330, 330, 440, 330, 320, 320, 320], float)
HH = -np.array([2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 3.5, 2.0, 2.0, 2.0])  # store sign
GS = np.array([2.0, 2.0, .5, .5, .5, .5, .5, 1.5, .5, .5, .5, .5]) / 1000.0  # m


def test_lemon_count_of_the_designed_fc():
    # FC at 0.5 m under the 3.5 hardness RG: all six lemons
    assert tsa_lemons(440, -1.0, 1.5, -3.5, 0.5, 0.5) == 6
    # basal DH under hardness-3 RG at 1.1 m: gs diff, persistent, gs only
    assert tsa_lemons(550, -2.0, 2.0, -3.0, 0.5, 1.1) == 3


def test_tsa_picks_the_buried_fc_and_basal_picks_the_ground():
    s, w, iz = split_wl_slab_tsa(GT, Z, HH, GS)
    assert iz == pytest.approx(-0.5)
    assert s[Z > -0.5].all() and not s[Z <= -0.5].any()
    assert w[7] and w.sum() == 1          # FC is followed by RG: run of one
    sb, wb, izb = split_wl_slab(GT, Z)
    assert izb == pytest.approx(-1.1)


def test_wl_run_extends_through_contiguous_persistent_elements():
    gt = GT.copy(); gt[6] = 990; gt[5] = 450      # FC over FCxr over FC
    s, w, iz = split_wl_slab_tsa(gt, Z, HH, GS)
    assert iz == pytest.approx(-0.5)
    assert w[7] and w[6] and w[5] and not w[4]


def test_min_depth_excludes_shallow_candidates():
    s, w, iz = split_wl_slab_tsa(GT, Z, HH, GS, min_depth_m=0.6)
    assert -iz >= 0.6


def test_upper_element_can_be_the_weak_layer():
    # soft coarse FC at 0.4 m lying ON a hard crust (code 880), soft RG above:
    # the FC's lemons come from the crust below it
    gt = np.array([330, 330, 330, 330, 330, 330, 330, 880, 440, 320, 320, 320], float)
    hh = -np.array([3, 3, 3, 3, 3, 3, 3, 5.0, 1.0, 1.5, 1.5, 1.5])
    gs = np.array([.5, .5, .5, .5, .5, .5, .5, .5, 1.6, .5, .5, .5]) / 1000.0
    s, w, iz = split_wl_slab_tsa(gt, Z, hh, gs)
    assert iz == pytest.approx(-0.4)
    assert w[8] and not w[7]


def test_select_basal_is_split_wl_slab_on_random_profiles():
    rng = np.random.default_rng(0)
    codes = np.array([320, 330, 341, 440, 450, 550, 990, 880])
    for _ in range(200):
        n = int(rng.integers(3, 30))
        z = -np.sort(rng.uniform(0.01, 2.0, n))[::-1]
        gt = rng.choice(codes, n).astype(float)
        a = split_wl_slab(gt, z)
        b = select_wl_slab(gt, z, method='basal')
        if a[0] is None:
            assert b[0] is None
        else:
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_select_rejects_unknown_method_and_missing_inputs():
    with pytest.raises(ValueError):
        select_wl_slab(GT, Z, method='nope')
    with pytest.raises(ValueError):
        select_wl_slab(GT, Z, method='threshold_sum')


def test_default_is_basal():
    assert mp.WL_SELECTION == 'basal'
    assert mp.WL_SELECTION in mp.WL_SELECTIONS


def _ds(with_structure=True):
    v = {'z': ('e', Z.copy()), 'grain_type': ('e', GT.copy()), 'HS': ((), 1.2)}
    if with_structure:
        v['hand_hardness'] = ('e', HH.copy())
        v['grain_size'] = ('e', GS.copy())
    return xr.Dataset(v)


def test_profile_features_follows_the_switch(monkeypatch):
    basal = profile_features(_ds(), 30.0)
    assert basal['slab_thickness'] == pytest.approx(1.1)
    monkeypatch.setattr(mp, 'WL_SELECTION', 'threshold_sum')
    loc = profile_features(_ds(), 30.0)
    assert loc['slab_thickness'] == pytest.approx(0.5)


def test_basal_path_needs_no_structure_variables():
    # the operational path must not start reading new store variables
    feat = profile_features(_ds(with_structure=False), 30.0)
    assert feat['slab_thickness'] == pytest.approx(1.1)
