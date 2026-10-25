"""Tests for the mechanics-free baseline arrest modes (baselines.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'avachain'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest

import energy_path as ep
import release_geometry as rg
from baselines import propagate_baseline
from release_geometry import propagate_release

from test_energy_propagation import BLOCK, T, scene


def _run(mode, s, stauchwall_deg=rg.STAUCHWALL_DEG, mask=None, state=None):
    return propagate_baseline(
        mode, s['trigger'], s['cluster_map'], s['dem'], s['slope_grid'], T,
        start_zone_mask=mask, stauchwall_deg=stauchwall_deg,
        snap_features=s['feats'], pixel_index=s['pixel_index'],
        max_clusters=100000, state_out=state)


def test_slope_only_stops_at_missing_weak_layer_only():
    s = scene(30, 60, (16, 31),
              lambda y, x: {'wl_shear_strength': np.nan} if 6.0 <= x < 12.0 else {})
    state = {}
    _, reached = _run('slope_only', s, state=state)
    xs = s['feats'].loc[list(reached), 'x']
    assert xs.max() < 6.0 and xs.min() == s['feats']['x'].min()
    assert set(state['rejected'].values()) == {'no_weak_layer'}


def test_slope_only_ignores_strength():
    """No mechanics: a tau_p step that arrests 'energy' does not arrest slope_only."""
    s = scene(30, 30, (16, 16), lambda y, x: {'wl_shear_strength': 1e5})
    _, reached = _run('slope_only', s)
    assert reached == set(s['pixel_index'])


def test_slope_only_stops_at_stauchwall_angle():
    s = scene(60, 30, (16, 16), lambda y, x: {},
              slope_deg=lambda y: 35.0 if y < 30 else 20.0)
    _, reached = _run('slope_only', s)
    rows = [s['pixel_index'][c][0].mean() for c in reached]
    assert max(rows) < 30 + BLOCK


def test_stability_connected_follows_min_sk38():
    s = scene(30, 30, (16, 16), lambda y, x: {'min_sk38': 0.8 if y >= -3 else 1.2})
    state = {}
    _, reached = _run('stability_connected', s, state=state)
    assert s['feats'].loc[list(reached), 'y'].min() >= -3
    assert set(state['rejected'].values()) == {'stability_stable'}


def test_mask_gate():
    s = scene(30, 60, (16, 31), lambda y, x: {})
    mask = np.zeros(s['cluster_map'].shape, bool)
    mask[:, :45] = True
    state = {}
    _, reached = _run('slope_only', s, mask=mask, state=state)
    assert max(s['pixel_index'][c][1].mean() for c in reached) < 45
    assert set(state['rejected'].values()) == {'outside_start_zone'}


def test_routed_through_propagate_release(monkeypatch):
    monkeypatch.setattr(rg, 'ARREST_MODE', 'slope_only')
    s = scene(30, 30, (16, 16), lambda y, x: {})
    poly, reached = propagate_release(
        trigger_cluster_id=s['trigger'], meloche_df=pd.DataFrame(),
        cluster_map=s['cluster_map'], dem=s['dem'], slope_grid=s['slope_grid'],
        transform=T, snap_features=s['feats'], pixel_index=s['pixel_index'],
        max_clusters=100000)
    assert reached == set(s['pixel_index'])
    assert poly.area == pytest.approx(30 * 30)


def test_unknown_baseline_raises():
    s = scene(30, 30, (16, 16), lambda y, x: {})
    with pytest.raises(ValueError):
        _run('nonsense', s)
