"""Shot-location triggers: the no-Sk38 fallback and [scenarios] shot_locations_only.

Sisters cycle 2026-11-20 failed all three forecast days on KeyError
['sk38_min']: at first snow analyze writes a features CSV with only
hs/group (no profile reached a weak layer) and no Meloche features, so every
start-zone cluster reached the Sk38 ranking without an Sk38. Ron, 2026-11-20:
use the shot locations then (no candidate met the trigger criteria), and add
an option that triggers the model set at every shot location and nowhere
else, with no gate or filter of any kind (on for Sisters).

Same frozen Jan 18 fixture as test_trigger_fallback.py; only Zarr reads are
stubbed.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import jan18_fixture as fx   # puts src/avachain on sys.path

import analysis_pipeline as ap
from config import ProjectConfig
from test_trigger_fallback import POINTS, SHOTS
import model_params as mp


def _run(tmp_path, monkeypatch, *, features='full', meloche=True,
         shots_only=None, cfg_shots_only=False, n_triggers=5, points=POINTS,
         radius=None, forced=False):
    d = fx.load()
    an = tmp_path / "analysis"
    an.mkdir()
    np.save(an / "cluster_map.npy", d['cluster_map'])
    snap = "2026-01-18"
    feats = d['snap_features']
    if features == 'hs_only':          # first snow: what analyze wrote on 11-19
        feats = feats[['hs']].copy() if 'hs' in feats.columns \
            else pd.DataFrame({'hs': 0.3}, index=feats.index)
        feats['group'] = 'adjacent'
    elif features == 'sk38_nan':
        sk = 'min_sk38' if 'min_sk38' in feats.columns else 'sk38_min'
        feats = feats.copy()
        feats[sk] = np.nan
    feats.to_csv(an / f"all_start_zone_features_{snap}.csv")
    if meloche:
        d['meloche_df'].to_csv(an / f"meloche_features_all_{snap}.csv")
    cfg = SimpleNamespace(
        analysis_dir=an, resampled_dir=fx.DATA, start_zone_kml=fx.START_ZONE_KML,
        slope_name="Professor_PTC", depth_scales={50: 1.0},
        explosive_trigger_points=list(points), shot_locations_only=cfg_shots_only,
        force_explosive_triggers=forced,
        release_geojson=tmp_path / "none.geojson",
        release_geojsons_for_date=lambda _d: [])
    args = SimpleNamespace(
        snapshot_date=snap, max_slab_thickness=2.0, stauchwall_deg=29.0,
        restrict_to_release_area=False, n_triggers=n_triggers,
        use_observed_release=False, no_propagation=True, size_factors=[1.0],
        depth_pcts=[50], max_clusters=2000, mu=0.155, xi=1500.0, pro_dir=None,
        zarr_path=None, slab_tensile_gate=None, shot_locations_only=shots_only,
        shot_radius_m=radius)
    ds = xr.Dataset(coords={'location': np.array([], dtype=str),
                            'time': [np.datetime64(f"{snap}T12:00")]})
    monkeypatch.setattr(ap, "load_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(ap, "depth_from_snowpack",
                        lambda cm, *a, **k: np.full(cm.shape, 0.5, np.float32))
    monkeypatch.setattr(ap, "_build_scour_depth_map", lambda *a, **k: {})
    for name in ("USE_SLAB_TENSILE_GATE", "ARREST_MODE", "MASK_MODE"):
        monkeypatch.setattr(mp, name, getattr(mp, name))
    monkeypatch.setattr(mp, "USE_SLAB_TENSILE_GATE", False)
    out = tmp_path / "set"
    out.mkdir()
    shot = tmp_path / "shots"
    ap._build_scenarios(cfg, args, out, shot_dir=shot if forced else None)
    meta = json.loads((out / "metadata.json").read_text()) \
        if (out / "metadata.json").exists() else None
    if forced:
        return out, meta, shot
    return out, meta


# --- no Sk38 among the candidates -> shot-location fallback -----------------

def test_first_snow_features_without_sk38_trigger_at_the_shots(tmp_path, monkeypatch):
    # The 11-19 shape: hs/group only, no Meloche features. Was KeyError.
    out, meta = _run(tmp_path, monkeypatch, features='hs_only', meloche=False)
    assert meta["trigger_source"] == "explosive_fallback"
    assert meta["n_trigger_locations"] == len(SHOTS)


def test_all_nan_sk38_also_triggers_at_the_shots(tmp_path, monkeypatch):
    # Column present but empty: published an empty model set before.
    out, meta = _run(tmp_path, monkeypatch, features='sk38_nan', meloche=False)
    assert meta["trigger_source"] == "explosive_fallback"
    assert meta["n_trigger_locations"] == len(SHOTS)


def test_has_sk38():
    f = pd.DataFrame({'sk38_min': [np.nan, 0.8]}, index=[1, 2])
    assert ap._has_sk38(f, [1, 2], 'sk38_min')
    assert not ap._has_sk38(f, [1], 'sk38_min')
    assert not ap._has_sk38(f, [1, 2], 'min_sk38')


# --- shot_locations_only: every shot, always, nothing else ----------------

def _trig(out):
    trig = json.loads((out / "trigger_locations.geojson").read_text())
    return [f['properties']['cluster_id'] for f in trig['features']]


def test_shot_locations_only_triggers_at_every_shot(tmp_path, monkeypatch, capsys):
    # Jan 18 features: 1 of the 4 Professor shots passes the gates, and
    # n_triggers=1. All four are still triggers, equally weighted.
    out, meta = _run(tmp_path, monkeypatch, shots_only=True, n_triggers=1)
    log = capsys.readouterr().out
    assert meta["trigger_source"] == "shot_locations"
    assert meta["n_trigger_locations"] == len(SHOTS)
    assert _trig(out) == SHOTS
    for gate in ("Candidates after", "elevation >= P", "Pi1 propagation gate"):
        assert gate not in log
    w = pd.read_csv(out / "summary.csv").groupby("trigger_cluster").weight.sum()
    assert np.allclose(w, 1 / len(SHOTS))


def test_shot_locations_only_from_the_toml_default(tmp_path, monkeypatch):
    out, meta = _run(tmp_path, monkeypatch, cfg_shots_only=True)
    assert meta["trigger_source"] == "shot_locations"


def test_cli_off_overrides_the_toml(tmp_path, monkeypatch):
    out, meta = _run(tmp_path, monkeypatch, cfg_shots_only=True, shots_only=False,
                     n_triggers=1)
    assert meta["trigger_source"] == "model"


def test_shot_locations_only_without_points_uses_the_start_zone(capsys):
    cfg = SimpleNamespace(explosive_trigger_points=[], shot_locations_only=True)
    assert ap._shot_locations_only(cfg, SimpleNamespace()) is False
    assert "explosive_trigger_points is empty" in capsys.readouterr().out


def test_shot_locations_only_on_first_snow(tmp_path, monkeypatch):
    # The 11-19 shape: no Sk38, no Meloche features. Still every shot.
    out, meta = _run(tmp_path, monkeypatch, features='hs_only', meloche=False,
                     shots_only=True)
    assert meta["trigger_source"] == "shot_locations"
    assert meta["n_trigger_locations"] == len(SHOTS)


# --- shot footprint: best seed within shot_radius_m ----------------------

def _seeds(radius, features=True):
    d = fx.load()
    feats = d['snap_features'] if features else d['snap_features'][[]]
    mel = d['meloche_df'] if features else pd.DataFrame()
    cfg = SimpleNamespace(explosive_trigger_points=list(POINTS), shot_radius_m=radius)
    return d, ap._shot_trigger_clusters(cfg, SimpleNamespace(), d['cluster_map'],
                                        d['transform'], d['start_zone_mask'],
                                        feats, mel)


def test_radius_zero_is_the_cluster_under_the_point():
    _, seeds = _seeds(0.0)
    assert seeds == SHOTS


def test_seed_has_the_lowest_rc_within_the_radius():
    d = fx.load()
    cm, tr, sz = d['cluster_map'], d['transform'], d['start_zone_mask']
    rc = d['snap_features']['rc_wl']
    moved = 0
    for x, y, shot in ap._points_on_map(POINTS, cm, tr):
        seed, _n = ap._footprint_seed(x, y, shot, 10.0, cm, tr, sz,
                                      d['snap_features'], d['meloche_df'])
        # brute force over the whole map
        foot = {shot}
        for cid in np.unique(cm[cm > 0]):
            rr, cc = np.nonzero(cm == cid)
            if sz[rr, cc].any() and np.hypot(tr.c + (cc.mean() + 0.5) * tr.a - x,
                                             tr.f + (rr.mean() + 0.5) * tr.e - y) <= 10.0:
                foot.add(int(cid))
        assert seed in foot
        best = min(rc[c] for c in foot if c in rc.index and np.isfinite(rc[c]))
        assert rc[seed] == best
        moved += seed != shot
    assert moved                                   # Jan 18: the footprint matters


def test_no_features_keeps_the_shot_cluster():
    _, seeds = _seeds(10.0, features=False)
    assert seeds == SHOTS


def test_forced_shots_and_model_set_use_the_same_seeds(tmp_path, monkeypatch):
    _, seeds = _seeds(10.0)
    out, meta, shot = _run(tmp_path, monkeypatch, shots_only=True, radius=10.0,
                           forced=True)
    assert _trig(out) == seeds
    assert _trig(shot) == seeds


@pytest.mark.parametrize("slope,on", [("Sisters", True), ("Professor_PTC", False),
                                      ("Widow", False), ("little_prof", False)])
def test_toml_key(slope, on):
    toml = fx.REPO / f"slopes/{slope}/slope_config.toml"
    cfg = ProjectConfig.from_toml(toml)
    assert cfg.shot_locations_only is on
    assert cfg.shot_radius_m == (10.0 if on else 0.0)
