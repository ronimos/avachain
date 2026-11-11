"""A bare-ground day publishes an empty "no_snow" set instead of failing.

First live night, 2026-11-08: the season started from bare ground on 11-07
and the slope still had no snow (CAABT 0-1 cm, modelled HS 0.000 everywhere).
analyze found no snow profile in the start zone, so it wrote no features CSV,
and `scenarios` raised "Run 'analyze' step first" on T_0..T_2. The daily run
ended FAILED with no report. The shot-location fallback could not help: it
sits after the CSV read.

Now a missing CSV with a start-zone max HS below MIN_RELEASE_DEPTH_M publishes
an empty set with trigger_source "no_snow" (and an empty forced-shot set).
With snow present a missing CSV is still a fault and still raises.

Jan 18 has snow everywhere, so it cannot see this: it is pinned here on the
frozen Jan 18 cluster map, with only the Zarr read stubbed.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import jan18_fixture as fx   # puts src/avachain on sys.path

import analysis_pipeline as ap
import release_geometry as _rg
import model_params as mp

SNAP = "2026-11-08"


def _run(tmp_path, monkeypatch, hs_m, with_hs=True, shots=False):
    d = fx.load()
    an = tmp_path / "analysis"
    an.mkdir()
    np.save(an / "cluster_map.npy", d['cluster_map'])
    # no features CSV: what analyze leaves on a day with no snow profile

    sz = sorted(int(c) for c in np.unique(d['cluster_map'][d['start_zone_mask']]) if c > 0)
    locs = np.array([f"cluster_{c:04d}" for c in sz])
    times = np.array([f"{SNAP}T12:00", f"{SNAP}T18:00"], dtype="datetime64[ns]")
    hs = np.zeros((len(locs), len(times)))
    if not np.isscalar(hs_m):
        cid, val = hs_m
        hs[sz.index(cid), :] = val
    data = {'HS': (('location', 'time'), hs)} if with_hs else {}
    ds = xr.Dataset(data, coords={'location': locs, 'time': times})
    monkeypatch.setattr(ap, "load_dataset", lambda *a, **k: ds)
    for name in ("USE_SLAB_TENSILE_GATE", "ARREST_MODE", "MASK_MODE"):
        monkeypatch.setattr(mp, name, getattr(mp, name))

    cfg = SimpleNamespace(
        analysis_dir=an, resampled_dir=fx.DATA, start_zone_kml=fx.START_ZONE_KML,
        slope_name="Professor_PTC", explosive_trigger_points=[])
    args = SimpleNamespace(snapshot_date=SNAP, snapshot_ts=f"{SNAP}T18:00",
                           pro_dir=None, zarr_path=None, slab_tensile_gate=None)
    out = tmp_path / "set"
    out.mkdir()
    shot = tmp_path / "shots" if shots else None
    ap._build_scenarios(cfg, args, out, shot_dir=shot)
    meta = json.loads((out / "metadata.json").read_text())
    shot_meta = json.loads((shot / "metadata.json").read_text()) if shots else None
    return meta, shot_meta


def test_bare_ground_publishes_a_no_snow_set(tmp_path, monkeypatch):
    meta, _ = _run(tmp_path, monkeypatch, 0.0)
    assert meta["trigger_source"] == "no_snow"
    assert meta["n_scenarios"] == 0 and meta["max_hs_m"] == 0.0
    assert meta["min_release_depth_m"] == ap.MIN_RELEASE_DEPTH_M


def test_bare_ground_also_empties_the_forced_shot_set(tmp_path, monkeypatch):
    _, shot = _run(tmp_path, monkeypatch, 0.0, shots=True)
    assert shot["trigger_source"] == "no_snow"
    assert shot["n_scenarios"] == 0 and shot["p_no_release"] == 1.0
    assert shot["shot_outcomes"] == {}


def test_a_dusting_under_the_floor_is_still_no_snow(tmp_path, monkeypatch):
    cid = sorted(int(c) for c in np.unique(fx.load()['cluster_map'][fx.load()['start_zone_mask']]) if c > 0)[0]
    meta, _ = _run(tmp_path, monkeypatch, (cid, 0.08))
    assert meta["trigger_source"] == "no_snow" and meta["max_hs_m"] == 0.08


def test_snow_with_no_features_still_fails(tmp_path, monkeypatch):
    # One start-zone cluster with 0.5 m: a missing CSV is a real fault.
    cid = sorted(int(c) for c in np.unique(fx.load()['cluster_map'][fx.load()['start_zone_mask']]) if c > 0)[0]
    with pytest.raises(FileNotFoundError, match="not bare ground"):
        _run(tmp_path, monkeypatch, (cid, 0.5))


def test_a_store_without_hs_still_fails(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="missing features CSV"):
        _run(tmp_path, monkeypatch, 0.0, with_hs=False)
