"""Forced explosive triggers: a scenario set at every shot location, every run.

Ron, 2026-11-06: shots are guaranteed to happen and will initiate; the
question is only whether the crack propagates from them. So the shot set
takes no initiation gate (tau_g, slope, Sk38, slab thickness, elevation,
Pi1), keeps everything that decides propagation (slab tensile gate, BFS
arrest, clips, the 0.10 m depth floor), and has no rectangle fallback: a
shot that does not propagate releases nothing, and the set says why. It is
published as its own set at <date>/shots[/T_n], each shot equally weighted,
beside an unchanged model set.

Jan 18 has model candidates, so it would never exercise the shot path on its
own; these tests force it on the frozen Jan 18 map and features.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import jan18_fixture as fx   # puts src/avachain on sys.path

import analysis_pipeline as ap
import handoff
import release_geometry as _rg
import runout_step
from config import ProjectConfig
from test_trigger_fallback import POINTS, SHOTS


def _run(tmp_path, monkeypatch, *, depth_m=0.5, thin=False, no_propagation=False,
         tensile=False, shots=True, size_factors=(1.0,), max_clusters=2000,
         points=POINTS):
    d = fx.load()
    an = tmp_path / "analysis"
    an.mkdir()
    np.save(an / "cluster_map.npy", d['cluster_map'])
    snap = "2026-01-18"
    d['snap_features'].to_csv(an / f"all_start_zone_features_{snap}.csv")
    mel = d['meloche_df']
    if thin:
        mel = mel[['tau_g']].copy()
        mel['tau_g'] = 10.0                   # no model candidate, no arrest indices
    mel.to_csv(an / f"meloche_features_all_{snap}.csv")
    cfg = SimpleNamespace(
        analysis_dir=an, resampled_dir=fx.DATA, start_zone_kml=fx.START_ZONE_KML,
        slope_name="Professor_PTC", depth_scales={50: 1.0},
        explosive_trigger_points=list(points), force_explosive_triggers=shots,
        release_geojson=tmp_path / "none.geojson",
        release_geojsons_for_date=lambda _d: [])
    args = SimpleNamespace(
        snapshot_date=snap, max_slab_thickness=2.0, stauchwall_deg=29.0,
        restrict_to_release_area=False, n_triggers=1, use_observed_release=False,
        no_propagation=no_propagation, size_factors=list(size_factors),
        depth_pcts=[50], max_clusters=max_clusters, mu=0.155, xi=1500.0, pro_dir=None,
        zarr_path=None, slab_tensile_gate=None)
    ds = xr.Dataset(coords={'location': np.array([], dtype=str),
                            'time': [np.datetime64(f"{snap}T12:00")]})
    monkeypatch.setattr(ap, "load_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(ap, "depth_from_snowpack",
                        lambda cm, *a, **k: np.full(cm.shape, depth_m, np.float32))
    monkeypatch.setattr(ap, "_build_scour_depth_map", lambda *a, **k: {})
    for name in ("USE_SLAB_TENSILE_GATE", "ARREST_MODE", "MASK_MODE"):
        monkeypatch.setattr(_rg, name, getattr(_rg, name))
    monkeypatch.setattr(_rg, "USE_SLAB_TENSILE_GATE", tensile)
    out, shot = tmp_path / "set", tmp_path / "shots"
    out.mkdir()
    ap._build_scenarios(cfg, args, out, shot_dir=shot if shots else None)
    meta = lambda p: json.loads((p / "metadata.json").read_text()) \
        if (p / "metadata.json").exists() else None
    return out, meta(out), shot, meta(shot)


def test_shots_propagate_on_jan18(tmp_path, monkeypatch):
    # One shot (each BFS call costs ~4 s of setup); 3500 propagates on the
    # frozen Jan 18 features with the tensile gate off. A real BFS polygon,
    # since the shot set has no rectangle to fall back on.
    out, meta, shot, smeta = _run(tmp_path, monkeypatch, points=POINTS[:1])
    assert smeta["trigger_source"] == "explosive_forced"
    assert smeta["n_trigger_locations"] == 1
    assert smeta["shot_outcomes"] == {"3500": {
        "n_scenarios": 1, "n_below_depth_floor": 0, "n_no_release": 0,
        "no_release_reason": None}}
    s = pd.read_csv(shot / "summary.csv")
    assert list(s.trigger_cluster) == [3500] and np.allclose(s.weight, 1.0)
    # The model set is the model's, not the shots'.
    assert meta["trigger_source"] == "model" and "shot_outcomes" not in meta


def test_model_set_unchanged_by_forcing(tmp_path, monkeypatch):
    # Rectangles (no_propagation) keep it fast; the model path is the same.
    runs = []
    for force in (False, True):
        d = tmp_path / str(force)
        d.mkdir()
        runs.append(_run(d, monkeypatch, shots=force, no_propagation=True))
    sa, sb = (pd.read_csv(r[0] / "summary.csv") for r in runs)
    pd.testing.assert_frame_equal(sa, sb)


def test_no_rectangle_for_a_shot_that_does_not_propagate(tmp_path, monkeypatch):
    # With propagation off the legacy path builds an oriented rectangle for
    # every trigger. A shot must not: nothing showed it propagates.
    out, meta, shot, smeta = _run(tmp_path, monkeypatch, no_propagation=True)
    assert meta["n_scenarios"] > 0                       # model keeps its rectangle
    assert smeta["n_scenarios"] == 0
    assert all(o["n_no_release"] == 1 and o["no_release_reason"] == "no_propagation"
               for o in smeta["shot_outcomes"].values())
    assert not (shot / "summary.csv").exists()


def test_thin_day_shots_without_arrest_indices_release_nothing(tmp_path, monkeypatch):
    # No cluster reached TAU_G_FEATURE_FLOOR: the model set falls back to the
    # shots as before (with rectangles), the forced set says no propagation.
    out, meta, shot, smeta = _run(tmp_path, monkeypatch, thin=True)
    assert meta["trigger_source"] == "explosive_fallback"
    assert smeta["n_scenarios"] == 0
    assert {o["no_release_reason"] for o in smeta["shot_outcomes"].values()} \
        <= {"no_propagation", "no_arrest_indices"}


def test_slab_tensile_failure_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(_rg, "cluster_tensile_support", lambda *a, **k: 0.5)
    out, meta, shot, smeta = _run(tmp_path, monkeypatch, tensile=True)
    assert smeta["n_scenarios"] == 0
    assert all(o["no_release_reason"] == "slab_tensile"
               for o in smeta["shot_outcomes"].values())


class TestPublish:
    def _cfg_args(self, tmp_path, horizon=None, force=True):
        cfg = SimpleNamespace(scenarios_dir=tmp_path / "release", toml_path=None,
                              slope_name="Professor_PTC",
                              explosive_trigger_points=list(POINTS),
                              force_explosive_triggers=force)
        args = SimpleNamespace(snapshot_date="2026-01-18", forecast_horizon=horizon)
        return cfg, args

    @staticmethod
    def _fake(cfg, args, out_dir, shot_dir=None):
        (out_dir / "metadata.json").write_text(json.dumps(
            {"n_scenarios": 3, "trigger_source": "model"}))
        if shot_dir is not None:
            shot_dir.mkdir(parents=True, exist_ok=True)
            (shot_dir / "metadata.json").write_text(json.dumps(
                {"n_scenarios": 0, "trigger_source": "explosive_forced"}))

    def test_single_day_layout_and_republish(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ap, "_build_scenarios", self._fake)
        cfg, args = self._cfg_args(tmp_path)
        for _ in range(2):                                  # republish keeps both
            final = ap.step_scenarios(cfg, args)
            shots = final / "shots"
            assert handoff.is_ready(final) and handoff.is_ready(shots)
            assert handoff.verify(final) == [] and handoff.verify(shots) == []
        m = json.loads((shots / "manifest.json").read_text())
        assert m["status"] == "no_scenarios" and m["trigger_source"] == "explosive_forced"
        # The runout never takes the shot set for a horizon.
        assert runout_step.release_sets(cfg, "2026-01-18") == {"T_0": final}

    def test_forecast_layout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ap, "_build_scenarios", self._fake)
        cfg, args = self._cfg_args(tmp_path, horizon="T_1")
        args.cycle_date = "2026-01-18"
        ap.step_scenarios(cfg, args)
        base = cfg.scenarios_dir / "2026-01-18"
        assert handoff.is_ready(base / "T_1") and handoff.is_ready(base / "shots" / "T_1")
        assert list(runout_step.release_sets(cfg, "2026-01-18")) == ["T_1"]

    def test_off_by_default_and_needs_points(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(ap, "_build_scenarios",
                            lambda c, a, o, **k: calls.append(k) or self._fake(c, a, o))
        for force, pts in ((False, POINTS), (True, [])):
            cfg, args = self._cfg_args(tmp_path, force=force)
            cfg.explosive_trigger_points = list(pts)
            final = ap.step_scenarios(cfg, args)
            assert not (final / "shots").exists()
        args.force_explosive_triggers = True                 # CLI overrides TOML
        cfg.explosive_trigger_points = list(POINTS)
        cfg.force_explosive_triggers = False
        assert (ap.step_scenarios(cfg, args) / "shots").exists()
        assert calls[:2] == [{}, {}]


def test_toml_key(tmp_path):
    # On for the operational slope (Ron, 2026-11-06), off for the dev slope.
    assert ProjectConfig.from_toml(
        fx.REPO / "slopes/Professor_PTC/slope_config.toml").force_explosive_triggers is True
    toml = fx.REPO / "slopes/little_prof/slope_config.toml"
    assert ProjectConfig.from_toml(toml).force_explosive_triggers is False
    src = toml.read_text()
    assert src.count("force_explosive_triggers = false") == 1
    p = tmp_path / "slope_config.toml"
    p.write_text(src.replace("force_explosive_triggers = false",
                             "force_explosive_triggers = true"))
    assert ProjectConfig.from_toml(p).force_explosive_triggers is True



def test_each_shot_holds_one_nth_and_the_rest_is_no_release(tmp_path, monkeypatch):
    # Ron, 2026-11-06: 1/N per shot, the shots that do not release count as
    # P(no release), nothing renormalised. The polygon builder is stubbed:
    # 4137 does not propagate, the other three release the same box.
    from shapely.geometry import box
    d = fx.load()
    rows, cols = np.nonzero(d['cluster_map'] == 3500)
    tr = d['transform']
    x0, y0 = tr * (cols.min(), rows.max() + 1)
    x1, y1 = tr * (cols.max() + 1, rows.min())
    poly = box(x0, y0, x1, y1)
    monkeypatch.setattr(ap, "make_release_polygon_2d",
                        lambda trigger_cluster_id, **k:
                        None if trigger_cluster_id == 4137 else poly)
    out, meta, shot, smeta = _run(tmp_path, monkeypatch,
                                  size_factors=(0.85, 1.0, 1.15))
    assert smeta["p_no_release"] == pytest.approx(0.25)
    s = pd.read_csv(shot / "summary.csv")
    per_shot = s.groupby("trigger_cluster").weight.sum()
    assert sorted(per_shot.index) == [2480, 2568, 3500]
    assert np.allclose(per_shot, 0.25)
    w = json.loads((shot / "scenario_weights.json").read_text())
    assert sum(w.values()) == pytest.approx(0.75)                 # not renormalised
    for f in (shot / "scenarios").glob("*/params.json"):
        prm = json.loads(f.read_text())
        assert prm["ensemble_weight_total"] == 1.0
        assert prm["scenario_probability"] == pytest.approx(w[f.parent.name], abs=1e-6)
    # the model set is renormalised as before, and carries no total
    m = json.loads((out / "scenario_weights.json").read_text())
    assert sum(m.values()) == pytest.approx(1.0)
    for f in (out / "scenarios").glob("*/params.json"):
        assert "ensemble_weight_total" not in json.loads(f.read_text())


def test_no_shot_releases_is_all_no_release(tmp_path, monkeypatch):
    out, meta, shot, smeta = _run(tmp_path, monkeypatch, no_propagation=True)
    assert smeta["n_scenarios"] == 0 and smeta["p_no_release"] == 1.0
