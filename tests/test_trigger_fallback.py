"""No-candidate days trigger at the slope's explosive-control shot locations.

Every thin early-season day has no cluster that passes the trigger gates, and
`scenarios` used to raise ValueError there, so the survey was marked FAILED
with no handoff (season-start rehearsal, 2025-11-27). Ron, 2026-11-04: use
the shot locations instead, drop a shot-location release holding < 0.10 m of
snow on average, and take release density from each scenario's own polygon.

Jan 18 cannot see any of this (it has candidates), so it is pinned here: the
frozen Jan 18 map and features, forced into a thin day (no candidates, and a
meloche frame without the arrest-index columns, as on 11-27 when no cluster
reached TAU_G_FEATURE_FLOOR). Only the Zarr reads are stubbed.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import jan18_fixture as fx   # puts src/avachain on sys.path

import analysis_pipeline as ap
import release_geometry as _rg
from config import ProjectConfig

# Little Professor shot locations (Ron), [lat, lon] WGS84: the centres of
# 2025-26 clusters 3500, 2568, 2480 and 4137, as in the slope TOML.
POINTS = [(39.6488366, -105.8788572), (39.6487075, -105.8782511),
          (39.6484944, -105.8791514), (39.6481336, -105.8787209)]
SHOTS = [3500, 2568, 2480, 4137]


def _thin_day(tmp_path, monkeypatch, depth_m, points=POINTS, model=False,
             n_triggers=5, no_propagation=False):
    """Run _build_scenarios with a uniform slab depth.

    By default a thin day (no candidate passes the gates). model=True keeps
    the real Jan 18 Meloche features, so the model's own triggers are used.
    """
    d = fx.load()
    an = tmp_path / "analysis"
    an.mkdir()
    np.save(an / "cluster_map.npy", d['cluster_map'])
    snap = "2026-01-18"
    d['snap_features'].to_csv(an / f"all_start_zone_features_{snap}.csv")
    if model:
        mel = d['meloche_df']
    else:
        mel = d['meloche_df'][['tau_g']].copy()
        mel['tau_g'] = 10.0                   # below MIN_TAU_G: no candidates
    mel.to_csv(an / f"meloche_features_all_{snap}.csv")

    cfg = SimpleNamespace(
        analysis_dir=an, resampled_dir=fx.DATA, start_zone_kml=fx.START_ZONE_KML,
        slope_name="Professor_PTC",
        depth_scales={50: 1.0}, explosive_trigger_points=list(points),
        release_geojson=tmp_path / "none.geojson",
        release_geojsons_for_date=lambda _d: [])
    args = SimpleNamespace(
        snapshot_date=snap, max_slab_thickness=2.0, stauchwall_deg=29.0,
        restrict_to_release_area=False, n_triggers=n_triggers, use_observed_release=False,
        no_propagation=no_propagation, size_factors=[1.0], depth_pcts=[50],
        max_clusters=2000, mu=0.155, xi=1500.0, pro_dir=None, zarr_path=None,
        slab_tensile_gate=None)

    ds = xr.Dataset(coords={'location': np.array([], dtype=str),
                            'time': [np.datetime64(f"{snap}T12:00")]})
    monkeypatch.setattr(ap, "load_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(ap, "depth_from_snowpack",
                        lambda cm, *a, **k: np.full(cm.shape, depth_m, np.float32))
    monkeypatch.setattr(ap, "_build_scour_depth_map", lambda *a, **k: {})
    # _build_scenarios writes these module flags; keep them out of other tests.
    for name in ("USE_SLAB_TENSILE_GATE", "ARREST_MODE", "MASK_MODE"):
        monkeypatch.setattr(_rg, name, getattr(_rg, name))
    monkeypatch.setattr(_rg, "USE_SLAB_TENSILE_GATE", False)

    out = tmp_path / "set"
    out.mkdir()
    ap._build_scenarios(cfg, args, out)
    meta = json.loads((out / "metadata.json").read_text()) \
        if (out / "metadata.json").exists() else None
    return out, meta


def test_no_candidates_triggers_at_the_shot_locations(tmp_path, monkeypatch):
    out, meta = _thin_day(tmp_path, monkeypatch, depth_m=0.5)
    assert meta["trigger_source"] == "explosive_fallback"
    assert meta["n_trigger_locations"] == 4 and meta["n_scenarios"] == 4
    summary = pd.read_csv(out / "summary.csv")
    assert sorted(summary.trigger_cluster) == sorted(SHOTS)
    # Equal weights: shot locations carry no instability ranking.
    assert np.allclose(summary.weight, 0.25)
    trig = json.loads((out / "trigger_locations.geojson").read_text())
    assert [f['properties']['cluster_id'] for f in trig['features']] == SHOTS


def test_shallow_shot_releases_are_dropped_and_recorded(tmp_path, monkeypatch):
    out, meta = _thin_day(tmp_path, monkeypatch, depth_m=0.05)
    assert meta == {"snapshot_date": "2026-01-18", "n_scenarios": 0,
                    "n_trigger_locations": 4,
                    "trigger_source": "explosive_fallback",
                    "n_below_depth_floor": 4}
    assert not (out / "summary.csv").exists()


def test_the_depth_floor_applies_to_model_triggers(tmp_path, monkeypatch):
    # Rehearsal 2025-11-30: both model triggers fell back to rectangles that
    # averaged 0.01-0.03 m of snow. Ron, 11-04: the floor applies to them too.
    # The floor is checked on the finished polygon, however it was built, so
    # the rectangle (no_propagation) keeps the test fast.
    out, meta = _thin_day(tmp_path, monkeypatch, depth_m=0.05, model=True,
                          n_triggers=1, no_propagation=True)
    assert meta["trigger_source"] == "model"
    assert meta["n_scenarios"] == 0 and meta["n_below_depth_floor"] > 0
    assert not (out / "summary.csv").exists()


def test_model_triggers_above_the_floor_are_kept(tmp_path, monkeypatch):
    out, meta = _thin_day(tmp_path, monkeypatch, depth_m=0.5, model=True,
                          n_triggers=1, no_propagation=True)
    assert meta["trigger_source"] == "model"
    assert meta["n_scenarios"] > 0 and meta["n_below_depth_floor"] == 0


def test_no_candidates_and_no_shot_locations_is_an_empty_set(tmp_path, monkeypatch):
    out, meta = _thin_day(tmp_path, monkeypatch, depth_m=0.5, points=[])
    assert meta is None                         # step_scenarios -> no_scenarios
    assert not (out / "scenarios").exists() or not any((out / "scenarios").iterdir())


def test_shot_location_not_on_the_map_is_skipped():
    d = fx.load()
    t = ap._explosive_triggers([4137, 999999, 3500, 4137], d['cluster_map'],
                               d['snap_features'], 'min_sk38')
    assert list(t.index) == [4137, 3500]        # configured order, no duplicates
    assert 'min_sk38' in t.columns


def test_featureless_shot_location_gets_a_nan_row():
    d = fx.load()
    feats = d['snap_features'].drop(index=[3500], errors='ignore')
    t = ap._explosive_triggers([3500], d['cluster_map'], feats, 'min_sk38')
    assert list(t.index) == [3500] and t.isna().all(axis=None)


class TestPolygonDensity:
    """Density from the release polygon's own clusters, never NaN."""

    def _poly(self, cid):
        from shapely.geometry import box
        d = fx.load()
        rows, cols = np.nonzero(d['cluster_map'] == cid)
        tr = d['transform']
        x0, y0 = tr * (cols.min(), rows.max() + 1)
        x1, y1 = tr * (cols.max() + 1, rows.min())
        return box(x0, y0, x1, y1), d

    def test_median_of_the_clusters_inside(self):
        poly, d = self._poly(1772)
        import rasterio.features
        mask = rasterio.features.geometry_mask(
            [poly.__geo_interface__], out_shape=d['cluster_map'].shape,
            transform=d['transform'], invert=True)
        cids = [c for c in np.unique(d['cluster_map'][mask]) if c > 0]
        rho = d['snap_features']['slab_density'].reindex(cids).dropna()
        assert len(rho) > 1
        mean, std = ap._polygon_density(poly, d['cluster_map'],
                                        d['snap_features'], d['transform'])
        assert mean == pytest.approx(rho.median()) and std == pytest.approx(rho.std())

    def test_no_slab_density_falls_back(self):
        poly, d = self._poly(1772)
        feats = d['snap_features'].assign(slab_density=np.nan)
        assert ap._polygon_density(poly, d['cluster_map'], feats, d['transform']) \
            == (ap.DEFAULT_SLAB_DENSITY, ap.DEFAULT_SLAB_DENSITY_STD)

    def test_does_not_depend_on_an_observed_release(self, tmp_path, monkeypatch):
        # The old density was the observed release group's median: NaN on
        # every day without an observed avalanche. A thin-day set must carry
        # finite densities into density.json (runout passes it to com1DFA).
        out, _ = _thin_day(tmp_path, monkeypatch, depth_m=0.5)
        for f in (out / "scenarios").glob("*/density.json"):
            assert np.isfinite(json.loads(f.read_text())["mean_kgm3"])


def test_points_find_the_2025_26_shot_clusters():
    d = fx.load()
    assert ap._clusters_at_points(POINTS, d['cluster_map'], d['transform']) == SHOTS


def test_point_order_is_lat_lon():
    # Swapped, every point is far off the raster, so nothing is found.
    d = fx.load()
    swapped = [(lon, lat) for lat, lon in POINTS]
    assert ap._clusters_at_points(swapped, d['cluster_map'], d['transform']) == []


def test_points_off_the_map_or_in_one_cluster(capsys):
    d = fx.load()
    pts = [POINTS[0], (39.70, -105.80), POINTS[0], POINTS[1]]
    assert ap._clusters_at_points(pts, d['cluster_map'], d['transform']) == SHOTS[:2]
    assert "on no cluster" in capsys.readouterr().out


def test_points_follow_the_ground_not_the_ids():
    # A renumbered map (a new season, terrain clusters) gives other ids at the
    # same places; the points pick up the new ids.
    d = fx.load()
    renum = np.where(d['cluster_map'] > 0, d['cluster_map'] + 100000, 0)
    assert ap._clusters_at_points(POINTS, renum, d['transform']) == \
        [c + 100000 for c in SHOTS]


@pytest.mark.parametrize("slope", ["Professor_PTC", "little_prof"])
def test_toml_shot_locations(slope, tmp_path):
    toml = fx.REPO / f"slopes/{slope}/slope_config.toml"
    assert ProjectConfig.from_toml(toml).explosive_trigger_points == POINTS
    src = toml.read_text()
    i = src.index("explosive_trigger_points")
    j = src.index("]\n]", i) + 2 if "]\n]" in src[i:] else src.index("\n]", i) + 2
    p = tmp_path / "slope_config.toml"
    p.write_text(src[:i] + src[j:])
    assert ProjectConfig.from_toml(p).explosive_trigger_points == []


def test_toml_rejects_lon_lat_order(tmp_path):
    from config import _lat_lon_points
    with pytest.raises(ValueError, match=r"\[lat, lon\]"):
        _lat_lon_points([[-105.8788572, 39.6488366]])


def test_a_live_date_gets_its_comparison_plot(tmp_path, monkeypatch, capsys):
    # No observed release on a live date: the plot used to raise
    # FileNotFoundError, log a traceback and not be written.
    out, meta = _thin_day(tmp_path, monkeypatch, depth_m=0.5)
    log = capsys.readouterr().out
    assert "comparison plot failed" not in log
    assert "plotting the model only" in log
    assert list(out.glob("release_comparison_*_physical_model.png"))
