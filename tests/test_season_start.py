"""Season start from bare ground (season_start.py, gap_fill's bare-ground
segment, cluster_carry.py) and the season plumbing under it.

What is pinned:
- Only surveys inside [season_start, season_end] count, so last season's
  rasters cannot pair with this season's first survey.
- [weather] source = "cache" gives load_weather the cache in the CSV frame's
  layout (tz-naive UTC index named 'time').
- The pre-survey ramp takes its timing from the station and its amount from
  the survey: it starts at 0 and ends exactly at the surveyed grid.
- The first survey's gap_fill replaces the bootstrap grids with that ramp.
- bootstrap starts SNOWPACK from template.sno at season_start, saves the
  survey anchor one profile step later and the daily anchor at the last
  observation, and refuses to run over an existing season.
- cluster_carry moves IDs by map coordinates, not array offsets.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import cluster_carry                                            # noqa: E402
import forcing_pipeline as fp                                   # noqa: E402
import pipeline_io                                              # noqa: E402
import season_start as ss                                       # noqa: E402
import snowpack_cycle as sc                                     # noqa: E402
from config import ProjectConfig                                # noqa: E402
from spatial_model import (bare_ground_fraction,                # noqa: E402
                           gap_fill_from_bare_ground)

T0 = pd.Timestamp("2026-10-15T00:00")
CRS = "EPSG:6342"


# ---------------------------------------------------------------- config

def _toml(tmp_path, extra=""):
    p = tmp_path / "slope_config.toml"
    p.write_text('[slope]\nname = "t"\n[paths]\nsurvey_dir = "%s"\n'
                 '[snowpack]\nseason_start = "2026-10-15"\nseason_end = "2027-04-30"\n'
                 '%s\n' % (tmp_path / "surveys", extra))
    return p


def test_season_fields_parse_and_default_off(tmp_path):
    cfg = ProjectConfig.from_toml(_toml(tmp_path))
    assert str(cfg.season_start) == "2026-10-15" and str(cfg.season_end) == "2027-04-30"
    assert cfg.start_from_bare_ground is False
    assert cfg.weather_source == "csv"
    cfg = ProjectConfig.from_toml(_toml(
        tmp_path, 'start_from_bare_ground = true\n[weather]\nsource = "cache"'))
    assert cfg.start_from_bare_ground is True and cfg.weather_source == "cache"


def test_unknown_weather_source_is_refused(tmp_path):
    with pytest.raises(ValueError, match="source"):
        ProjectConfig.from_toml(_toml(tmp_path, '[weather]\nsource = "api"'))


def test_surveys_outside_the_season_are_ignored(tmp_path):
    d = tmp_path / "surveys"
    d.mkdir()
    for name in ("260331", "261014", "261015", "261120", "270501"):
        (d / f"{name}_t_snowHeight.tif").write_text("")
    cfg = ProjectConfig.from_toml(_toml(tmp_path))
    dates = [x.isoformat() for x, _ in pipeline_io.discover_surveys(cfg)]
    assert dates == ["2026-10-15", "2026-11-20"]


def test_no_season_set_keeps_every_survey(tmp_path):
    d = tmp_path / "surveys"
    d.mkdir()
    for name in ("260331", "261120"):
        (d / f"{name}_t_snowHeight.tif").write_text("")
    p = tmp_path / "s.toml"
    p.write_text('[slope]\nname = "t"\n[paths]\nsurvey_dir = "%s"\n' % d)
    assert len(pipeline_io.discover_surveys(ProjectConfig.from_toml(p))) == 2


def test_weather_cache_is_returned_in_the_csv_layout(tmp_path):
    wdir = tmp_path / "weather"
    wdir.mkdir()
    ts = pd.date_range("2026-10-15", periods=3, freq="h", tz="UTC")
    pd.DataFrame({"timestamp": ts[[2, 0, 1]], "TA": [3.0, 1.0, 2.0],
                  "HS": [30.0, 10.0, 20.0]}).to_parquet(wdir / "CAABT.parquet")
    cfg = SimpleNamespace(weather_source="cache", weather_cache_dir=wdir,
                          summit_id="CAABT")
    wx = pipeline_io.load_weather(cfg)
    assert wx.index.name == "time" and wx.index.tz is None
    assert list(wx.index) == list(ts.tz_localize(None))
    assert list(wx["HS"]) == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------- the ramp

def _series(vals):
    return pd.Series(vals, index=pd.date_range(T0, periods=len(vals), freq="h"))


def test_ramp_follows_station_depth_and_ends_at_one():
    # The station peaks before the survey (settlement): f > 1 there, and the
    # survey hour, not the peak, is 1.
    f, rule = bare_ground_fraction(_series([0.0, 0.2, 0.1, 0.0, 0.6, 0.4]))
    assert rule == "station_hs"
    np.testing.assert_allclose(f, [0, 0.5, 0.25, 0, 1.5, 1.0])


def test_ramp_on_a_shallow_station_uses_cumulative_snowfall():
    f, rule = bare_ground_fraction(_series([0.0, 0.04, 0.02, 0.06]))
    assert rule == "station_snowfall"
    np.testing.assert_allclose(f, [0, 0.5, 0.5, 1.0])     # snowfall only adds


def test_ramp_without_station_snow_is_linear():
    f, rule = bare_ground_fraction(_series([0.0, 0.0, np.nan, 0.0, 0.0]))
    assert rule == "linear"
    np.testing.assert_allclose(f, [0, 0.25, 0.5, 0.75, 1.0])


def test_ramp_is_capped_against_station_spikes():
    f, _ = bare_ground_fraction(_series([0.0, 5.0, 0.2]))
    assert f.max() == pytest.approx(3.0)


def test_bare_ground_grids_end_exactly_at_the_survey():
    hs = np.array([[0.5, 2.0], [np.nan, 0.0]], dtype=np.float32)
    grids, _ = gap_fill_from_bare_ground(hs, _series([0.0, 0.9, 0.6]))
    g = list(grids.values())
    np.testing.assert_array_equal(g[-1], hs)
    np.testing.assert_allclose(g[1], hs * 1.5)
    assert np.all(g[0][~np.isnan(hs)] == 0) and np.isnan(g[1][1, 0])


def test_first_survey_gap_fill_replaces_the_bootstrap_grids(tmp_path):
    for d in ("surveys", "resampled", "grids"):
        (tmp_path / d).mkdir()
    (tmp_path / "surveys" / "261120_t_snowHeight.tif").write_text("")
    hs1 = np.full((2, 2), 0.8, dtype=np.float32)
    np.save(tmp_path / "resampled" / "hs_2026-11-20.npy", hs1)
    stale = tmp_path / "grids" / "hourly_2026-10-15__bootstrap.npz"
    stale.write_bytes(b"")
    p = _toml(tmp_path)
    p.write_text(p.read_text().replace(
        "[paths]\n", f'[paths]\nresampled_dir = "{tmp_path / "resampled"}"\n'
                     f'grids_dir = "{tmp_path / "grids"}"\n'))
    cfg = ProjectConfig.from_toml(p)
    assert cfg.grids_dir == tmp_path / "grids"
    idx = pd.date_range(T0, "2026-11-21", freq="h")
    wx = pd.DataFrame({"HS": np.linspace(0, 40, len(idx))}, index=idx)
    n = fp._gap_fill_bare_ground(cfg, wx, None)
    out = np.load(tmp_path / "grids" / "hourly_2026-10-15__2026-11-20.npz")
    assert not stale.exists()
    assert n == len(out["timestamps"])
    assert out["timestamps"][0] == str(T0)
    assert out["timestamps"][-1] == "2026-11-20 18:00:00"
    np.testing.assert_array_equal(out["grids"][-1], hs1)
    np.testing.assert_array_equal(out["grids"][0], 0 * hs1)


@pytest.mark.parametrize("flag", [True, False])
def test_gap_fill_writes_the_segment_only_with_the_flag(tmp_path, monkeypatch, flag):
    cfg = ProjectConfig.from_toml(_toml(tmp_path, f"start_from_bare_ground = {str(flag).lower()}"))
    calls = []
    monkeypatch.setattr(fp, "load_dem", lambda c: (np.zeros((2, 2)), None, None))
    monkeypatch.setattr(fp, "load_transport_meta", lambda c: [])
    monkeypatch.setattr(fp, "load_weather", lambda c: "wx")
    monkeypatch.setattr(fp, "_gap_fill_bare_ground",
                        lambda c, wx, cm: calls.append(wx) or 0)
    fp.step_gap_fill(cfg)
    assert calls == (["wx"] if flag else [])


def test_bare_ground_segment_waits_for_a_survey(tmp_path):
    (tmp_path / "surveys").mkdir()
    cfg = ProjectConfig.from_toml(_toml(tmp_path))
    wx = pd.DataFrame({"HS": [0.0, 1.0]}, index=pd.date_range(T0, periods=2, freq="h"))
    assert fp._gap_fill_bare_ground(cfg, wx, None) == 0


# ---------------------------------------------------------------- bootstrap

STUB = r"""#!/bin/bash
while [[ $# -gt 0 ]]; do
  case "$1" in -c) ini="$2"; shift 2;; -b) b="$2"; shift 2;; -e) e="$2"; shift 2;; *) shift;; esac
done
exp=$(awk -F'= *' '/^EXPERIMENT/{print $2}' "$ini")
out=$(awk -F'= *' '/^\[Output\]/{o=1} o && /^SNOWPATH/{print $2; exit}' "$ini")
snin=$(awk -F'= *' '/^\[Input\]/{i=1} i && /^SNOWPATH/{print $2; exit}' "$ini")
pin=$(awk -F'= *' '/^ProfileDate/{print $2}' "$snin/$exp.sno")
echo "$b $e $pin" >> "$out/$exp.hist"
printf 'ProfileDate      = %s\n' "$e" > "$out/${exp}_${exp}.sno"
"""
CIDS = ("cluster_0001", "cluster_0002")


@pytest.fixture
def boot(tmp_path, monkeypatch):
    slope = tmp_path / "slope"
    for d in ("input/smet", "input/snow", "output"):
        (slope / d).mkdir(parents=True)
    (slope / "input/snow/template.sno").write_text("ProfileDate = WRFDATE\n")
    stub = tmp_path / "stub"
    stub.write_text(STUB)
    stub.chmod(0o755)
    monkeypatch.setenv("SNOWPACK_BIN", str(stub))
    res = tmp_path / "resampled"
    res.mkdir()
    with rasterio.open(res / "dem_1m.tif", "w", driver="GTiff", height=2, width=2,
                       count=1, dtype="float32", crs=CRS, nodata=-9999,
                       transform=from_origin(424000, 4389000, 1, 1)) as dst:
        dst.write(np.array([[3600, 3601], [3602, -9999]], dtype=np.float32), 1)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    np.save(analysis / "cluster_map.npy", np.array([[1, 1], [2, 0]]))
    cfg = SimpleNamespace(
        slope_dir=slope, pro_dir=slope / "output", smet_dir=slope / "input/smet",
        grids_dir=tmp_path / "grids", resampled_dir=res, analysis_dir=analysis,
        cluster_map_path=analysis / "cluster_map.npy",
        zarr_path=slope / "output" / "slope_snowpack.zarr",
        slopes_config_dir=REPO / "slopes" / "little_prof_replay",
        season_start=T0.date(), start_from_bare_ground=True, flight_hour_utc=18)
    idx = pd.date_range(T0, "2026-10-20T13:00", freq="h")
    wx = pd.DataFrame({"HS": np.r_[np.zeros(48), np.linspace(0, 30, len(idx) - 48)]},
                      index=idx)
    return cfg, wx


def _fake_smet(cfg):
    grids = list(cfg.grids_dir.glob("hourly_*.npz"))
    ts = np.load(grids[0])["timestamps"]
    for cid in CIDS:
        (cfg.smet_dir / f"{cid}.smet").write_text(
            f"SMET 1.1 ASCII\n[HEADER]\nstation_id = {cid}\nslope_angle = 31\n"
            f"slope_azi = 120\nfields = timestamp TA\n[DATA]\n"
            f"{pd.Timestamp(str(ts[0])):%Y-%m-%dT%H:%M:%S} -5\n"
            f"{pd.Timestamp(str(ts[-1])):%Y-%m-%dT%H:%M:%S} -6\n")


def test_bootstrap_runs_from_bare_ground_and_sets_both_anchors(boot):
    cfg, wx = boot
    zarr = []
    out = ss.bootstrap(cfg, smet_fn=_fake_smet, zarr_fn=lambda c: zarr.append(1), wx=wx)
    s_anchor, d_anchor = pd.Timestamp("2026-10-15T06:00"), pd.Timestamp("2026-10-20T12:00")
    assert out == {"clusters": 2, "survey_anchor": s_anchor, "daily_anchor": d_anchor}
    for cid in CIDS:
        hist = [ln.split() for ln in (cfg.pro_dir / f"{cid}.hist").read_text().splitlines()]
        # first run: fresh from template.sno at season_start; second: from it
        assert hist[0] == ["2026-10-15T00:00:00", "2026-10-15T06:00", "2026-10-15"]
        assert hist[1][1:] == ["2026-10-20T12:00", "2026-10-15T06:00"]
        assert sc._profile_date(cfg.pro_dir / f"{cid}_survey.sno") == s_anchor
        assert sc._profile_date(cfg.pro_dir / f"{cid}_anchor.sno") == d_anchor
    assert sc.read_survey_anchor(cfg) == s_anchor
    assert zarr == [1]
    g = np.load(cfg.grids_dir / "hourly_2026-10-15__bootstrap.npz")
    assert g["timestamps"][0] == str(T0) and g["timestamps"][-1] == str(d_anchor)
    assert np.nanmax(g["grids"][0]) == 0.0
    assert np.nanmax(g["grids"][-1]) == pytest.approx(wx.loc[d_anchor, "HS"] / 100)


def test_bootstrap_refuses_an_existing_season(boot):
    cfg, wx = boot
    (cfg.pro_dir / "cluster_0001_anchor.sno").write_text("ProfileDate = 2026-03-31T18:00\n")
    with pytest.raises(sc.CycleError, match="Archive"):
        ss.bootstrap(cfg, smet_fn=_fake_smet, zarr_fn=lambda c: None, wx=wx)


def test_bootstrap_needs_the_bare_ground_flag(boot):
    cfg, wx = boot
    cfg.start_from_bare_ground = False
    with pytest.raises(sc.CycleError, match="start_from_bare_ground"):
        ss.bootstrap(cfg, smet_fn=_fake_smet, zarr_fn=lambda c: None, wx=wx)


def test_bootstrap_refuses_a_cache_that_starts_after_season_start(boot):
    cfg, wx = boot
    with pytest.raises(sc.CycleError, match="season_start"):
        ss.bootstrap(cfg, smet_fn=_fake_smet, zarr_fn=lambda c: None,
                     wx=wx.loc["2026-10-16":])


# ---------------------------------------------------------------- carry

def test_carry_moves_ids_by_map_coordinates():
    # Old map: 6 x 6 at 1 m, origin 2 m south and 1 m east of the new one.
    # Rows count southward, so old (r, c) sits at new (r + 2, c + 1). Array
    # offsets alone would put it at (r, c).
    old = np.arange(1, 37, dtype=np.int32).reshape(6, 6)
    t_old = from_origin(1001, 1998, 1, 1)
    t_new = from_origin(1000, 2000, 1, 1)
    domain = np.zeros((8, 8), bool)
    domain[2:8, 1:7] = True                 # exactly the old footprint
    cmap, dom, st = cluster_carry.carry_ids(old, t_old, CRS, (8, 8), t_new, CRS,
                                            domain, max_fill_m=0)
    np.testing.assert_array_equal(cmap[2:8, 1:7], old)
    assert (cmap[~domain] == 0).all()
    assert st["cells_filled_from_nearest"] == 0 and st["cells_left_out"] == 0


def test_carry_fills_near_cells_and_leaves_far_ones_out():
    old = np.ones((4, 4), dtype=np.int32)
    old[:, 2:] = 7
    t = from_origin(0, 4, 1, 1)
    domain = np.zeros((4, 10), bool)
    domain[:, :9] = True                    # 5 new columns east of the old map
    domain[0, 0] = False                    # and one old cell now outside
    cmap, dom, st = cluster_carry.carry_ids(old, t, CRS, (4, 10), t, CRS,
                                            domain, max_fill_m=2.0)
    assert (cmap[:, 4:6] == 7).all()        # within 2 m of the old edge
    assert (cmap[:, 6:] == 0).all() and not dom[:, 6:].any()
    assert cmap[0, 0] == 0 and st["cells_dropped_outside_domain"] == 1
    assert st["cells_filled_from_nearest"] == 8 and st["cells_left_out"] == 12
    assert st["lost_clusters"] == []
