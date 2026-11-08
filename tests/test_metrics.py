"""Season analytics (metrics.py, docs/analytics_design.md, Ron 2026-11-08).

The recorder's one rule is that it never changes a run's outcome, so most of
these tests break it on purpose and check that nothing escapes. The log
parser is pinned on the real logs of the first live night, 2026-11-08 (the
bare-ground failure and its rerun), with the alert address scrubbed.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import jan18_fixture  # noqa: F401  (puts src/avachain on sys.path)

import handoff
import metrics
from config import ProjectConfig

DATA = Path(__file__).parent / "data" / "metrics"


def _cfg(tmp_path, **kw) -> ProjectConfig:
    toml = tmp_path / "slope_config.toml"
    toml.write_text('[calibration]\nversion = "test-1"\n')
    return ProjectConfig(output_dir=tmp_path / "out", slope_dir=tmp_path / "sp",
                         slope_name="Professor_PTC", toml_path=toml,
                         season_start=dt.date(2026, 11, 7), **kw)


def _lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


# --- log parser, on the real 11-08 logs ----------------------------------

class TestParseRunLog:
    def test_failed_first_night(self):
        p = metrics.parse_run_log((DATA / "daily_20261108_000001.log").read_text())
        assert p["stations"]["CAABT"] == {"rows": 36, "first_utc": "2026-11-06 20:00:00+00:00",
                                          "last_utc": "2026-11-08 07:00:00+00:00", "cached": 36}
        assert p["smet_append"] == {"files_updated": 6541, "files": 6541, "rows": 85033}
        assert p["t_stable"] == "2026-11-08T06:00"
        assert p["nwp_edate"] == "2026-11-10T18:00"
        passes = p["snowpack_passes"]
        assert [x["pass"] for x in passes] == ["observed", "forecast"]
        assert [x["seconds"] for x in passes] == [539, 535]      # 07:01:20-07:10:19, 07:12:17-07:21:12
        assert all(x["ok"] == 6541 and x["failed"] == 0 and x["slope_clamped"] == 41
                   for x in passes)
        assert p["zarr_update_s"] == 64
        assert p["forecast_zarr_s"] == 150
        assert p["cycle_date"] == "2026-11-08" and p["n_forecast_days"] == 3
        assert {k: v["status"] for k, v in p["forecast_days"].items()} == \
            {"T_0": "failed", "T_1": "failed", "T_2": "failed"}
        assert "days_published" not in p
        assert p["alerts"] == ["[avachain Professor_PTC] daily run FAILED (exit 1) at "
                               "Step 5: Analysis + scenario generation"]

    def test_no_address_reaches_a_record(self):
        p = metrics.parse_run_log("ALERT sent to someone@state.co.us: [x] subject\n")
        assert p["alerts"] == ["[x] subject"]
        assert "@" not in json.dumps(p)

    def test_no_snow_rerun(self):
        p = metrics.parse_run_log((DATA / "rerun_20261108_nosnow.log").read_text())
        assert p["no_snow_max_hs_m"] == 0.0
        assert p["days_published"] == 3
        assert p["runout_exit"] == {"model": 0, "shots": 0}

    def test_empty_log(self):
        p = metrics.parse_run_log("")
        assert p["snowpack_passes"] == [] and p["alerts"] == []


# --- the record never changes the run -------------------------------------

def test_run_record_written(tmp_path):
    cfg = _cfg(tmp_path)
    out = metrics.run_record(cfg, "daily", 1, DATA / "daily_20261108_000001.log",
                             steps="aws_ingest=12 smet_append=30 snowpack=1400 analysis=40",
                             total_s=1460)
    rec = _lines(out)[-1]
    assert out == tmp_path / "out" / "metrics" / "2026-27" / "runs.jsonl"
    assert rec["kind"] == "run" and rec["schema"] == metrics.SCHEMA
    assert rec["exit_code"] == 1
    assert rec["failed_step"] == "Step 5: Analysis + scenario generation"
    assert rec["steps_s"] == {"aws_ingest": 12, "smet_append": 30, "snowpack": 1400,
                              "analysis": 40}
    assert rec["calibration_version"] == "test-1"
    assert rec["cycle_date"] == "2026-11-08"


def test_record_never_raises(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "metrics").write_text("a file where the directory should be")
    assert metrics.record(cfg, "run", {"x": 1}) is None
    assert "WARNING: metrics" in capsys.readouterr().out


def test_cli_returns_zero_on_a_broken_toml(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("this is [not toml")
    assert metrics.main(["--toml", str(bad), "run", "--kind", "daily",
                         "--exit-code", "0"]) == 0
    assert metrics.main(["--toml", str(bad), "survey", "--date", "2026-11-26"]) == 0
    assert "WARNING: metrics" in capsys.readouterr().out


def test_report_deadline_margin(tmp_path):
    cfg = _cfg(tmp_path)
    www = tmp_path / "www"
    cfg.toml_path.write_text(f'[runout]\npath_profile = "Professor_PTC"\n'
                             f'publish_root = "{www}"\n')
    rep = www / "forecasts" / "Professor_PTC" / "20261108" / "report.html"
    rep.parent.mkdir(parents=True)
    rep.write_text("<html/>")
    t = dt.datetime(2026, 11, 8, 2, 16).timestamp()           # local 02:16
    import os
    os.utime(rep, (t, t))
    info = metrics._report_info(cfg, "2026-11-08")
    assert info == {"report_published": True, "report_mtime_local": "2026-11-08T02:16:00",
                    "deadline_margin_min": 44}


# --- survey record --------------------------------------------------------

def test_survey_summary(tmp_path):
    cfg = _cfg(tmp_path)
    an = cfg.analysis_dir
    an.mkdir(parents=True)
    d = "2026-11-26"
    stats = [{"cid": 1, "hs_model_m": 0.50, "hs_survey_m": 0.40},
             {"cid": 2, "hs_model_m": 0.20, "hs_survey_m": 0.30},
             {"cid": 3, "hs_model_m": 0.90, "hs_survey_m": 0.60}]
    (an / f"anchor_stats_{d}.json").write_text(json.dumps(stats))
    (an / f"anchor_counts_{d}.json").write_text(json.dumps({"top_down": 2, "below": 1}))
    (an / f"anchor_check_{d}.json").write_text(json.dumps({"clusters": 3, "p1_m": -0.1,
                                                          "p99_m": 0.01, "below": 1}))
    (an / f"survey_check_{d}.json").write_text(json.dumps(
        {"verdict": "PASS", "p10_m": 0.02, "p50_m": 0.4, "cover": 0.9, "messages": []}))
    smet = cfg.slope_dir / "input" / "smet"
    smet.mkdir(parents=True)
    for cid, alt, azi in ((1, 3590.2, 10.0), (2, 3612.0, 95.0), (3, 3688.0, 350.0)):
        (smet / f"cluster_{cid:04d}.smet").write_text(
            f"SMET 1.1 ASCII\n[HEADER]\naltitude = {alt}\nslope_azi = {azi}\n[DATA]\n")
    s = metrics.survey_summary(cfg, d)
    assert s["frame_check"]["verdict"] == "PASS"
    assert s["anchor_counts"] == {"top_down": 2, "below": 1}
    assert s["after_anchor"]["below"] == 1
    allp = s["prior_minus_survey_m"]["all"]
    assert allp["n"] == 3 and allp["p50"] == pytest.approx(0.1)
    assert set(s["prior_minus_survey_m"]["by_elevation_100m"]) == {"3500", "3600"}
    assert s["prior_minus_survey_m"]["by_aspect"]["N"]["n"] == 2       # 10 and 350 deg
    assert s["prior_minus_survey_m"]["by_aspect"]["E"]["n"] == 1


def test_octants_are_compass_bearings():
    assert [metrics._octant(a) for a in (0, 44, 46, 90, 180, 270, 337, 338)] == \
        ["N", "NE", "NE", "E", "S", "W", "NW", "N"]


# --- keeping what a rerun or the next cycle would overwrite ----------------

def test_keep_forecast_store_takes_the_snapshot_timesteps(tmp_path):
    cfg = _cfg(tmp_path)
    times = pd.date_range("2026-11-08T12:00", periods=6, freq="6h")
    ds = xr.Dataset({"HS": (("location", "time"), np.arange(12.0).reshape(2, 6))},
                    coords={"location": ["cluster_0001", "cluster_0002"], "time": times})
    cfg.pro_dir.mkdir(parents=True)
    ds.to_zarr(cfg.forecast_zarr_path, mode="w")
    out = metrics.keep_forecast_store(
        cfg, "2026-11-08", [pd.Timestamp("2026-11-08T18:00"), pd.Timestamp("2026-11-09T18:00")])
    kept = xr.open_zarr(out)
    assert out == tmp_path / "out" / "issued" / "2026-27" / "2026-11-08" / "forecast_store.zarr"
    assert list(pd.DatetimeIndex(kept.time.values)) == \
        [pd.Timestamp("2026-11-08T18:00"), pd.Timestamp("2026-11-09T18:00")]
    assert kept.HS.values.tolist() == [[1.0, 5.0], [7.0, 11.0]]


def test_keep_forecast_store_without_a_store(tmp_path):
    assert metrics.keep_forecast_store(_cfg(tmp_path), "2026-11-08",
                                       [pd.Timestamp("2026-11-08T18:00")]) is None


def _stage(root, text):
    st = root / f"stage_{text}"
    st.mkdir()
    (st / "metadata.json").write_text(json.dumps({"v": text}))
    return st


def test_handoff_keeps_the_replaced_set(tmp_path):
    final = tmp_path / "release" / "2026-11-08" / "T_0"
    final.parent.mkdir(parents=True)
    handoff.publish(_stage(tmp_path, "first"), final, "release")
    keep = tmp_path / "issued" / "superseded" / "T_0"
    handoff.publish(_stage(tmp_path, "second"), final, "release", keep_old=keep)
    assert json.loads((final / "metadata.json").read_text()) == {"v": "second"}
    assert json.loads((keep / "metadata.json").read_text()) == {"v": "first"}
    assert not list(final.parent.glob(".T_0.old-*"))


def test_handoff_without_keep_still_deletes(tmp_path):
    final = tmp_path / "release" / "T_0"
    final.parent.mkdir(parents=True)
    handoff.publish(_stage(tmp_path, "a"), final, "release")
    handoff.publish(_stage(tmp_path, "b"), final, "release")
    assert not list(final.parent.glob(".T_0.old-*"))


def test_keep_published_reports_copies_model_and_shots(tmp_path):
    cfg = _cfg(tmp_path)
    www = tmp_path / "www"
    cfg.toml_path.write_text(f'[runout]\npath_profile = "Professor_PTC"\n'
                             f'publish_root = "{www}"\n')
    for sub in ("", "shots"):
        d = www / sub / "forecasts" / "Professor_PTC" / "20261108"
        d.mkdir(parents=True)
        (d / "report.html").write_text(f"old {sub}")
    dest = metrics.keep_published_reports(cfg, "2026-11-08")
    assert (dest / "model" / "report.html").read_text() == "old "
    assert (dest / "shots" / "report.html").read_text() == "old shots"
    # a copy: the live page stays up
    assert (www / "forecasts" / "Professor_PTC" / "20261108" / "report.html").exists()


def test_keep_published_reports_first_run_keeps_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.toml_path.write_text(f'[runout]\npath_profile = "P"\npublish_root = "{tmp_path}/www"\n')
    assert metrics.keep_published_reports(cfg, "2026-11-08") is None


# --- observations ---------------------------------------------------------

def test_observe_appends(tmp_path):
    cfg = _cfg(tmp_path)
    out = metrics.observe(cfg, "control_shot", "2026-12-03 07:40", "CDOT log",
                          observer="Ron", fields={"shot_point": "PW-3", "result": "no_result"})
    rec = _lines(out)[-1]
    assert out.name == "observations.jsonl"
    assert rec["type"] == "control_shot" and rec["result"] == "no_result"
    assert rec["time_local"] == "2026-12-03T07:40" and rec["linked_cycle"] == "2026-12-03"


def test_observe_rejects_an_unknown_type(tmp_path):
    with pytest.raises(ValueError):
        metrics.observe(_cfg(tmp_path), "slide", "2026-12-03 07:40", "x")


def test_observe_cli_fields(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.toml_path.write_text("[snowpack]\nseason_start = 2026-11-07\n"
                             f'[paths]\noutput_dir = "{tmp_path / "out"}"\n')
    assert metrics.main(["--toml", str(cfg.toml_path), "observe", "--type", "avalanche",
                         "--time", "2026-12-10 14:05", "--source", "CAIC obs",
                         "--field", "size=\"D2\"", "--field", "reached_road=false",
                         "--field", "crown_depth_m=0.8"]) == 0
    rec = _lines(tmp_path / "out" / "metrics" / "2026-27" / "observations.jsonl")[-1]
    assert rec["size"] == "D2" and rec["reached_road"] is False and rec["crown_depth_m"] == 0.8


# --- trigger funnel -------------------------------------------------------

def test_funnel_and_near_misses_are_recorded(tmp_path, monkeypatch):
    import analysis_pipeline as ap
    from test_trigger_fallback import _thin_day
    _thin_day(tmp_path, monkeypatch, depth_m=0.5, model=True, n_triggers=1,
              no_propagation=True)
    f = ap._FUNNEL
    assert f["start_zone_with_features"] >= f["tau_g_slope_sk38"] >= f["slab_thickness"] \
        >= f["elevation"] > 0
    nm = [r for r in ap._NEAR_MISS if "cid" in r]
    assert 0 < len(nm) <= ap.N_NEAR_MISS
    assert [r["sk38"] for r in nm] == sorted(r["sk38"] for r in nm)
    assert "thresholds" in ap._NEAR_MISS[-1]


def test_cycle_record_from_a_published_set(tmp_path):
    cfg = _cfg(tmp_path)
    final = tmp_path / "release" / "2026-11-08" / "T_0"
    final.parent.mkdir(parents=True)
    st = tmp_path / "stage"
    st.mkdir()
    (st / "metadata.json").write_text(json.dumps({"n_trigger_locations": 2,
                                                  "n_below_depth_floor": 0}))
    pd.DataFrame({"trigger_cluster": [5, 5, 9], "release_area_m2": [100.0, 200.0, 300.0],
                  "mean_depth_m": [0.5, 0.6, 0.7]}).to_csv(st / "summary.csv", index=False)
    handoff.publish(st, final, "release", snapshot_date="2026-11-08", horizon="T_0",
                    status="ok", n_scenarios=3, trigger_source="model",
                    cycle_date="2026-11-08")
    out = metrics.cycle_record(cfg, final, funnel={"start_zone_with_features": 9},
                               near_miss=[])
    rec = _lines(out)[-1]
    assert rec["horizon"] == "T_0" and rec["n_scenarios"] == 3
    assert rec["trigger_clusters"] == [5, 9]
    assert rec["release_area_m2"]["p50"] == 200.0
    assert rec["funnel"] == {"start_zone_with_features": 9}
    assert rec["shots"] is False


# --- the generated scripts call the recorder and keep their own exit status --

def _fake_project(tmp_path, aws_rc: int, metrics_rc: int, monkeypatch):
    """A project dir whose entry points are stubs: aws_ingest exits aws_rc,
    metrics.py logs its argv and exits metrics_rc; every later step succeeds."""
    import importlib.util
    import sys
    import tomllib
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("install_mod", repo / "install.py")
    inst = importlib.util.module_from_spec(spec)
    sys.modules["install_mod"] = inst
    spec.loader.exec_module(inst)
    cfg = tomllib.load(open(repo / "slopes" / "little_prof" / "slope_config.toml", "rb"))
    proj = tmp_path / "proj"
    src = proj / "src" / "avachain"
    src.mkdir(parents=True)
    logs = tmp_path / "logs"
    calls = tmp_path / "metrics_calls.txt"
    (src / "config.py").write_text(
        f"print('AVA_SLOPE_DIR={tmp_path}/sp'); print('AVA_LOGS_DIR={logs}')\n")
    (src / "aws_ingest.py").write_text(f"import sys; sys.exit({aws_rc})\n")
    for name in ("smet_append", "nwp_ingest", "snowpack_cycle", "analysis_pipeline",
                 "runout_step", "alerts"):
        (src / f"{name}.py").write_text("pass\n")
    (src / "metrics.py").write_text(
        f"import sys, os\nopen({str(calls)!r}, 'a').write(' '.join(sys.argv[1:]) + "
        f"' RUN_ID=' + os.environ.get('AVACHAIN_RUN_ID', '') + '\\n')\nsys.exit({metrics_rc})\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").symlink_to(sys.executable)
    (proj / ".venv" / "bin").mkdir(parents=True)
    (proj / ".venv" / "bin" / "activate").write_text(f"export PATH={bindir}:$PATH\n")
    script = proj / "run_daily_little_prof.sh"
    script.write_text(inst._render_run_daily(cfg))
    script.chmod(0o755)
    return script, calls


@pytest.mark.parametrize("aws_rc,metrics_rc,expect", [(1, 0, 1), (1, 3, 1), (0, 3, 0)])
def test_daily_script_records_and_keeps_its_exit_status(tmp_path, monkeypatch,
                                                        aws_rc, metrics_rc, expect):
    import subprocess
    script, calls = _fake_project(tmp_path, aws_rc, metrics_rc, monkeypatch)
    env = {k: v for k, v in __import__("os").environ.items() if k != "AVACHAIN_RUN_ID"}
    r = subprocess.run(["bash", str(script), "--no-nwp"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == expect, r.stdout + r.stderr
    line = calls.read_text().splitlines()[-1]
    assert f"run --kind daily --exit-code {aws_rc}" in line
    assert "--log " in line and "RUN_ID=20" in line
    if aws_rc == 0:
        assert "aws_ingest=" in line and "runout=" in line
