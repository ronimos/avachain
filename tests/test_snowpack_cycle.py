"""SNOWPACK restart orchestration: survey and daily cycles (snowpack_cycle.py).

These drive the real generated run_snowpack.sh with a stub SNOWPACK. The stub
appends one history line per run to output/<cid>.hist: -b, -e, the input .sno
ProfileDate, and whether that input had been scoured. analyze/reinit/Zarr are
replaced by recorders; the fake reinit marks the .sno files it scours.

What is pinned:
- A survey re-runs from the PREVIOUS survey's state, not the latest one, so
  the window the new survey now brackets is re-simulated.
- A detected avalanche splits the run at the event: stop, analyze, scour,
  continue. The continuation starts from the scoured state, for the whole
  domain, at the event time.
- The daily cycle restarts from the daily anchor, never from the +48 h
  forecast state left in <cid>_<cid>.sno.
- Every segment ends with every cluster at the same time, or the cycle stops.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import snowpack_cycle as sc                                     # noqa: E402

STUB = r"""#!/bin/bash
while [[ $# -gt 0 ]]; do
  case "$1" in -c) ini="$2"; shift 2;; -b) b="$2"; shift 2;; -e) e="$2"; shift 2;; *) shift;; esac
done
exp=$(awk -F'= *' '/^EXPERIMENT/{print $2}' "$ini")
out=$(awk -F'= *' '/^\[Output\]/{o=1} o && /^SNOWPATH/{print $2; exit}' "$ini")
snin=$(awk -F'= *' '/^\[Input\]/{i=1} i && /^SNOWPATH/{print $2; exit}' "$ini")
if [[ -n "${STUB_FAIL:-}" && "$exp" == "$STUB_FAIL" ]]; then echo "stub failure"; exit 1; fi
pin=$(awk -F'= *' '/^ProfileDate/{print $2}' "$snin/$exp.sno")
sc=$(grep -q SCOURED "$snin/$exp.sno" && echo yes || echo no)
echo "$b $e $pin $sc" >> "$out/$exp.hist"
printf 'ProfileDate      = %s\n' "$e" > "$out/${exp}_${exp}.sno"
echo "FINISHED running stub"
"""

S0 = pd.Timestamp("2026-02-05T18:00")          # previous survey (anchor)
S1 = pd.Timestamp("2026-02-10T18:00")          # new survey
CIDS = ("cluster_0001", "cluster_0002")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    slope = tmp_path / "slope"
    for d in ("input/smet", "input/snow", "output"):
        (slope / d).mkdir(parents=True)
    (slope / "input/snow/template.sno").write_text("ProfileDate = WRFDATE\n")
    for cid in CIDS:
        (slope / "input/smet" / f"{cid}.smet").write_text(
            f"SMET 1.1 ASCII\n[HEADER]\nstation_id = {cid}\nslope_angle = 31\n"
            "slope_azi = 120\nfields = timestamp TA\n[DATA]\n"
            "2025-11-26T18:00:00 -5\n2026-02-11T00:00:00 -6\n")
        for kind in ("survey", "anchor"):
            (slope / "output" / f"{cid}_{kind}.sno").write_text(
                f"ProfileDate      = {S0:%Y-%m-%dT%H:%M:%S}\n")
        # The latest state: the end of yesterday's forecast, NOT a restart point.
        (slope / "output" / f"{cid}_{cid}.sno").write_text(
            "ProfileDate      = 2026-02-07T18:00:00\n")
    (slope / "output" / sc.SURVEY_ANCHOR_META).write_text(
        json.dumps({"survey_ts": S0.strftime(sc.TS_FMT)}))
    stub = tmp_path / "stub"
    stub.write_text(STUB)
    stub.chmod(0o755)
    monkeypatch.setenv("SNOWPACK_BIN", str(stub))
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    return SimpleNamespace(
        slope_dir=slope, pro_dir=slope / "output", smet_dir=slope / "input/smet",
        analysis_dir=analysis, boundaries_dir=tmp_path / "boundaries",
        slopes_config_dir=REPO / "slopes" / "little_prof_replay",
        flight_hour_utc=18, toml_path=None,
        event_mask_for=lambda ts: None)


class Rec:
    def __init__(self):
        self.calls = []

    def zarr(self, cfg, since):
        self.calls.append(("zarr", since))

    def forecast(self, cfg, since):
        self.calls.append(("forecast", since))

    def scour_check(self, cfg, ev):
        self.calls.append(("scour_check", ev.ts))

    def analyze(self, cfg, date):
        self.calls.append(("analyze", date))

    def reinit(self, cfg, ev, scour=("cluster_0002",)):
        self.calls.append(("reinit", ev.ts))
        for cid in scour:
            p = cfg.pro_dir / f"{cid}_{cid}.sno"
            p.write_text(p.read_text() + "SCOURED\n")


def _hist(cfg, cid):
    return [ln.split() for ln in (cfg.pro_dir / f"{cid}.hist").read_text().splitlines()]


def _pd(cfg, name):
    return sc._profile_date(cfg.pro_dir / name)


def _events(cfg, **evs):
    (cfg.analysis_dir / "avalanche_events.json").write_text(json.dumps(
        {pid: {"reinit_needed": True, "event_timestamp": ts} for pid, ts in evs.items()}))


def _survey(cfg, rec, obs_end, nwp=None):
    return sc.survey_cycle(cfg, S1, obs_end, nwp, analyze_fn=rec.analyze,
                           reinit_fn=rec.reinit, zarr_fn=rec.zarr,
                           forecast_fn=rec.forecast, scour_check_fn=rec.scour_check)


def test_survey_reruns_from_the_previous_survey_not_the_latest_state(cfg):
    rec = Rec()
    obs = pd.Timestamp("2026-02-11T00:00")
    _survey(cfg, rec, obs, pd.Timestamp("2026-02-13T00:00"))
    for cid in CIDS:
        h = _hist(cfg, cid)
        assert h[0][:3] == ["2026-02-05T18:00:00", "2026-02-10T18:00",
                            "2026-02-05T18:00:00"]          # from S0, not 02-07
        assert h[1][:2] == ["2026-02-10T18:00", "2026-02-11T00:00"]
        assert h[2][:2] == ["2026-02-11T00:00", "2026-02-13T00:00"]   # forecast from anchor
        assert _pd(cfg, f"{cid}_survey.sno") == S1
        assert _pd(cfg, f"{cid}_anchor.sno") == obs
    assert sc.read_survey_anchor(cfg) == S1
    assert rec.calls == [("zarr", S0), ("forecast", obs)]


def test_avalanche_splits_the_run_and_the_whole_domain_continues_scoured(cfg):
    rec = Rec()
    ev = "2026-02-08T12:00:00Z"
    _events(cfg, **{"2026-02-05__2026-02-10": ev})
    _survey(cfg, rec, pd.Timestamp("2026-02-11T00:00"))
    t = pd.Timestamp("2026-02-08T12:00")
    assert rec.calls[:3] == [("zarr", S0), ("analyze", "2026-02-08"), ("reinit", t)]
    for cid, scoured in (("cluster_0001", "no"), ("cluster_0002", "yes")):
        h = _hist(cfg, cid)
        assert h[0][1] == "2026-02-08T12:00"                       # stop at event
        assert h[1][0] == "2026-02-08T12:00"                       # all resume there
        assert h[1][1] == "2026-02-10T18:00"
        assert h[1][3] == scoured                                  # from scoured state


def test_events_outside_the_survey_window_are_not_reinitialised(cfg):
    rec = Rec()
    _events(cfg, **{"a__b": "2026-02-05T12:00:00Z",                 # before S0
                    "c__d": "2026-02-10T20:00:00Z"})                # after S1
    _survey(cfg, rec, pd.Timestamp("2026-02-11T00:00"))
    assert not any(c[0] == "reinit" for c in rec.calls)


def test_incoherent_survey_anchor_stops_before_running(cfg):
    (cfg.pro_dir / "cluster_0002_survey.sno").write_text(
        "ProfileDate      = 2026-01-31T18:00:00\n")
    with pytest.raises(sc.CycleError, match="not at 2026-02-05T18:00"):
        _survey(cfg, Rec(), pd.Timestamp("2026-02-11T00:00"))
    assert not (cfg.pro_dir / "cluster_0001.hist").exists()


def test_a_cluster_that_fails_stops_the_cycle(cfg, monkeypatch):
    monkeypatch.setenv("STUB_FAIL", "cluster_0002")
    with pytest.raises(sc.CycleError, match="cluster_0002"):
        _survey(cfg, Rec(), pd.Timestamp("2026-02-11T00:00"))
    assert sc.read_survey_anchor(cfg) == S0                        # anchor not moved


def test_new_survey_must_be_after_the_anchor(cfg):
    with pytest.raises(sc.CycleError, match="not after"):
        sc.survey_cycle(cfg, S0, pd.Timestamp("2026-02-11T00:00"),
                        zarr_fn=Rec().zarr)


def test_daily_restarts_from_the_anchor_not_the_forecast_state(cfg):
    rec = Rec()
    a = pd.Timestamp("2026-02-10T18:00")
    for cid in CIDS:
        (cfg.pro_dir / f"{cid}_anchor.sno").write_text(
            f"ProfileDate      = {a:%Y-%m-%dT%H:%M:%S}\n")
        (cfg.pro_dir / f"{cid}_{cid}.sno").write_text(
            "ProfileDate      = 2026-02-12T18:00:00\n")       # +48 h forecast end
    obs = pd.Timestamp("2026-02-11T18:00")
    sc.daily_cycle(cfg, obs, pd.Timestamp("2026-02-13T18:00"),
                   zarr_fn=rec.zarr, forecast_fn=rec.forecast)
    for cid in CIDS:
        h = _hist(cfg, cid)
        assert h[0][:3] == ["2026-02-10T18:00:00", "2026-02-11T18:00",
                            "2026-02-10T18:00:00"]
        assert h[1][0] == "2026-02-11T18:00"                       # forecast from new anchor
        assert _pd(cfg, f"{cid}_anchor.sno") == obs
    assert rec.calls == [("zarr", a), ("forecast", obs)]


def test_last_observed_ts_ignores_nwp_rows(cfg):
    from nwp_ingest import NWP_FLAG
    p = cfg.smet_dir / "cluster_0001.smet"
    p.write_text(p.read_text() + NWP_FLAG + "\n2026-02-13T00:00:00 -9\n")
    assert sc.last_observed_ts(cfg) == pd.Timestamp("2026-02-11T00:00")


def test_restart_points_snap_to_the_6h_profile_grid(cfg):
    """SNOWPACK writes profiles every 6 h (PROF_DAYS_BETWEEN = 0.25). An anchor
    at the last hourly observation (14:00) left the main store ending at 12:00,
    two hours before its own restart point (replay tick 2026-02-07)."""
    rec = Rec()
    a = pd.Timestamp("2026-02-10T18:00")
    for cid in CIDS:
        (cfg.pro_dir / f"{cid}_anchor.sno").write_text(
            f"ProfileDate      = {a:%Y-%m-%dT%H:%M:%S}\n")
    out = sc.daily_cycle(cfg, pd.Timestamp("2026-02-11T14:00"),
                         zarr_fn=rec.zarr, forecast_fn=rec.forecast)
    assert out["obs_end"] == pd.Timestamp("2026-02-11T12:00")
    assert _hist(cfg, "cluster_0001")[0][1] == "2026-02-11T12:00"
    assert _pd(cfg, "cluster_0001_anchor.sno") == pd.Timestamp("2026-02-11T12:00")


def test_grid_floor():
    assert sc.on_output_grid(pd.Timestamp("2026-02-07T14:00")) == pd.Timestamp("2026-02-07T12:00")
    assert sc.on_output_grid(pd.Timestamp("2026-02-07T18:00")) == pd.Timestamp("2026-02-07T18:00")
    assert sc.on_output_grid(pd.Timestamp("2026-02-08T05:59")) == pd.Timestamp("2026-02-08T00:00")


# ---------------------------------------------------------------------------
# A reinit must survive the restart (check_scour_held)
# ---------------------------------------------------------------------------

def test_scour_checked_after_the_segment_that_restarts_from_it(cfg):
    rec = Rec()
    _events(cfg, **{"a__b": "2026-02-07T12:00:00Z", "c__d": "2026-02-09T12:00:00Z"})
    _survey(cfg, rec, pd.Timestamp("2026-02-11T00:00"))
    t1, t2 = pd.Timestamp("2026-02-07T12:00"), pd.Timestamp("2026-02-09T12:00")
    names = [c for c in rec.calls if c[0] in ("reinit", "scour_check")]
    assert names == [("reinit", t1), ("scour_check", t1),
                     ("reinit", t2), ("scour_check", t2)]


def _scour_fixture(cfg, hs_mod_cm, hs_after_m=0.60, slope=60.0, stamp="2026-02-08T12:00"):
    (cfg.analysis_dir / "reinit_stats_2026-02-08.json").write_text(
        json.dumps([{"cid": 2, "hs_after_m": hs_after_m}]))
    rows = f"{stamp}:00   1.0   {hs_mod_cm}   3.0\n" if hs_mod_cm is not None else ""
    (cfg.pro_dir / "cluster_0002_cluster_0002.smet").write_text(
        f"SMET 1.1 ASCII\n[HEADER]\nslope_angle = {slope}\n"
        f"fields = timestamp MS_Snow HS_mod HS_meas\n[DATA]\n{rows}"
        "2026-02-08T18:00:00   1.0   999.0   3.0\n")


EV = sc.Event(pd.Timestamp("2026-02-08T12:00"), "a__b", None)


def test_scour_held_passes(cfg):
    _scour_fixture(cfg, 60.0)                  # HS_mod is vertical, like hs_after
    assert sc.check_scour_held(cfg, EV)["max_deposition_m"] == pytest.approx(0.0)


def test_scour_refilled_stops_the_cycle(cfg):
    _scour_fixture(cfg, 70.0)                  # +10 cm deposited at the restart
    with pytest.raises(sc.CycleError, match="cluster_0002 \\+10 cm"):
        sc.check_scour_held(cfg, EV)


def test_scour_missing_restart_row_stops_the_cycle(cfg):
    _scour_fixture(cfg, None)
    with pytest.raises(sc.CycleError, match="without a restart row"):
        sc.check_scour_held(cfg, EV)
