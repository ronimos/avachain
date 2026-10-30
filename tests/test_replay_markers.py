"""Replay driver with survey markers (replay_run --markers).

The deployment acceptance replay must send each survey through the same
door as operations: the producer's .READY marker, claimed by
run_<slope>.sh. These pin the driver's side of that: future surveys are
invisible after setup, a tick delivers exactly its own survey, and the
post-tick check fails unless the claim ran to DONE.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import replay_run as rr                          # noqa: E402
import survey_markers as sm                      # noqa: E402
from config import ProjectConfig                 # noqa: E402
from pipeline_io import discover_surveys         # noqa: E402

T0, D1, D2 = dt.date(2026, 2, 5), dt.date(2026, 2, 10), dt.date(2026, 2, 12)


def _cfg(tmp_path: Path, states: dict) -> ProjectConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    toml = tmp_path / "slope_config.toml"
    toml.write_text(f'[slope]\nname = "Professor_PTC"\n\n'
                    f'[paths]\nproject_dir = "{tmp_path}"\nsurvey_dir = "surveys"\n'
                    f'output_dir = "out"\nslope_dir = "work"\n\n'
                    f'[surveys]\nmarkers = true\n')
    sdir = tmp_path / "surveys"
    sdir.mkdir()
    for d, state in states.items():
        tif = sdir / f"{d:%y%m%d}_Professor_PTC_snowHeight.tif"
        tif.write_bytes(b"")
        if state:
            sm.marker_path(tif, state).write_text("{}")
    return ProjectConfig.from_toml(toml)


def _states(cfg):
    return {d: sm.state_of(t) for d, t in sm.survey_tifs(cfg)}


def test_setup_hides_only_surveys_after_t_start(tmp_path):
    cfg = _cfg(tmp_path, {T0: "DONE", D1: "DONE", D2: "DONE"})
    assert len(rr.hide_future_surveys(cfg, T0)) == 2
    assert _states(cfg) == {T0: "DONE", D1: None, D2: None}
    assert [d for d, _ in discover_surveys(cfg)] == [T0]


def test_delivery_is_claimed_by_the_survey_script(tmp_path):
    cfg = _cfg(tmp_path, {T0: "DONE", D1: None, D2: None})
    rr.deliver_survey(cfg, D1, "2026-02-10T20:00")
    assert _states(cfg) == {T0: "DONE", D1: "READY", D2: None}
    assert sm.claim(cfg, pid=1) == D1           # D2 is not delivered, so not taken


def test_failed_survey_is_redelivered_as_a_retry(tmp_path):
    cfg = _cfg(tmp_path, {T0: "DONE", D1: "FAILED"})
    rr.deliver_survey(cfg, D1, "2026-02-10T20:00")
    assert _states(cfg)[D1] == "READY"


@pytest.mark.parametrize("state", ["READY", "RUNNING", "DONE"])
def test_delivering_a_survey_that_already_has_a_marker_fails(tmp_path, state):
    cfg = _cfg(tmp_path, {T0: "DONE", D1: state})
    with pytest.raises(rr.Tier3Failure, match=state):
        rr.deliver_survey(cfg, D1, "2026-02-10T20:00")


def test_check_requires_done_and_nothing_running(tmp_path):
    tick = {"date": D1.isoformat(), "kind": "survey"}
    cfg = _cfg(tmp_path, {T0: "DONE", D1: "FAILED"})
    with pytest.raises(rr.Tier3Failure, match="FAILED, not DONE"):
        rr.check_markers(cfg, tick)

    cfg2 = _cfg(tmp_path / "b", {T0: "DONE", D1: "RUNNING"})
    with pytest.raises(rr.Tier3Failure, match="RUNNING"):
        rr.check_markers(cfg2, {"date": "2026-02-11", "kind": "daily"})

    cfg3 = _cfg(tmp_path / "c", {T0: "DONE", D1: "DONE"})
    sm.marker_path(sm.survey_tifs(cfg3)[1][1], "DONE").write_text(
        json.dumps({"status": "DONE", "exit_code": 0}))
    assert rr.check_markers(cfg3, tick)["marker"]["exit_code"] == 0
