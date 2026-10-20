"""Tick schedule of the replay driver (replay_run.plan)."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from replay_run import plan                                       # noqa: E402

SURVEYS = [date(2026, 1, 27), date(2026, 2, 5), date(2026, 2, 10), date(2026, 2, 12)]


def _p(**kw):
    return plan(SURVEYS, date(2026, 2, 6), date(2026, 2, 12), date(2026, 2, 5),
                kw.get("scen", {"surveys"}))


def test_survey_days_run_the_survey_script_after_the_flight():
    by = {t["date"]: t for t in _p()}
    assert by["2026-02-10"]["kind"] == "survey"
    assert by["2026-02-10"]["as_of"] == "2026-02-10T20:00"     # flight at 18:00
    assert by["2026-02-11"]["kind"] == "daily"
    assert by["2026-02-11"]["as_of"] == "2026-02-11T14:00"


def test_surveys_at_or_before_the_anchor_are_not_rerun():
    assert all(t["kind"] == "daily" for t in plan(
        SURVEYS, date(2026, 2, 5), date(2026, 2, 9), date(2026, 2, 5), set()))


def test_one_tick_per_day_inclusive():
    assert [t["date"] for t in _p()][0] == "2026-02-06"
    assert len(_p()) == 7


def test_scenarios_only_on_survey_days_and_named_dates():
    by = {t["date"]: t["scenarios"] for t in _p(scen={"surveys", "2026-02-08"})}
    assert by == {"2026-02-06": False, "2026-02-07": False, "2026-02-08": True,
                  "2026-02-09": False, "2026-02-10": True, "2026-02-11": False,
                  "2026-02-12": True}
