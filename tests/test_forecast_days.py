"""Which store and time each forecast day reads (plan_forecast_days).

Ron, 2026-11-04: snapshots at 18 UTC; the daily run at 00:00 local reads the
observed store to T_stable (~06 UTC) and the NWP forecast store after it;
"if T+2 is past the WRF outlook, use WRF's last forward-looking timestamp".
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
from analysis_pipeline import plan_forecast_days  # noqa: E402

T = pd.Timestamp


def _plan(main_end, fc_end, n=3):
    main = (T("2026-11-07"), T(main_end)) if main_end else None
    fc = (T("2026-12-02T00:00"), T(fc_end)) if fc_end else None
    return plan_forecast_days("2026-12-02", n, 18, main, fc)


def test_midnight_run_reads_every_day_from_the_forecast():
    p = _plan("2026-12-02T06:00", "2026-12-04T18:00")
    assert [(d["label"], d["store"], d["ts"], d["clamped"]) for d in p] == [
        ("T_0", "forecast", T("2026-12-02T18:00"), False),
        ("T_1", "forecast", T("2026-12-03T18:00"), False),
        ("T_2", "forecast", T("2026-12-04T18:00"), False)]


def test_afternoon_survey_run_reads_today_from_observations():
    p = _plan("2026-12-02T18:00", "2026-12-05T06:00")
    assert [d["store"] for d in p] == ["observed", "forecast", "forecast"]


def test_t2_past_the_outlook_is_clamped_to_its_last_step():
    p = _plan("2026-12-02T06:00", "2026-12-04T12:00")
    assert (p[2]["ts"], p[2]["clamped"], p[2]["date"]) == (T("2026-12-04T12:00"), True, "2026-12-04")


def test_a_day_the_outlook_cannot_reach_is_skipped():
    p = _plan("2026-12-02T06:00", "2026-12-03T06:00")
    assert [d["store"] for d in p] == ["forecast", "forecast", None]
    assert p[1]["clamped"] and p[1]["ts"] == T("2026-12-03T06:00")
    assert "not after the previous day" in p[2]["reason"]


def test_no_nwp_only_what_the_observations_cover():
    p = _plan("2026-12-02T18:00", None)
    assert [d["store"] for d in p] == ["observed", None, None]
    assert p[1]["reason"] == "no forecast store"


def test_a_stale_forecast_store_is_not_used():
    """nwp_ingest failed today: the cycle keeps yesterday's forecast store,
    which still covers today's 18 UTC. It must not be read as today's."""
    from analysis_pipeline import fresh_forecast_span
    main = (T("2025-10-01"), T("2026-12-02T06:00"))
    assert fresh_forecast_span(main, (T("2026-12-02T12:00"), T("2026-12-04T18:00")))  # this cycle's
    assert fresh_forecast_span(main, (T("2026-12-01T12:00"), T("2026-12-03T18:00"))) is None  # yesterday's
    assert fresh_forecast_span(main, (T("2026-12-03T00:00"), T("2026-12-05T00:00"))) is None  # a gap
    assert fresh_forecast_span(None, (T("2026-12-02T12:00"), T("2026-12-04T18:00"))) is None
