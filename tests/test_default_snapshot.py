"""analysis_pipeline's default snapshot date.

It was hard-coded to "2026-01-18" (the validation event). run_<slope>.sh and
run_daily_<slope>.sh call analyze/scenarios without --snapshot-date unless
given one, so every operational run would have analysed and written scenarios
for January 18. The default must be "now": the replay clock if set, else
today's UTC date.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import analysis_pipeline as ap                                 # noqa: E402


def test_default_follows_the_replay_clock(monkeypatch):
    monkeypatch.setenv("AVACHAIN_AS_OF", "2026-02-10T20:00")
    assert ap.default_snapshot_date() == "2026-02-10"


def test_default_is_today_utc_when_live(monkeypatch):
    monkeypatch.delenv("AVACHAIN_AS_OF", raising=False)
    assert ap.default_snapshot_date() == datetime.now(timezone.utc).date().isoformat()


def test_an_explicit_snapshot_is_kept(monkeypatch):
    monkeypatch.setenv("AVACHAIN_AS_OF", "2026-03-07T14:00")
    assert ap.resolve_snapshot_date("2026-01-18") == "2026-01-18"
    assert ap.resolve_snapshot_date(None) == "2026-03-07"
