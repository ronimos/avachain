"""Tests for the operational-replay clock ($AVACHAIN_AS_OF).

The replay harness hides the future from the pipeline through exactly two
seams: `discover_surveys` (every survey consumer re-globs independently, so
the cutoff has to live here) and `get_stable_ts` (the WRF observed/forecast
boundary). Both previously read the wall clock or no bound at all.

This needs unit coverage rather than an end-to-end check: a replay run that
silently saw one survey too many would still produce plausible output, and
the only visible symptom would be a divergence metric that looked better than
it should.
"""
import datetime
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from config import AS_OF_ENV, ProjectConfig, replay_as_of   # noqa: E402
from nwp_ingest import get_stable_ts                        # noqa: E402
from pipeline_io import discover_surveys                    # noqa: E402

SURVEY_DATES = ["2025-11-26", "2026-01-14", "2026-02-05", "2026-02-14", "2026-03-31"]


@pytest.fixture(autouse=True)
def _clear_clock(monkeypatch):
    """Never inherit a real $AVACHAIN_AS_OF from the caller's environment."""
    monkeypatch.delenv(AS_OF_ENV, raising=False)


@pytest.fixture
def cfg(tmp_path):
    survey_dir = tmp_path / "data" / "surveys"
    survey_dir.mkdir(parents=True)
    for d in SURVEY_DATES:
        y, m, day = d.split("-")
        (survey_dir / f"{y[2:]}{m}{day}_Professor_PTC_snowHeight.tif").touch()
    return ProjectConfig(project_dir=tmp_path)


@pytest.fixture
def wrf_smet(tmp_path):
    """Minimal WRF SMET: 6-hourly ticks spanning the replay window."""
    ticks = ["2026-02-04T18:00:00", "2026-02-05T00:00:00", "2026-02-05T06:00:00",
             "2026-02-14T00:00:00", "2026-03-08T00:00:00", "2026-06-04T18:00:00"]
    p = tmp_path / "wrf.smet"
    p.write_text("SMET 1.1 ASCII\n[HEADER]\nfields = timestamp TA\n[DATA]\n"
                 + "".join(f"{t} 270.0\n" for t in ticks))
    return p


class TestReplayAsOf:
    def test_none_when_unset(self):
        assert replay_as_of() is None

    def test_empty_env_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "")
        assert replay_as_of() is None

    def test_reads_date_from_env(self, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2026-02-14")
        assert replay_as_of() == datetime.datetime(2026, 2, 14)

    def test_reads_datetime_from_env(self, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2026-02-14T06:00")
        assert replay_as_of() == datetime.datetime(2026, 2, 14, 6, 0)

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2026-02-14")
        assert replay_as_of("2026-03-08") == datetime.datetime(2026, 3, 8)

    def test_datetime_passes_through(self):
        dt = datetime.datetime(2026, 2, 5)
        assert replay_as_of(dt) is dt


class TestDiscoverSurveysCutoff:
    def test_no_cutoff_sees_every_survey(self, cfg):
        assert len(discover_surveys(cfg)) == len(SURVEY_DATES)

    def test_cutoff_hides_later_surveys(self, cfg, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2026-02-05")
        got = [d.isoformat() for d, _ in discover_surveys(cfg)]
        assert got == ["2025-11-26", "2026-01-14", "2026-02-05"]

    def test_cutoff_date_itself_is_visible(self, cfg, monkeypatch):
        """A survey landing on the simulated day has arrived, not pending."""
        monkeypatch.setenv(AS_OF_ENV, "2026-02-14")
        assert datetime.date(2026, 2, 14) in [d for d, _ in discover_surveys(cfg)]

    def test_explicit_as_of_overrides_env(self, cfg, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2026-03-31")
        got = [d for d, _ in discover_surveys(cfg, as_of="2026-01-14")]
        assert max(got) == datetime.date(2026, 1, 14)

    def test_cutoff_before_all_surveys_yields_none(self, cfg, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2025-01-01")
        assert discover_surveys(cfg) == []

    def test_result_stays_sorted(self, cfg, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2026-02-14")
        dates = [d for d, _ in discover_surveys(cfg)]
        assert dates == sorted(dates)


class TestStableTsCutoff:
    def test_env_clock_bounds_the_boundary(self, wrf_smet, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2026-02-05T06:00")
        assert str(get_stable_ts(wrf_smet)) == "2026-02-05 06:00:00+00:00"

    def test_picks_last_tick_at_or_before_clock(self, wrf_smet, monkeypatch):
        """02-14 sits between the 02-05T06 and 02-14 ticks when clocked early."""
        monkeypatch.setenv(AS_OF_ENV, "2026-02-13")
        assert str(get_stable_ts(wrf_smet)) == "2026-02-05 06:00:00+00:00"

    def test_explicit_now_overrides_env(self, wrf_smet, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2026-02-05")
        assert str(get_stable_ts(wrf_smet, now="2026-03-08")) == \
            "2026-03-08 00:00:00+00:00"

    def test_clock_before_file_start_gives_none(self, wrf_smet, monkeypatch):
        monkeypatch.setenv(AS_OF_ENV, "2025-01-01")
        assert get_stable_ts(wrf_smet) is None

    def test_unset_clock_reaches_the_file_end(self, wrf_smet):
        """Live behaviour: wall clock is past every tick in the fixture."""
        assert str(get_stable_ts(wrf_smet)) == "2026-06-04 18:00:00+00:00"
