"""Readers of the WRF file wait out the CAIC morning rebuild.

/ssd/snowpack/fcst_snowpack.sh does `rm -rf fcst/<season>`, copies output/
back, then writes the forecast (~07:39-07:55 local, 2026-11-04). A reader in
that window found no file (RefillError, survey FAILED) or, worse, a file
without its forecast rows, and built a short forecast without any error.
"""
import os
import time
from pathlib import Path

import pandas as pd
import pytest

import nwp_ingest as ni
import forcing_pipeline as fp
from config import ProjectConfig


def _wrf(path, last="2026-11-07T18:00:00"):
    path.write_text("SMET 1.1 ASCII\n[HEADER]\nfields = timestamp TA\n[DATA]\n"
                    f"2026-11-01T00:00:00 -3\n2026-11-04T12:00:00 -4\n{last} -5\n")
    return path


class Clock:
    """Fake wall clock; sleeping advances it and can run a callback."""
    def __init__(self, t, on_sleep=None):
        self.t, self.on_sleep, self.sleeps = t, on_sleep, 0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps += 1
        self.t += s
        if self.on_sleep:
            self.on_sleep(self)


@pytest.fixture
def real_settle(monkeypatch):
    monkeypatch.setattr(ni, "WRF_SETTLE_S", 600)


NOW = pd.Timestamp("2026-11-04T15:00")      # naive, as the replay clock passes it


def test_ready_file(tmp_path, real_settle):
    p = _wrf(tmp_path / "w.smet")
    old = time.time() - 3600
    os.utime(p, (old, old))
    assert ni.wrf_ready(p, 48, now=NOW) is None


def test_a_file_being_rewritten_is_not_ready(tmp_path, real_settle):
    p = _wrf(tmp_path / "w.smet")
    assert "changed" in ni.wrf_ready(p, now=NOW)


def test_forecast_rows_not_written_yet(tmp_path, real_settle):
    # output/ copied back, forecast step still running: data end at "now"
    p = _wrf(tmp_path / "w.smet", last="2026-11-04T15:00:00")
    old = time.time() - 3600
    os.utime(p, (old, old))
    assert "short of T_stable" in ni.wrf_ready(p, 48, now=NOW)
    assert ni.wrf_ready(p, 0, now=NOW) is None          # the refill needs no forecast


def test_waits_for_the_rebuild_to_finish(tmp_path, real_settle):
    p = tmp_path / "w.smet"                              # deleted by the rebuild
    t0 = time.time()

    def rebuild(clock):
        if clock.sleeps == 3:                            # reappears after 3 polls
            _wrf(p)
            os.utime(p, (clock.t, clock.t))
    c = Clock(t0, rebuild)
    why = ni.wait_for_wrf(p, 48, now=NOW, timeout_s=1800, poll_s=60,
                          sleep=c.sleep, clock=c)
    assert why is None
    assert c.sleeps == 3 + 10                            # appear, then settle 600 s


def test_gives_up_after_the_timeout(tmp_path, real_settle):
    c = Clock(time.time())
    why = ni.wait_for_wrf(tmp_path / "gone.smet", timeout_s=300, poll_s=60,
                          sleep=c.sleep, clock=c)
    assert "does not exist" in why and c.sleeps == 5


def test_refill_source_raises_only_after_waiting(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ni, "wait_for_wrf", lambda p, *a, **k: calls.append(p) or "gone")
    with pytest.raises(fp.RefillError, match="not usable: gone"):
        fp.refill_source(ProjectConfig(wrf_smet_file=tmp_path / "w.smet"))
    assert calls == [tmp_path / "w.smet"]


def test_scripts_take_t_stable_only_after_nwp_ingest_succeeded():
    import importlib.util, tomllib
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("install_mod", repo / "install.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = tomllib.loads((repo / "slopes" / "Professor_PTC" / "slope_config.toml").read_text())
    for sh in (m._render_run_operational(cfg), m._render_run_daily(cfg)):
        a = sh.index("$NWP --lead-hours")
        block = sh[a:sh.index('STABLE_TS=$(echo', a)]
        assert "NWP_OK=0" in block and "if [[ $NWP_OK -eq 1 ]]; then" in block


class TestForecastWindow:
    """Ron, 2026-11-04: lead 60 h, and if T+2 is past the WRF outlook use its
    last forward-looking timestamp. One window for SMETs and the pass."""

    def test_full_lead_inside_the_outlook(self, tmp_path):
        p = _wrf(tmp_path / "w.smet", last="2026-11-07T18:00:00")
        stable, end, clamped = ni.forecast_window(p, 60, now=pd.Timestamp("2026-11-04T19:00"))
        assert (stable, end, clamped) == (pd.Timestamp("2026-11-04T12:00", tz="UTC"),
                                          pd.Timestamp("2026-11-07T00:00", tz="UTC"), False)

    def test_clamped_to_the_last_forecast_row(self, tmp_path):
        # a run 6 h after midnight-MST: T_stable 11-06 12 UTC, +60 h > last row
        p = _wrf(tmp_path / "w.smet", last="2026-11-06T12:00:00 -4\n2026-11-07T18:00:00")
        _, end, clamped = ni.forecast_window(p, 60, now=pd.Timestamp("2026-11-06T13:00"))
        assert end == pd.Timestamp("2026-11-07T18:00", tz="UTC") and clamped

    def test_no_forecast_rows(self, tmp_path):
        p = _wrf(tmp_path / "w.smet", last="2026-11-04T12:00:00")
        assert ni.forecast_window(p, 60, now=pd.Timestamp("2026-11-05T00:00")) is None


def test_scripts_take_the_window_from_nwp_ingest():
    import importlib.util, tomllib
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("install_mod_w", repo / "install.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = tomllib.loads((repo / "slopes" / "Professor_PTC" / "slope_config.toml").read_text())
    for sh in (m._render_run_operational(cfg), m._render_run_daily(cfg)):
        assert "$NWP --lead-hours 60 ||" in sh
        assert "$NWP --lead-hours 60 --print-window" in sh
        assert "+ pd.Timedelta(hours=48)" not in sh        # no second, hard-coded window
