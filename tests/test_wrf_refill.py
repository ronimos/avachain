"""
tests/test_wrf_refill.py — the NWP refill comes from the slope's WRF SMET.

The refill used to be data/weather/refill.smet, a one-off copy of the WRF
SMET. A new season silently outran it: outside its range the augmentation
writes nodata and SNOWPACK runs on that without complaint. These pin the
three replacements: one source (config.resolve_wrf_smet, which nwp_ingest
also reads), the season taken from season_start rather than "latest year
directory", and a loud failure when the refill is missing or short.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import forcing_pipeline as fp                                   # noqa: E402
from config import ProjectConfig, resolve_wrf_smet, wrf_season  # noqa: E402

WRF = {"fcst_base_dir": "/ssd/snowpack/fcst", "season": "", "wrf_zone": "zone055",
       "wrf_grid_id": "077963",
       "wrf_smet_file": "/ssd/snowpack/fcst/2025/zone055/077963/0779632_res.smet"}


class TestSeason:
    def test_from_season_start(self):
        assert wrf_season(WRF, "2025-10-01") == "2025"
        assert wrf_season(WRF, "2026-10-01") == "2026"

    def test_explicit_season_wins(self):
        assert wrf_season(dict(WRF, season="2024-case"), "2025-10-01") == "2024-case"

    def test_unset_season_start(self):
        assert wrf_season(WRF, "YYYY-MM-DD") is None and wrf_season(WRF, None) is None


class TestResolve:
    def test_next_season_follows_season_start_not_the_stamp(self):
        """The stamp says 2025; a 2026-27 TOML must read 2026/."""
        assert resolve_wrf_smet(WRF, "2026-10-01") == Path(
            "/ssd/snowpack/fcst/2026/zone055/077963/0779632_res.smet")

    def test_missing_parts_fall_back_to_the_stamp(self):
        wrf = {"wrf_smet_file": "/x/y.smet"}
        assert resolve_wrf_smet(wrf, "2025-10-01") == Path("/x/y.smet")
        assert resolve_wrf_smet({}, "2025-10-01") is None

    @pytest.mark.parametrize("slope", ["little_prof", "little_prof_replay",
                                       "little_prof_replay_jan", "Professor_PTC"])
    def test_existing_slopes_are_unchanged(self, slope):
        cfg = ProjectConfig.from_toml(REPO / "slopes" / slope / "slope_config.toml")
        assert cfg.wrf_smet_file == Path(WRF["wrf_smet_file"])
        assert cfg.wrf_smet_dir == Path(WRF["wrf_smet_file"]).parent


class TestRefillSource:
    def test_no_wrf_configured(self):
        with pytest.raises(fp.RefillError, match="no \\[wrf\\]"):
            fp.refill_source(ProjectConfig())

    def test_missing_file_names_the_path(self, tmp_path):
        cfg = ProjectConfig(wrf_smet_file=tmp_path / "nope.smet")
        with pytest.raises(fp.RefillError, match="nope.smet"):
            fp.refill_source(cfg)

    def test_step_smet_fails_before_writing_anything(self, tmp_path):
        cfg = ProjectConfig(project_dir=tmp_path, wrf_smet_file=tmp_path / "nope.smet")
        with pytest.raises(fp.RefillError):
            fp.step_smet(cfg)


class TestCoverage:
    idx = pd.date_range("2025-09-23T12:00", "2026-04-15T18:00", freq="6h")
    refill = pd.DataFrame({"ILWR": 1.0}, index=idx)

    def test_covered(self):
        fp.check_refill_covers(self.refill, pd.Timestamp("2025-10-01"),
                               pd.Timestamp("2026-03-31T18:00"), "r")

    def test_within_one_sample_is_covered(self):
        fp.check_refill_covers(self.refill, pd.Timestamp("2025-09-23T08:00"),
                               pd.Timestamp("2026-04-15T23:00"), "r")

    def test_season_outruns_the_refill(self):
        with pytest.raises(fp.RefillError, match="covers"):
            fp.check_refill_covers(self.refill, pd.Timestamp("2025-10-01"),
                                   pd.Timestamp("2026-05-01"), "r")

    def test_refill_starts_late(self):
        with pytest.raises(fp.RefillError):
            fp.check_refill_covers(self.refill, pd.Timestamp("2025-09-01"),
                                   pd.Timestamp("2026-03-31"), "r")


def test_install_picks_the_season_of_season_start_not_the_latest(tmp_path):
    """/ssd/snowpack/fcst/2026/ existed by 2026-10; install.py used to take it
    for every slope."""
    for season in ("2025", "2026"):
        d = tmp_path / season / "zone055" / "077963"
        d.mkdir(parents=True)
        (d / "077963_res.smet").write_text(
            "SMET 1.1 ASCII\n[HEADER]\nlatitude = 39.625\nlongitude = -105.875\n[DATA]\n")
    spec = importlib.util.spec_from_file_location("install_wrf", REPO / "install.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {"slope": {"latitude": 39.63, "longitude": -105.87, "aspect_deg": 90},
           "snowpack": {"season_start": "2025-10-01"},
           "wrf": {"fcst_base_dir": str(tmp_path), "season": ""}}
    best = mod._find_best_wrf(cfg)
    assert Path(best["smet_dir"]).parent.parent.name == "2025"
