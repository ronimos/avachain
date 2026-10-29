"""
tests/test_config.py — Tests for ProjectConfig.from_toml() and related methods.

Run with:  pytest tests/test_config.py -v
"""

from __future__ import annotations

from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from config import ProjectConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_TOML = """\
[slope]
name = "test_slope"

[paths]
project_dir = "{project_dir}"
dem         = "data/dem/test.tif"
output_dir  = "outputs/test"
slope_dir   = "snowpack/test"

[[stations]]
role        = "summit"
sql_id      = "TESTST"
name        = "Test Summit"
lat         = 39.64
lon         = -105.87
elev_m      = 3800.0
sql_columns = ["temp", "rh"]

[[stations]]
role        = "base"
sql_id      = "TESTBS"
name        = "Test Base"
lat         = 39.63
lon         = -105.86
elev_m      = 3500.0
sql_columns = ["pcpac"]

[scenarios]
n_triggers = 3
size_factors = [0.85, 1.00, 1.15]

[wrf]
wrf_smet_file = "/ssd/fcst/test.smet"
"""


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "slope_config.toml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# from_toml — core fields
# ---------------------------------------------------------------------------

class TestFromToml:
    def test_slope_name_loaded(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.slope_name == "test_slope"

    def test_summit_id_loaded(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.summit_id == "TESTST"

    def test_base_id_loaded(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.base_id == "TESTBS"

    def test_summit_elevation_loaded(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.summit_alt_m == pytest.approx(3800.0)

    def test_stations_list_populated(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert len(cfg.stations) == 2
        roles = {s["role"] for s in cfg.stations}
        assert roles == {"summit", "base"}

    def test_n_triggers_loaded(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.n_triggers == 3

    def test_size_factors_loaded(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.size_factors == pytest.approx([0.85, 1.00, 1.15])

    def test_wrf_smet_file_loaded(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.wrf_smet_file == Path("/ssd/fcst/test.smet")

    def test_project_dir_resolves(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.project_dir == Path(str(tmp_path))

    def test_output_dir_is_under_project_dir(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        # __post_init__ resolves output_dir relative to project_dir
        assert cfg.output_dir == Path(str(tmp_path)) / "outputs/test"


# ---------------------------------------------------------------------------
# from_toml — missing optional sections fall back to defaults
# ---------------------------------------------------------------------------

class TestFromTomlDefaults:
    BARE_TOML = """\
[slope]
name = "bare"

[paths]
project_dir = "{project_dir}"

[[stations]]
role   = "summit"
sql_id = "BARE"
name   = "Bare Summit"
lat    = 39.0
lon    = -105.0
elev_m = 3000.0
sql_columns = []
"""

    def test_missing_wrf_section_gives_none(self, tmp_path):
        p = _write_toml(tmp_path, self.BARE_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.wrf_smet_file is None
        assert cfg.wrf_smet_dir is None

    def test_missing_scenarios_section_uses_dataclass_defaults(self, tmp_path):
        p = _write_toml(tmp_path, self.BARE_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.n_triggers == 5
        assert 1.00 in cfg.size_factors

    def test_missing_base_station_gives_empty_strings(self, tmp_path):
        p = _write_toml(tmp_path, self.BARE_TOML.format(project_dir=str(tmp_path)))
        cfg = ProjectConfig.from_toml(p)
        assert cfg.base_id == ""

    def test_empty_stations_list(self, tmp_path):
        toml = """\
[slope]
name = "empty"
[paths]
project_dir = "{project_dir}"
""".format(project_dir=str(tmp_path))
        p = _write_toml(tmp_path, toml)
        cfg = ProjectConfig.from_toml(p)
        assert cfg.stations == []


# ---------------------------------------------------------------------------
# release_geojsons_for_date
# ---------------------------------------------------------------------------

class TestReleaseGeojsonsForDate:
    def _cfg_with_boundaries(self, tmp_path: Path) -> ProjectConfig:
        # ProjectConfig resolves boundary_kml relative to project_dir;
        # boundaries_dir is then that parent.  We set boundary_kml to a dummy
        # path whose parent is tmp_path so boundaries_dir == tmp_path.
        from dataclasses import replace
        cfg = ProjectConfig(project_dir=tmp_path)
        # Override the boundary_kml so boundaries_dir == tmp_path
        cfg.boundary_kml = tmp_path / "dummy.kml"
        return cfg

    def _write_geojson(self, directory: Path, date_str: str):
        p = directory / f"avalanche_release_area_{date_str}.geojson"
        p.write_text('{"type": "FeatureCollection", "features": []}')
        return p

    def test_returns_files_at_or_before_snapshot(self, tmp_path):
        cfg = self._cfg_with_boundaries(tmp_path)
        self._write_geojson(tmp_path, "20260115")
        self._write_geojson(tmp_path, "20260118")
        self._write_geojson(tmp_path, "20260122")
        result = cfg.release_geojsons_for_date("2026-01-18")
        names = [p.name for p in result]
        assert "avalanche_release_area_20260115.geojson" in names
        assert "avalanche_release_area_20260118.geojson" in names
        assert "avalanche_release_area_20260122.geojson" not in names

    def test_returns_empty_list_when_none_match(self, tmp_path):
        cfg = self._cfg_with_boundaries(tmp_path)
        self._write_geojson(tmp_path, "20260122")
        result = cfg.release_geojsons_for_date("2026-01-15")
        assert result == []

    def test_results_sorted_chronologically(self, tmp_path):
        cfg = self._cfg_with_boundaries(tmp_path)
        self._write_geojson(tmp_path, "20260118")
        self._write_geojson(tmp_path, "20260110")
        self._write_geojson(tmp_path, "20260115")
        result = cfg.release_geojsons_for_date("2026-01-20")
        dates = [p.stem.replace("avalanche_release_area_", "") for p in result]
        assert dates == sorted(dates)

    def test_invalid_snapshot_date_returns_empty(self, tmp_path):
        cfg = self._cfg_with_boundaries(tmp_path)
        self._write_geojson(tmp_path, "20260115")
        result = cfg.release_geojsons_for_date("not-a-date")
        assert result == []


# ---------------------------------------------------------------------------
# validation_dir: the scoring target, never the reinit mask
# ---------------------------------------------------------------------------

class TestValidationDir:
    """A validation_dir polygon replaces the boundaries_dir one of the same
    date for scoring (release_geojsons_for_date) and nowhere else."""

    def _cfg(self, tmp_path: Path) -> ProjectConfig:
        (tmp_path / "boundaries").mkdir()
        (tmp_path / "validation").mkdir()
        cfg = ProjectConfig(project_dir=tmp_path, validation_dir=Path("validation"))
        cfg.boundary_kml = tmp_path / "boundaries" / "dummy.kml"
        return cfg

    def _gj(self, d: Path, date_str: str) -> Path:
        p = d / f"avalanche_release_area_{date_str}.geojson"
        p.write_text('{"type": "FeatureCollection", "features": []}')
        return p

    def test_validation_file_wins_for_its_date(self, tmp_path):
        cfg = self._cfg(tmp_path)
        self._gj(tmp_path / "boundaries", "20260110")
        self._gj(tmp_path / "boundaries", "20260118")
        val = self._gj(tmp_path / "validation", "20260118")
        result = cfg.release_geojsons_for_date("2026-01-20")
        assert [p.name[-13:-8] for p in result] == ["60110", "60118"]
        assert result[-1] == val

    def test_validation_only_date_is_included(self, tmp_path):
        cfg = self._cfg(tmp_path)
        val = self._gj(tmp_path / "validation", "20260118")
        assert cfg.release_geojsons_for_date("2026-01-18") == [val]
        assert cfg.release_geojsons_for_date("2026-01-17") == []

    def test_reinit_mask_ignores_validation_dir(self, tmp_path):
        from datetime import date
        cfg = self._cfg(tmp_path)
        obs = self._gj(tmp_path / "boundaries", "20260118")
        self._gj(tmp_path / "validation", "20260118")
        assert cfg.event_mask_for(date(2026, 1, 18)) == obs

    def test_off_by_default(self, tmp_path):
        p = _write_toml(tmp_path, MINIMAL_TOML.format(project_dir=tmp_path))
        assert ProjectConfig.from_toml(p).validation_dir is None

    def test_read_from_toml(self, tmp_path):
        toml = MINIMAL_TOML.format(project_dir=tmp_path).replace(
            'slope_dir   = "snowpack/test"',
            'slope_dir   = "snowpack/test"\nvalidation_dir = "data/validation"')
        cfg = ProjectConfig.from_toml(_write_toml(tmp_path, toml))
        assert cfg.validation_dir == tmp_path / "data/validation"


# ---------------------------------------------------------------------------
# Per-directory [paths] overrides (deployment layout)
# ---------------------------------------------------------------------------

import subprocess

from config import OVERRIDABLE_DIRS, shell_paths

REPO = Path(__file__).resolve().parent.parent


class TestDirOverrides:
    def test_defaults_are_the_historical_layout(self, tmp_path):
        cfg = ProjectConfig.from_toml(_write_toml(tmp_path, MINIMAL_TOML.format(project_dir=tmp_path)))
        out = tmp_path / "outputs" / "test"
        assert cfg.analysis_dir == out / "analysis"
        assert cfg.scenarios_dir == out / "scenarios"
        assert cfg.grids_dir == out / "hourly_grids"
        assert cfg.resampled_dir == out / "resampled_1m"
        assert cfg.event_masks_dir == out / "event_masks"
        assert cfg.logs_dir == out / "logs"
        assert cfg.weather_cache_dir == tmp_path / "outputs" / "weather"

    def test_relative_override_resolves_against_project_dir(self, tmp_path):
        toml = MINIMAL_TOML.format(project_dir=tmp_path).replace(
            '[paths]\n', '[paths]\nscenarios_dir = "handoff/P/release"\n')
        cfg = ProjectConfig.from_toml(_write_toml(tmp_path, toml))
        assert cfg.scenarios_dir == tmp_path / "handoff" / "P" / "release"
        assert cfg.analysis_dir == tmp_path / "outputs" / "test" / "analysis"

    def test_unknown_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown"):
            ProjectConfig(dir_overrides={"analysis": "x"})

    def test_deploy_tree_leaves_nothing_under_the_repo(self, tmp_path):
        """Every path a deployment TOML sets is absolute; none may fall back
        into the repo, which is where a forgotten key would silently write."""
        deploy = tmp_path / "deploy"
        keys = {
            "survey_dir": "data/surveys/snow_height/P",
            "dem": "data/static/P/dem.tif",
            "weather_csv": "data/weather/weather_data.csv",
            "boundary_kml": "data/static/P/boundary.kml",
            "start_zone_kml": "data/static/P/start_zone.kml",
            "release_geojson": "data/static/P/release.geojson",
            "avalanche_events": "data/static/P/avalanche_events.json",
            "validation_dir": "data/static/P/validation",
            "output_dir": "outputs/P",
            "slope_dir": "work/snowpack/P",
            "analysis_dir": "work/release/P/analysis",
            "event_masks_dir": "work/release/P/event_masks",
            "grids_dir": "work/snowpack/P/hourly_grids",
            "resampled_dir": "work/snowpack/P/resampled_1m",
            "models_dir": "work/snowpack/P/models",
            "scenarios_dir": "handoff/P/release",
            "plots_dir": "outputs/P/plots",
            "logs_dir": "logs/P",
            "weather_cache_dir": "work/weather",
        }
        assert set(OVERRIDABLE_DIRS) <= set(keys)
        body = "\n".join(f'{k} = "{deploy / v}"' for k, v in keys.items())
        toml = (f'[slope]\nname = "P"\n\n[paths]\nproject_dir = "{REPO}"\n{body}\n')
        cfg = ProjectConfig.from_toml(_write_toml(tmp_path, toml))

        resolved = {name: getattr(cfg, name) for name in (
            "survey_dir", "dem_path", "weather_csv", "boundary_kml", "start_zone_kml",
            "release_geojson", "avalanche_events_path", "validation_dir", "output_dir",
            "slope_dir", "smet_dir", "pro_dir", "zarr_path", *OVERRIDABLE_DIRS)}
        inside = {k: v for k, v in resolved.items() if not Path(v).is_relative_to(deploy)}
        assert not inside, f"paths left under the repo: {inside}"

    def test_print_paths_is_shell_evaluable(self, tmp_path):
        toml = MINIMAL_TOML.format(project_dir=tmp_path).replace(
            '[paths]\n', '[paths]\nlogs_dir = "my logs/P"\n')
        p = _write_toml(tmp_path, toml)
        cfg = ProjectConfig.from_toml(p)
        script = (f'eval "$({sys.executable} {REPO}/src/avachain/config.py '
                  f'--toml {p} --print-paths)"; echo "$AVA_LOGS_DIR|$AVA_ANALYSIS_DIR"')
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        logs, analysis = out.stdout.strip().split("|")
        assert logs == str(tmp_path / "my logs" / "P")          # space survives quoting
        assert analysis == str(cfg.analysis_dir)
        assert set(shell_paths(cfg)) >= {"AVA_SLOPE_DIR", "AVA_LOGS_DIR",
                                         "AVA_ANALYSIS_DIR", "AVA_SCENARIOS_DIR"}
