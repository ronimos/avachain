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
