"""wrf_archive keeps one snapshot per WRF-SNOWPACK forecast cycle.

The CAIC run deletes and rebuilds /ssd/snowpack/fcst/<season>/ every day, so
the forecast a cycle issued is lost unless copied. 2025-26 has no per-cycle
archive and supports no forecast-skill claim; this is the 2026-27 fix.
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
import wrf_archive as wa

SMET = """SMET 1.1 ASCII
[HEADER]
fields = timestamp TA
[DATA]
2026-10-14T12:00:00   -3.1
{rows}"""


def _cell(root, season="2026", rows="2026-11-07T18:00:00   -5.0\n", age_min=60):
    cell = root / "fcst" / season / "zone055" / "077963"
    cell.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        (cell / f"077963{i}_res.smet").write_text(SMET.format(rows=rows))
        (cell / f"077963{i}_res.sno").write_text("sno")
    t = dt.datetime.now(dt.timezone.utc).timestamp() - age_min * 60
    for p in cell.iterdir():
        os.utime(p, (t, t))
    return cell


def test_first_run_snapshots_the_whole_cell(tmp_path):
    cell = _cell(tmp_path)
    dest = wa.archive_cell(cell, tmp_path / "arch")
    assert dest.parent == tmp_path / "arch" / "2026" / "zone055" / "077963"
    assert sorted(p.name for p in dest.iterdir()) == sorted(
        [p.name for p in cell.iterdir()] + ["manifest.json"])
    m = json.loads((dest / "manifest.json").read_text())
    assert m["data_first"] == "2026-10-14T12:00:00" and m["data_last"] == "2026-11-07T18:00:00"
    assert m["cycle_written_utc"] == dest.name


def test_unchanged_cycle_is_not_archived_twice(tmp_path):
    cell = _cell(tmp_path)
    assert wa.archive_cell(cell, tmp_path / "arch")
    assert wa.archive_cell(cell, tmp_path / "arch") is None


def test_same_content_rewritten_later_is_not_a_new_cycle(tmp_path):
    # The rebuild copies output/ back every morning; if the forecast step
    # failed, the files come back byte-identical with a new mtime.
    cell = _cell(tmp_path, age_min=24 * 60)
    assert wa.archive_cell(cell, tmp_path / "arch")
    _cell(tmp_path, age_min=30)                     # same rows, newer mtime
    assert wa.archive_cell(cell, tmp_path / "arch") is None
    assert len(list((tmp_path / "arch" / "2026" / "zone055" / "077963").glob("*Z"))) == 1


def test_next_cycle_gets_its_own_snapshot(tmp_path):
    cell = _cell(tmp_path, age_min=24 * 60 + 30)
    first = wa.archive_cell(cell, tmp_path / "arch")
    _cell(tmp_path, rows="2026-11-08T18:00:00   -7.0\n", age_min=30)    # next day's rebuild
    second = wa.archive_cell(cell, tmp_path / "arch")
    assert second and second != first
    assert json.loads((second / "manifest.json").read_text())["data_last"] == "2026-11-08T18:00:00"
    assert (first / "0779632_res.smet").read_text() != (second / "0779632_res.smet").read_text()


def test_a_rebuild_in_progress_is_left_alone(tmp_path):
    cell = _cell(tmp_path, age_min=2)
    assert wa.archive_cell(cell, tmp_path / "arch", min_age_min=10) is None
    assert not (tmp_path / "arch").exists()


def test_cell_from_toml_takes_the_newest_season(tmp_path):
    _cell(tmp_path, season="2025")
    newest = _cell(tmp_path, season="2026")
    (tmp_path / "fcst" / "2024-case").mkdir()           # not a season folder
    toml = tmp_path / "slope.toml"
    toml.write_text(f'[wrf]\nfcst_base_dir = "{tmp_path / "fcst"}"\n'
                    f'wrf_zone = "zone055"\nwrf_grid_id = "077963"\n')
    assert wa.cell_dir_from_toml(toml) == newest


def test_missing_cell_is_an_error_not_a_silent_skip(tmp_path):
    (tmp_path / "fcst" / "2026").mkdir(parents=True)
    toml = tmp_path / "slope.toml"
    toml.write_text(f'[wrf]\nfcst_base_dir = "{tmp_path / "fcst"}"\n'
                    f'wrf_zone = "zone055"\nwrf_grid_id = "077963"\n')
    assert wa.main(["--toml", str(toml), "--archive-dir", str(tmp_path / "a")]) == 1


def test_operational_toml_names_a_cell():
    toml = Path(__file__).resolve().parents[1] / "slopes" / "Professor_PTC" / "slope_config.toml"
    import tomllib
    wrf = tomllib.loads(toml.read_text())["wrf"]
    assert wrf["fcst_base_dir"] and "wrf_zone" in wrf and "wrf_grid_id" in wrf
