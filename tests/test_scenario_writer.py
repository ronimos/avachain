"""
tests/test_scenario_writer.py — Tests for scenario_writer.py.

Run with:  pytest tests/test_scenario_writer.py -v

Key risks:
  write_asc      : yllcorner = transform.f + nrows * transform.e
                   transform.e is negative; a sign error silently flips the grid
  write_scenario_weights : must normalise to sum 1.0 for com1DFA probabilities
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import Affine

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from scenario_writer import (
    write_asc,
    write_scenario_weights,
    write_summary_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_asc_header(path: Path) -> dict:
    """Parse key/value pairs from ASC header (first 6 lines)."""
    header = {}
    with open(path) as f:
        for _ in range(6):
            line = f.readline().split()
            header[line[0].lower()] = float(line[1])
    return header


def _read_asc_data(path: Path, nrows: int, ncols: int, nodata: float) -> np.ndarray:
    """Read data rows from ASC file into float array; nodata → NaN."""
    rows = []
    with open(path) as f:
        for _ in range(6):          # skip header
            f.readline()
        for _ in range(nrows):
            vals = [float(v) for v in f.readline().split()]
            rows.append(vals)
    arr = np.array(rows)
    arr[arr == nodata] = float("nan")
    return arr


def _make_transform(xllcorner: float, yllcorner: float,
                    cellsize: float, nrows: int) -> Affine:
    """Build a north-up Affine for a grid whose lower-left is (xllcorner, yllcorner)."""
    return Affine(cellsize, 0.0, xllcorner,
                  0.0, -cellsize, yllcorner + nrows * cellsize)


# ---------------------------------------------------------------------------
# write_asc — header geometry
# ---------------------------------------------------------------------------

class TestWriteAsc:
    def test_ncols_header_correct(self, tmp_path):
        arr = np.ones((4, 5), dtype=float)
        t = _make_transform(500000, 4400000, 1.0, 4)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        h = _parse_asc_header(p)
        assert h["ncols"] == 5

    def test_nrows_header_correct(self, tmp_path):
        arr = np.ones((4, 5), dtype=float)
        t = _make_transform(500000, 4400000, 1.0, 4)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        h = _parse_asc_header(p)
        assert h["nrows"] == 4

    def test_cellsize_header_correct(self, tmp_path):
        arr = np.ones((3, 3), dtype=float)
        t = _make_transform(500000, 4400000, 2.5, 3)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        h = _parse_asc_header(p)
        assert h["cellsize"] == pytest.approx(2.5)

    def test_xllcorner_correct(self, tmp_path):
        arr = np.ones((3, 3), dtype=float)
        t = _make_transform(512345.0, 4400000.0, 1.0, 3)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        h = _parse_asc_header(p)
        assert h["xllcorner"] == pytest.approx(512345.0)

    def test_yllcorner_correct(self, tmp_path):
        """Core geometry test: yllcorner must be lower-left, not upper-left."""
        nrows, cellsize = 3, 1.0
        yllcorner_expected = 4400000.0
        arr = np.ones((nrows, 3), dtype=float)
        t = _make_transform(500000.0, yllcorner_expected, cellsize, nrows)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        h = _parse_asc_header(p)
        assert h["yllcorner"] == pytest.approx(yllcorner_expected)

    def test_yllcorner_not_upper_left(self, tmp_path):
        """Regression: wrong sign would put yllcorner above the grid."""
        nrows, cellsize = 10, 1.0
        yllcorner_expected = 4400000.0
        arr = np.ones((nrows, 5), dtype=float)
        t = _make_transform(500000.0, yllcorner_expected, cellsize, nrows)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        h = _parse_asc_header(p)
        upper_left_y = t.f
        assert h["yllcorner"] < upper_left_y, \
            "yllcorner must be BELOW upper-left (transform.f)"

    def test_nan_values_written_as_nodata(self, tmp_path):
        arr = np.array([[1.0, float("nan")],
                        [float("nan"), 2.0]])
        t = _make_transform(500000, 4400000, 1.0, 2)
        p = tmp_path / "test.asc"
        nodata = -9999.0
        write_asc(arr, t, p, nodata=nodata)
        data = _read_asc_data(p, 2, 2, nodata)
        assert math.isnan(data[0, 1])
        assert math.isnan(data[1, 0])

    def test_finite_values_preserved(self, tmp_path):
        arr = np.array([[1.2345, 6.789],
                        [0.001,  100.0]])
        t = _make_transform(500000, 4400000, 1.0, 2)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        data = _read_asc_data(p, 2, 2, -9999.0)
        assert data[0, 0] == pytest.approx(1.2345, abs=1e-4)
        assert data[1, 1] == pytest.approx(100.0, abs=1e-4)

    def test_all_nan_array(self, tmp_path):
        arr = np.full((2, 2), float("nan"))
        t = _make_transform(500000, 4400000, 1.0, 2)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        data = _read_asc_data(p, 2, 2, -9999.0)
        assert np.all(np.isnan(data))

    def test_round_trip_5x5(self, tmp_path):
        """Values survive write → parse within formatting precision."""
        rng = np.random.default_rng(42)
        arr = rng.uniform(0.0, 3.0, size=(5, 5)).astype(float)
        t = _make_transform(500000, 4400000, 1.0, 5)
        p = tmp_path / "test.asc"
        write_asc(arr, t, p)
        data = _read_asc_data(p, 5, 5, -9999.0)
        np.testing.assert_allclose(data, arr, atol=1e-4)


# ---------------------------------------------------------------------------
# write_scenario_weights — normalisation
# ---------------------------------------------------------------------------

class TestWriteScenarioWeights:
    def test_already_normalised_written_unchanged(self, tmp_path):
        weights = {"scenario_001": 0.4, "scenario_002": 0.6}
        p = tmp_path / "weights.json"
        write_scenario_weights(weights, p)
        loaded = json.loads(p.read_text())
        assert sum(loaded.values()) == pytest.approx(1.0, abs=1e-9)

    def test_unnormalised_weights_sum_to_one(self, tmp_path):
        weights = {"s1": 2.0, "s2": 3.0, "s3": 5.0}
        p = tmp_path / "weights.json"
        write_scenario_weights(weights, p)
        loaded = json.loads(p.read_text())
        assert sum(loaded.values()) == pytest.approx(1.0, abs=1e-9)

    def test_normalised_values_correct(self, tmp_path):
        weights = {"s1": 1.0, "s2": 1.0, "s3": 2.0}
        p = tmp_path / "weights.json"
        write_scenario_weights(weights, p)
        loaded = json.loads(p.read_text())
        assert loaded["s1"] == pytest.approx(0.25)
        assert loaded["s2"] == pytest.approx(0.25)
        assert loaded["s3"] == pytest.approx(0.50)

    def test_all_keys_preserved(self, tmp_path):
        weights = {"scenario_001": 0.3, "scenario_002": 0.3, "scenario_003": 0.4}
        p = tmp_path / "weights.json"
        write_scenario_weights(weights, p)
        loaded = json.loads(p.read_text())
        assert set(loaded.keys()) == set(weights.keys())

    def test_single_scenario_weight_is_one(self, tmp_path):
        weights = {"scenario_001": 5.0}
        p = tmp_path / "weights.json"
        write_scenario_weights(weights, p)
        loaded = json.loads(p.read_text())
        assert loaded["scenario_001"] == pytest.approx(1.0)

    def test_many_equal_weights_sum_to_one(self, tmp_path):
        """Floating-point accumulation test: 15 equal weights."""
        n = 15
        weights = {f"scenario_{i:03d}": 1.0 for i in range(1, n + 1)}
        p = tmp_path / "weights.json"
        write_scenario_weights(weights, p)
        loaded = json.loads(p.read_text())
        assert abs(sum(loaded.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# write_summary_csv — column ordering and NaN fill
# ---------------------------------------------------------------------------

class TestWriteSummaryCsv:
    def test_expected_columns_present(self, tmp_path):
        import pandas as pd
        rows = [{
            "scenario_id": "scenario_001",
            "trigger_cluster": 42,
            "A_ca_m": 12.5,
            "size_factor": 1.0,
            "depth_percentile": 50,
            "release_area_m2": 5000.0,
            "mean_depth_m": 0.8,
            "total_volume_m3": 4000.0,
            "weight": 0.5,
        }]
        p = tmp_path / "summary.csv"
        write_summary_csv(rows, p)
        df = pd.read_csv(p)
        for col in ["scenario_id", "trigger_cluster", "A_ca_m", "weight"]:
            assert col in df.columns

    def test_optional_columns_filled_with_nan(self, tmp_path):
        import pandas as pd
        rows = [{"scenario_id": "s1", "trigger_cluster": 1, "A_ca_m": 10.0,
                 "size_factor": 1.0, "depth_percentile": 50,
                 "release_area_m2": 100.0, "mean_depth_m": 0.5,
                 "total_volume_m3": 50.0, "weight": 1.0}]
        p = tmp_path / "summary.csv"
        write_summary_csv(rows, p)
        df = pd.read_csv(p)
        # scour_depth_m was not in the row; should exist as NaN
        assert "scour_depth_m" in df.columns
        assert pd.isna(df["scour_depth_m"].iloc[0])

    def test_row_count_matches(self, tmp_path):
        import pandas as pd
        rows = [{"scenario_id": f"s{i}", "trigger_cluster": i,
                 "A_ca_m": float(i), "size_factor": 1.0,
                 "depth_percentile": 50, "release_area_m2": 100.0,
                 "mean_depth_m": 0.5, "total_volume_m3": 50.0,
                 "weight": 1.0 / 3} for i in range(3)]
        p = tmp_path / "summary.csv"
        write_summary_csv(rows, p)
        df = pd.read_csv(p)
        assert len(df) == 3
