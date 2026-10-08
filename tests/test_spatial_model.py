"""
tests/test_spatial_model.py — Unit tests for gap-fill functions in spatial_model.py.

Run with:  pytest tests/test_spatial_model.py -v

Covers:
  gap_fill_station_only  — baseline scalar-only fill
  gap_fill_period        — transport-field + wind²-weighted fill
  validate_prediction    — endpoint validation metrics
"""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import numpy as np
import pandas as pd
import pytest

from spatial_model import gap_fill_period, gap_fill_station_only, validate_prediction


_IDX = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")


def _grid(values) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _mask_all() -> np.ndarray:
    return np.ones((2, 2), dtype=bool)


# ---------------------------------------------------------------------------
# gap_fill_station_only
# ---------------------------------------------------------------------------

class TestGapFillStationOnly:
    def test_snowfall_adds_uniformly_to_all_cells(self):
        hs = _grid([[1.0, 0.5], [0.8, 1.2]])
        stn = pd.Series([0.50, 0.60], index=_IDX[:2])  # +0.10
        grids = gap_fill_station_only(hs, stn, _mask_all())
        np.testing.assert_allclose(grids[_IDX[1]], hs + 0.10, atol=1e-9)

    def test_settlement_scales_all_cells_by_station_ratio(self):
        hs = _grid([[2.0, 0.5], [1.0, 1.5]])
        stn = pd.Series([0.60, 0.54], index=_IDX[:2])  # ratio 0.9
        grids = gap_fill_station_only(hs, stn, _mask_all())
        np.testing.assert_allclose(grids[_IDX[1]], hs * 0.9, atol=1e-9)

    def test_deep_shallow_ratio_preserved_after_settlement(self):
        hs = _grid([[2.0, 0.4], [0.4, 2.0]])
        stn = pd.Series([0.60, 0.54], index=_IDX[:2])  # ratio 0.9
        grids = gap_fill_station_only(hs, stn, _mask_all())
        result = grids[_IDX[1]]
        # Deep cells (2.0) should still be 5× the shallow cells (0.4)
        assert result[0, 0] / result[0, 1] == pytest.approx(2.0 / 0.4, rel=1e-6)

    def test_no_cell_goes_negative(self):
        hs = _grid([[0.01, 0.0], [0.05, 0.02]])
        stn = pd.Series([1.00, 0.001], index=_IDX[:2])  # extreme settlement
        grids = gap_fill_station_only(hs, stn, _mask_all())
        for arr in grids.values():
            assert np.all(arr >= 0)

    def test_out_of_mask_cells_frozen_on_snowfall(self):
        hs = _grid([[1.0, 0.5], [0.8, 1.2]])
        mask = np.array([[True, True], [False, True]])
        stn = pd.Series([0.50, 0.60], index=_IDX[:2])  # +0.10
        grids = gap_fill_station_only(hs, stn, mask)
        # Out-of-mask [1,0]: unchanged
        assert grids[_IDX[1]][1, 0] == pytest.approx(hs[1, 0])

    def test_first_grid_equals_hs_start(self):
        hs = _grid([[1.0, 0.5], [0.8, 1.2]])
        stn = pd.Series([0.50, 0.60], index=_IDX[:2])
        grids = gap_fill_station_only(hs, stn, _mask_all())
        np.testing.assert_array_equal(grids[_IDX[0]], hs)

    def test_output_keys_match_stn_hs_index(self):
        hs = _grid([[1.0, 0.5], [0.8, 1.2]])
        stn = pd.Series([0.50, 0.55, 0.60, 0.58], index=_IDX[:4])
        grids = gap_fill_station_only(hs, stn, _mask_all())
        assert set(grids.keys()) == set(stn.index)

    def test_multi_step_snowfall_accumulates(self):
        hs = _grid([[1.0, 1.0], [1.0, 1.0]])
        stn = pd.Series([0.50, 0.55, 0.60], index=_IDX[:3])  # +0.05 each step
        grids = gap_fill_station_only(hs, stn, _mask_all())
        np.testing.assert_allclose(grids[_IDX[2]], hs + 0.10, atol=1e-9)


# ---------------------------------------------------------------------------
# gap_fill_period
# ---------------------------------------------------------------------------

class TestGapFillPeriod:
    def _flat_stn(self, n, val=0.50):
        return pd.Series([val] * n, index=_IDX[:n])

    def test_transport_conservation_flat_station_zero_anchor_wind(self):
        # Anchor ws=0 so total_we == sum of transport-period ws² → fracs sum to 1
        hs = _grid([[1.0, 1.0], [1.0, 1.0]])
        transport = _grid([[0.10, 0.05], [0.20, 0.15]])
        stn = self._flat_stn(4)  # no station change
        ws = pd.Series([0.0, 2.0, 4.0, 1.0], index=_IDX[:4])
        grids = gap_fill_period(hs, transport, stn, ws, _mask_all())
        # Net change over period should equal transport_field exactly
        net_change = grids[_IDX[3]] - hs
        np.testing.assert_allclose(net_change, transport, atol=1e-9)

    def test_higher_wind_hour_gets_proportionally_more_transport(self):
        # Two transport hours with ws ratio 1:2 → ws² ratio 1:4
        hs = np.ones((1, 1))
        transport = np.array([[1.0]])
        stn = self._flat_stn(3)
        # ws2 = [0, 1, 4], total = 5 → fracs 1/5 and 4/5
        ws = pd.Series([0.0, 1.0, 2.0], index=_IDX[:3])
        grids = gap_fill_period(hs, transport, stn, ws, np.ones((1, 1), dtype=bool))
        assert grids[_IDX[1]][0, 0] == pytest.approx(1.20, abs=1e-9)
        assert grids[_IDX[2]][0, 0] == pytest.approx(2.00, abs=1e-9)

    def test_zero_wind_energy_no_transport_applied(self):
        hs = np.ones((2, 2))
        transport = np.full((2, 2), 0.5)
        stn = self._flat_stn(3)
        ws = pd.Series([0.0, 0.0, 0.0], index=_IDX[:3])
        grids = gap_fill_period(hs, transport, stn, ws, _mask_all())
        # frac=0 every step → no transport → HS unchanged
        np.testing.assert_allclose(grids[_IDX[2]], np.ones((2, 2)), atol=1e-9)

    def test_station_delta_applied_independent_of_transport(self):
        hs = np.ones((1, 1))
        transport = np.zeros((1, 1))
        stn = pd.Series([0.50, 0.60, 0.70], index=_IDX[:3])  # +0.10 each step
        ws = pd.Series([1.0, 1.0, 1.0], index=_IDX[:3])
        grids = gap_fill_period(hs, transport, stn, ws, np.ones((1, 1), dtype=bool))
        assert grids[_IDX[2]][0, 0] == pytest.approx(1.20, abs=1e-9)

    def test_no_negative_values_after_strong_erosion(self):
        hs = np.full((2, 2), 0.05)
        transport = np.full((2, 2), -0.30)  # exceeds hs_start
        stn = self._flat_stn(4)
        ws = pd.Series([0.0, 3.0, 3.0, 3.0], index=_IDX[:4])
        grids = gap_fill_period(hs, transport, stn, ws, _mask_all())
        for arr in grids.values():
            assert np.all(arr >= 0)

    def test_first_grid_equals_hs_start(self):
        hs = _grid([[1.0, 0.5], [0.8, 1.2]])
        transport = _grid([[0.1, 0.0], [0.0, -0.1]])
        stn = self._flat_stn(4)
        ws = pd.Series([1.0, 2.0, 3.0, 1.0], index=_IDX[:4])
        grids = gap_fill_period(hs, transport, stn, ws, _mask_all())
        np.testing.assert_array_equal(grids[_IDX[0]], hs)

    def test_out_of_mask_gets_station_only_no_transport(self):
        hs = _grid([[1.0, 1.0], [1.0, 1.0]])
        transport = _grid([[0.5, 0.5], [0.5, 0.5]])
        mask = np.array([[True, False], [True, True]])
        stn = pd.Series([0.50, 0.60], index=_IDX[:2])  # +0.10
        # ws2 = [0, 4], total = 4, frac at idx=1 = 1.0
        ws = pd.Series([0.0, 2.0], index=_IDX[:2])
        grids = gap_fill_period(hs, transport, stn, ws, mask)
        result = grids[_IDX[1]]
        # Out-of-mask [0,1]: station dhs only (+0.10), no transport
        assert result[0, 1] == pytest.approx(1.10, abs=1e-9)
        # In-mask [0,0]: station dhs + transport (0.10 + 0.50 = 0.60 added)
        assert result[0, 0] == pytest.approx(1.60, abs=1e-9)

    def test_nan_transport_treated_as_zero(self):
        hs = np.ones((2, 2))
        transport = _grid([[np.nan, 0.1], [0.2, np.nan]])
        stn = self._flat_stn(3)
        ws = pd.Series([0.0, 2.0, 2.0], index=_IDX[:3])
        # Should not raise; NaN cells in transport_field treated as 0
        grids = gap_fill_period(hs, transport, stn, ws, _mask_all())
        result = grids[_IDX[2]]
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# validate_prediction
# ---------------------------------------------------------------------------

class TestValidatePrediction:
    def test_perfect_prediction_zero_error(self):
        obs = _grid([[1.0, 2.0], [3.0, 4.0]])
        mask = _mask_all()
        r = validate_prediction(obs, obs, mask)
        assert r['rmse'] == pytest.approx(0.0, abs=1e-9)
        assert r['bias'] == pytest.approx(0.0, abs=1e-9)
        assert r['r'] == pytest.approx(1.0, abs=1e-9)
        assert r['n'] == 4

    def test_known_rmse_and_bias(self):
        pred = np.array([[0.0, 0.0]])
        obs = np.array([[3.0, 4.0]])
        mask = np.ones((1, 2), dtype=bool)
        with pytest.warns(RuntimeWarning, match="invalid value"):
            r = validate_prediction(pred, obs, mask)
        # residuals [3, 4]: rmse = sqrt((9+16)/2) = sqrt(12.5)
        assert r['rmse'] == pytest.approx(np.sqrt(12.5), rel=1e-6)
        assert r['bias'] == pytest.approx(3.5)   # mean(obs - pred) = (3+4)/2

    def test_bias_sign_under_prediction_is_positive(self):
        # predicted too low → observed > predicted → positive bias
        pred = np.array([[1.0, 1.0]])
        obs = np.array([[2.0, 2.0]])
        mask = np.ones((1, 2), dtype=bool)
        with pytest.warns(RuntimeWarning, match="invalid value"):
            r = validate_prediction(pred, obs, mask)
        assert r['bias'] > 0

    def test_empty_mask_returns_nan_stats_and_zero_n(self):
        pred = _grid([[1.0, 2.0], [3.0, 4.0]])
        obs = _grid([[1.0, 2.0], [3.0, 4.0]])
        mask = np.zeros((2, 2), dtype=bool)
        r = validate_prediction(pred, obs, mask)
        assert r['n'] == 0
        assert np.isnan(r['rmse'])
        assert np.isnan(r['bias'])
        assert np.isnan(r['mae'])

    def test_nan_cells_excluded_from_metrics(self):
        pred = _grid([[1.0, np.nan], [3.0, 4.0]])
        obs = _grid([[1.0, 2.0], [3.0, 4.0]])
        mask = _mask_all()
        r = validate_prediction(pred, obs, mask)
        # NaN in predicted should be excluded; remaining 3 pairs are perfect
        assert r['rmse'] == pytest.approx(0.0, abs=1e-9)
        assert r['n'] == 3
