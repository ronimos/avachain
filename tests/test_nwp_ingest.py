"""
tests/test_nwp_ingest.py — Unit and integration tests for nwp_ingest.py.

Run with:  pytest tests/test_nwp_ingest.py -v
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from nwp_ingest import (
    NWP_FLAG,
    LAPSE_RATE_K_PER_M,
    _apply_lapse_rate,
    _format_nwp_row,
    _resample_to_hourly,
    _truncate_smet_to,
    append_nwp_rows,
    extend_all_smets,
    get_stable_ts,
    prune_nwp_rows,
    read_wrf_smet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WRF_HEADER = textwrap.dedent("""\
    SMET 1.1 ASCII
    [HEADER]
    station_id       = WRF_GRID_1
    altitude         = 3200.0
    fields           = timestamp TA RH VW DW ISWR ILWR MS_Snow
    [DATA]
""")

CLUSTER_HEADER = textwrap.dedent("""\
    SMET 1.1 ASCII
    [HEADER]
    station_id       = cluster_0001
    altitude         = 3000.0
    fields           = timestamp TA RH VW DW ISWR ILWR MS_Snow HS
    [DATA]
""")


def _make_wrf_smet(tmp_path: Path, rows: list[tuple]) -> Path:
    """Write a minimal WRF SMET with the given (timestamp_str, TA, RH, VW, DW, ISWR, ILWR, MS_Snow) rows."""
    p = tmp_path / "wrf_grid.smet"
    lines = [WRF_HEADER]
    for r in rows:
        ts, ta, rh, vw, dw, iswr, ilwr, ms = r
        lines.append(f"{ts}\t{ta}\t{rh}\t{vw}\t{dw}\t{iswr}\t{ilwr}\t{ms}\n")
    p.write_text("".join(lines))
    return p


def _make_cluster_smet(tmp_path: Path,
                       name: str,
                       altitude: float,
                       rows: list[tuple]) -> Path:
    """Write a cluster SMET to tmp_path/name.smet."""
    header = textwrap.dedent(f"""\
        SMET 1.1 ASCII
        [HEADER]
        station_id       = {name}
        altitude         = {altitude}
        fields           = timestamp TA RH VW DW ISWR ILWR MS_Snow HS
        [DATA]
    """)
    p = tmp_path / f"{name}.smet"
    lines = [header]
    for r in rows:
        ts, ta, rh, vw, dw, iswr, ilwr, ms, hs = r
        lines.append(f"{ts}\t{ta}\t{rh}\t{vw}\t{dw}\t{iswr}\t{ilwr}\t{ms}\t{hs}\n")
    p.write_text("".join(lines))
    return p


def _ts(dt_str: str) -> pd.Timestamp:
    return pd.Timestamp(dt_str, tz="UTC")


# ---------------------------------------------------------------------------
# get_stable_ts
# ---------------------------------------------------------------------------

class TestGetStableTs:
    def test_returns_last_past_timestamp(self, tmp_path):
        past1 = "2020-01-01T00:00"
        past2 = "2020-01-01T06:00"
        future = "2099-12-31T00:00"
        p = _make_wrf_smet(tmp_path, [
            (past1,  -5.0, 80, 3.0, 270, 0, 250, 0),
            (past2,  -4.0, 78, 2.5, 260, 0, 248, 0),
            (future, -3.0, 75, 2.0, 250, 50, 245, 0),
        ])
        ts = get_stable_ts(p)
        assert ts == _ts(past2)

    def test_all_future_returns_none(self, tmp_path):
        p = _make_wrf_smet(tmp_path, [
            ("2099-01-01T00:00", -5, 80, 3, 270, 0, 250, 0),
        ])
        assert get_stable_ts(p) is None

    def test_empty_data_section_returns_none(self, tmp_path):
        p = tmp_path / "empty.smet"
        p.write_text(WRF_HEADER)
        assert get_stable_ts(p) is None


# ---------------------------------------------------------------------------
# read_wrf_smet
# ---------------------------------------------------------------------------

class TestReadWrfSmet:
    def _make_file(self, tmp_path):
        rows = [
            ("2020-01-01T00:00", -5.0, 80.0, 3.0, 270.0, 0.0, 250.0, 0.0),
            ("2020-01-01T06:00", -4.0, 78.0, 2.5, 260.0, 20.0, 248.0, 1.0),
            ("2020-01-01T12:00", -3.0, 75.0, 2.0, 250.0, 200.0, 245.0, 0.5),
        ]
        return _make_wrf_smet(tmp_path, rows)

    def test_altitude_parsed(self, tmp_path):
        p = self._make_file(tmp_path)
        _, alt = read_wrf_smet(p, _ts("2020-01-01T00:00"), _ts("2020-01-01T12:00"))
        assert alt == 3200.0

    def test_rows_in_range(self, tmp_path):
        p = self._make_file(tmp_path)
        df, _ = read_wrf_smet(p, _ts("2020-01-01T06:00"), _ts("2020-01-01T12:00"))
        assert len(df) == 2
        assert _ts("2020-01-01T06:00") in df.index
        assert _ts("2020-01-01T12:00") in df.index
        assert _ts("2020-01-01T00:00") not in df.index

    def test_nodata_mapped_to_nan(self, tmp_path):
        rows = [("2020-01-01T00:00", -999.0, 80.0, 3.0, 270.0, 0.0, 250.0, 0.0)]
        p = _make_wrf_smet(tmp_path, rows)
        df, _ = read_wrf_smet(p, _ts("2020-01-01T00:00"), _ts("2020-01-01T06:00"))
        assert math.isnan(df.loc[_ts("2020-01-01T00:00"), "TA"])

    def test_empty_dataframe_on_no_rows_in_range(self, tmp_path):
        p = self._make_file(tmp_path)
        df, _ = read_wrf_smet(p, _ts("2025-01-01T00:00"), _ts("2025-01-02T00:00"))
        assert df.empty


# ---------------------------------------------------------------------------
# _resample_to_hourly
# ---------------------------------------------------------------------------

class TestResampleToHourly:
    def _make_6h_df(self):
        idx = pd.date_range("2020-01-01T00:00", periods=9, freq="6h", tz="UTC")
        data = {
            "TA":     [0.0,  6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0],
            "RH":     [80.0]*9,
            "VW":     [2.0]*9,
            "DW":     [180.0]*9,
            "ISWR":   [0.0]*9,
            "ILWR":   [250.0]*9,
            "MS_Snow":[0.1]*9,
        }
        return pd.DataFrame(data, index=idx)

    def test_output_row_count(self):
        df = self._make_6h_df()
        out = _resample_to_hourly(df)
        # 8 × 6h intervals = 48h → 49 rows (inclusive on both ends)
        assert len(out) == 49

    def test_boundary_values_preserved(self):
        df = self._make_6h_df()
        out = _resample_to_hourly(df)
        assert out.loc[_ts("2020-01-01T00:00"), "TA"] == pytest.approx(0.0)
        assert out.loc[_ts("2020-01-01T06:00"), "TA"] == pytest.approx(6.0)
        # 9 periods × 6h = 48h from Jan-01 → last timestamp is Jan-03T00:00 (TA=48)
        assert out.loc[_ts("2020-01-03T00:00"), "TA"] == pytest.approx(48.0)

    def test_intermediate_ta_interpolated(self):
        df = self._make_6h_df()
        out = _resample_to_hourly(df)
        # Between 0h (TA=0) and 6h (TA=6), 3h should be ~3
        assert out.loc[_ts("2020-01-01T03:00"), "TA"] == pytest.approx(3.0, abs=1e-6)

    def test_dw_circular_crossing_zero(self):
        idx = pd.date_range("2020-01-01T00:00", periods=3, freq="6h", tz="UTC")
        df = pd.DataFrame({
            "TA": [0.0]*3, "RH": [80.0]*3, "VW": [2.0]*3,
            "DW": [350.0, 10.0, 30.0],
            "ISWR": [0.0]*3, "ILWR": [250.0]*3, "MS_Snow": [0.0]*3,
        }, index=idx)
        out = _resample_to_hourly(df)
        # At 3h (midpoint between 350° and 10°) the circular mean should be ~0° (or ~360°)
        mid_dw = out.loc[_ts("2020-01-01T03:00"), "DW"]
        # Should go through 0/360, not through 180
        assert mid_dw < 20 or mid_dw > 340, f"DW at midpoint = {mid_dw}, expected near 0°"

    def test_ms_snow_step_hold(self):
        idx = pd.date_range("2020-01-01T00:00", periods=3, freq="6h", tz="UTC")
        df = pd.DataFrame({
            "TA": [0.0]*3, "RH": [80.0]*3, "VW": [2.0]*3, "DW": [180.0]*3,
            "ISWR": [0.0]*3, "ILWR": [250.0]*3,
            "MS_Snow": [0.2, 0.8, 0.0],
        }, index=idx)
        out = _resample_to_hourly(df)
        # Hours 1–5 should all have MS_Snow = 0.2 (ffill from 0h)
        for h in range(1, 6):
            ts = pd.Timestamp(f"2020-01-01T{h:02d}:00", tz="UTC")
            assert out.loc[ts, "MS_Snow"] == pytest.approx(0.2), \
                f"MS_Snow at T+{h}h should be 0.2 (step-hold)"

    def test_physical_clip_iswr(self):
        idx = pd.date_range("2020-01-01T00:00", periods=2, freq="6h", tz="UTC")
        df = pd.DataFrame({
            "TA": [0.0]*2, "RH": [80.0]*2, "VW": [2.0]*2, "DW": [180.0]*2,
            "ISWR": [-50.0, -10.0], "ILWR": [250.0]*2, "MS_Snow": [0.0]*2,
        }, index=idx)
        out = _resample_to_hourly(df)
        assert (out["ISWR"] >= 0.0).all()

    def test_physical_clip_rh(self):
        idx = pd.date_range("2020-01-01T00:00", periods=2, freq="6h", tz="UTC")
        df = pd.DataFrame({
            "TA": [0.0]*2, "RH": [110.0, -5.0], "VW": [2.0]*2, "DW": [180.0]*2,
            "ISWR": [0.0]*2, "ILWR": [250.0]*2, "MS_Snow": [0.0]*2,
        }, index=idx)
        out = _resample_to_hourly(df)
        assert out["RH"].max() <= 100.0
        assert out["RH"].min() >= 0.0

    def test_physical_clip_vw(self):
        idx = pd.date_range("2020-01-01T00:00", periods=2, freq="6h", tz="UTC")
        df = pd.DataFrame({
            "TA": [0.0]*2, "RH": [80.0]*2, "VW": [-1.0, -5.0], "DW": [180.0]*2,
            "ISWR": [0.0]*2, "ILWR": [250.0]*2, "MS_Snow": [0.0]*2,
        }, index=idx)
        out = _resample_to_hourly(df)
        assert (out["VW"] >= 0.0).all()

    def test_empty_df_returns_empty(self):
        out = _resample_to_hourly(pd.DataFrame())
        assert out.empty


# ---------------------------------------------------------------------------
# _apply_lapse_rate
# ---------------------------------------------------------------------------

class TestApplyLapseRate:
    def _simple_df(self, ta=0.0):
        idx = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
        return pd.DataFrame({"TA": [ta]*3}, index=idx)

    def test_cluster_above_wrf_cools(self):
        df = self._simple_df(ta=0.0)
        out = _apply_lapse_rate(df, wrf_alt_m=3000.0, cluster_alt_m=3200.0)
        expected_delta = LAPSE_RATE_K_PER_M * 200.0  # -1.3 K
        assert out["TA"].iloc[0] == pytest.approx(0.0 + expected_delta, abs=1e-9)

    def test_cluster_below_wrf_warms(self):
        df = self._simple_df(ta=0.0)
        out = _apply_lapse_rate(df, wrf_alt_m=3200.0, cluster_alt_m=3000.0)
        expected_delta = LAPSE_RATE_K_PER_M * (-200.0)  # +1.3 K
        assert out["TA"].iloc[0] == pytest.approx(0.0 + expected_delta, abs=1e-9)

    def test_nan_wrf_alt_returns_unchanged(self):
        df = self._simple_df(ta=5.0)
        out = _apply_lapse_rate(df, wrf_alt_m=float("nan"), cluster_alt_m=3000.0)
        assert (out["TA"] == 5.0).all()

    def test_nan_cluster_alt_returns_unchanged(self):
        df = self._simple_df(ta=5.0)
        out = _apply_lapse_rate(df, wrf_alt_m=3000.0, cluster_alt_m=float("nan"))
        assert (out["TA"] == 5.0).all()

    def test_no_ta_column_returns_unchanged(self):
        idx = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({"RH": [80.0, 80.0]}, index=idx)
        out = _apply_lapse_rate(df, wrf_alt_m=3000.0, cluster_alt_m=3200.0)
        assert list(out.columns) == ["RH"]


# ---------------------------------------------------------------------------
# _truncate_smet_to
# ---------------------------------------------------------------------------

class TestTruncateSmetTo:
    def _make_smet(self, tmp_path, rows: list[str]) -> Path:
        """rows: list of ISO timestamp strings (past) or 'nwp_flag'."""
        p = tmp_path / "cluster_test.smet"
        lines = [
            "SMET 1.1 ASCII\n",
            "[HEADER]\n",
            "station_id = test\n",
            "altitude   = 3000.0\n",
            "fields     = timestamp TA HS\n",
            "[DATA]\n",
        ]
        for r in rows:
            if r == "nwp_flag":
                lines.append(f"{NWP_FLAG}\n")
            else:
                lines.append(f"{r}\t-5.0\t0.5\n")
        p.write_text("".join(lines))
        return p

    def test_keeps_rows_at_or_before_ts(self, tmp_path):
        p = self._make_smet(tmp_path, [
            "2020-01-01T00:00",
            "2020-01-01T01:00",
            "2020-01-01T02:00",
        ])
        removed = _truncate_smet_to(p, _ts("2020-01-01T01:00"))
        assert removed == 1
        text = p.read_text()
        assert "2020-01-01T00:00" in text
        assert "2020-01-01T01:00" in text
        assert "2020-01-01T02:00" not in text

    def test_strips_nwp_flag_lines(self, tmp_path):
        p = self._make_smet(tmp_path, [
            "2020-01-01T00:00",
            "nwp_flag",
            "2020-01-01T01:00",
        ])
        _truncate_smet_to(p, _ts("2020-01-01T06:00"))
        text = p.read_text()
        assert NWP_FLAG not in text

    def test_header_preserved(self, tmp_path):
        p = self._make_smet(tmp_path, ["2020-01-01T00:00"])
        _truncate_smet_to(p, _ts("2020-01-01T06:00"))
        text = p.read_text()
        assert "[HEADER]" in text
        assert "[DATA]" in text
        assert "altitude" in text

    def test_returns_correct_removed_count(self, tmp_path):
        p = self._make_smet(tmp_path, [
            "2020-01-01T00:00",
            "2020-01-01T01:00",
            "2020-01-01T02:00",
            "2020-01-01T03:00",
        ])
        n = _truncate_smet_to(p, _ts("2020-01-01T01:00"))
        assert n == 2


# ---------------------------------------------------------------------------
# _format_nwp_row
# ---------------------------------------------------------------------------

class TestFormatNwpRow:
    def _fields(self):
        return ["timestamp", "TA", "RH", "VW", "DW", "ISWR", "ILWR", "MS_Snow", "HS"]

    def test_hs_always_minus999(self):
        ts = _ts("2020-01-01T06:00")
        row = {"TA": -5.0, "RH": 80.0, "VW": 3.0, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "MS_Snow": 0.5, "HS": 1.2}
        line = _format_nwp_row(ts, row, self._fields())
        parts = line.split("\t")
        hs_idx = self._fields().index("HS")
        assert parts[hs_idx] == "-999"

    def test_ta_formatted_three_decimals(self):
        ts = _ts("2020-01-01T06:00")
        row = {"TA": -5.123456, "RH": 80.0, "VW": 3.0, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "MS_Snow": 0.5}
        line = _format_nwp_row(ts, row, self._fields())
        parts = line.split("\t")
        ta_idx = self._fields().index("TA")
        assert parts[ta_idx] == "-5.123"

    def test_vw_dw_formatted_four_decimals(self):
        ts = _ts("2020-01-01T06:00")
        row = {"TA": -5.0, "RH": 80.0, "VW": 3.12356, "DW": 270.12356,
               "ISWR": 0.0, "ILWR": 250.0, "MS_Snow": 0.5}
        line = _format_nwp_row(ts, row, self._fields())
        parts = line.split("\t")
        vw_idx = self._fields().index("VW")
        dw_idx = self._fields().index("DW")
        assert parts[vw_idx] == "3.1236"
        assert parts[dw_idx] == "270.1236"

    def test_timestamp_format(self):
        ts = _ts("2020-03-15T09:30")
        row = {"TA": 0.0, "RH": 80.0, "VW": 2.0, "DW": 180.0,
               "ISWR": 0.0, "ILWR": 250.0, "MS_Snow": 0.0}
        line = _format_nwp_row(ts, row, self._fields())
        assert line.startswith("2020-03-15T09:30")

    def test_unknown_column_written_as_minus999(self):
        fields = ["timestamp", "TA", "EXTRA_FIELD"]
        ts = _ts("2020-01-01T00:00")
        row = {"TA": -5.0}
        line = _format_nwp_row(ts, row, fields)
        parts = line.split("\t")
        assert parts[2] == "-999"

    def test_nan_value_written_as_minus999(self):
        fields = ["timestamp", "TA"]
        ts = _ts("2020-01-01T00:00")
        row = {"TA": float("nan")}
        line = _format_nwp_row(ts, row, fields)
        assert "-999" in line


# ---------------------------------------------------------------------------
# append_nwp_rows
# ---------------------------------------------------------------------------

class TestAppendNwpRows:
    def _make_cluster(self, tmp_path, last_ts_str="2020-01-01T12:00") -> Path:
        p = tmp_path / "cluster_0001.smet"
        p.write_text(textwrap.dedent(f"""\
            SMET 1.1 ASCII
            [HEADER]
            station_id = cluster_0001
            altitude   = 3000.0
            fields     = timestamp TA RH VW DW ISWR ILWR MS_Snow HS
            [DATA]
            2020-01-01T00:00\t-5.0\t80.0\t2.0\t270.0\t0.0\t250.0\t0.0\t0.5
            {last_ts_str}\t-4.0\t78.0\t2.5\t260.0\t0.0\t248.0\t0.0\t0.5
        """))
        return p

    def _make_forecast(self, start_str="2020-01-01T13:00", n=3):
        idx = pd.date_range(start_str, periods=n, freq="h", tz="UTC")
        return pd.DataFrame({
            "TA": [-3.0]*n, "RH": [76.0]*n, "VW": [2.0]*n,
            "DW": [255.0]*n, "ISWR": [0.0]*n, "ILWR": [245.0]*n,
            "MS_Snow": [0.0]*n,
        }, index=idx)

    def test_nwp_flag_written_before_rows(self, tmp_path):
        p = self._make_cluster(tmp_path)
        fc = self._make_forecast()
        append_nwp_rows(p, fc)
        text = p.read_text()
        flag_pos = text.index(NWP_FLAG)
        first_row_pos = text.index("2020-01-01T13:00")
        assert flag_pos < first_row_pos

    def test_deduplication_skips_existing_rows(self, tmp_path):
        p = self._make_cluster(tmp_path, last_ts_str="2020-01-01T12:00")
        # Forecast starts at 12:00 — same as last existing row
        fc = self._make_forecast(start_str="2020-01-01T12:00", n=5)
        n = append_nwp_rows(p, fc)
        # Only T+13h..T+16h should be appended (12:00 is deduplicated)
        assert n == 4

    def test_dry_run_writes_nothing(self, tmp_path):
        p = self._make_cluster(tmp_path)
        fc = self._make_forecast()
        before = p.read_text()
        n = append_nwp_rows(p, fc, dry_run=True)
        assert n == 0
        assert p.read_text() == before

    def test_empty_forecast_returns_zero(self, tmp_path):
        p = self._make_cluster(tmp_path)
        n = append_nwp_rows(p, pd.DataFrame())
        assert n == 0
        assert NWP_FLAG not in p.read_text()

    def test_returns_correct_count(self, tmp_path):
        p = self._make_cluster(tmp_path)
        fc = self._make_forecast(n=10)
        n = append_nwp_rows(p, fc)
        assert n == 10


# ---------------------------------------------------------------------------
# prune_nwp_rows
# ---------------------------------------------------------------------------

class TestPruneNwpRows:
    def _make_smet_with_nwp(self, tmp_path, n_obs=3, n_nwp=5) -> Path:
        p = tmp_path / "cluster_prune.smet"
        lines = [
            "SMET 1.1 ASCII\n[HEADER]\nstation_id = test\nfields = timestamp TA\n[DATA]\n"
        ]
        for i in range(n_obs):
            ts = pd.Timestamp("2020-01-01T00:00", tz="UTC") + pd.Timedelta(hours=i)
            lines.append(f"{ts.strftime('%Y-%m-%dT%H:%M')}\t-5.0\n")
        lines.append(f"{NWP_FLAG}\n")
        for i in range(n_nwp):
            ts = pd.Timestamp("2020-01-01T00:00", tz="UTC") + pd.Timedelta(hours=n_obs + i)
            lines.append(f"{ts.strftime('%Y-%m-%dT%H:%M')}\t-4.0\n")
        p.write_text("".join(lines))
        return p

    def test_removes_flag_and_nwp_rows(self, tmp_path):
        p = self._make_smet_with_nwp(tmp_path, n_obs=3, n_nwp=5)
        removed = prune_nwp_rows(p)
        assert removed == 5
        text = p.read_text()
        assert NWP_FLAG not in text

    def test_observed_rows_preserved(self, tmp_path):
        p = self._make_smet_with_nwp(tmp_path, n_obs=3, n_nwp=5)
        prune_nwp_rows(p)
        text = p.read_text()
        for i in range(3):
            ts = (pd.Timestamp("2020-01-01T00:00", tz="UTC") + pd.Timedelta(hours=i))
            assert ts.strftime("%Y-%m-%dT%H:%M") in text

    def test_no_flag_returns_zero_and_unchanged(self, tmp_path):
        p = tmp_path / "clean.smet"
        content = "SMET 1.1 ASCII\n[HEADER]\nfields=timestamp TA\n[DATA]\n2020-01-01T00:00\t-5\n"
        p.write_text(content)
        n = prune_nwp_rows(p)
        assert n == 0
        assert p.read_text() == content

    def test_returns_correct_count(self, tmp_path):
        p = self._make_smet_with_nwp(tmp_path, n_obs=2, n_nwp=8)
        assert prune_nwp_rows(p) == 8


# ---------------------------------------------------------------------------
# extend_all_smets (integration)
# ---------------------------------------------------------------------------

class TestExtendAllSmets:
    def _make_wrf_hourly(self, stable_ts: pd.Timestamp, n_hours: int = 48):
        idx = pd.date_range(
            stable_ts + pd.Timedelta(hours=1), periods=n_hours, freq="h", tz="UTC"
        )
        return pd.DataFrame({
            "TA": [0.0]*n_hours, "RH": [80.0]*n_hours, "VW": [2.0]*n_hours,
            "DW": [180.0]*n_hours, "ISWR": [0.0]*n_hours, "ILWR": [250.0]*n_hours,
            "MS_Snow": [0.0]*n_hours,
        }, index=idx)

    def test_lapse_rate_applied_per_cluster(self, tmp_path):
        wrf_alt = 3200.0
        stable_ts = _ts("2020-01-01T12:00")
        obs_row = ("2020-01-01T12:00", 0.0, 80.0, 2.0, 180.0, 0.0, 250.0, 0.0, 0.5)

        low = _make_cluster_smet(tmp_path, "cluster_0001", 3000.0, [obs_row])
        high = _make_cluster_smet(tmp_path, "cluster_0002", 3400.0, [obs_row])

        wrf_fc = self._make_wrf_hourly(stable_ts)
        extend_all_smets(tmp_path, wrf_fc, wrf_alt, stable_ts)

        # Read back the first appended TA from each file
        def first_nwp_ta(p: Path) -> float:
            in_nwp = False
            with open(p) as f:
                for line in f:
                    if NWP_FLAG in line:
                        in_nwp = True
                        continue
                    if in_nwp and line.strip() and not line.startswith("#"):
                        return float(line.split()[1])
            return float("nan")

        ta_low = first_nwp_ta(low)
        ta_high = first_nwp_ta(high)
        # low cluster is 200m below WRF → warmer (+1.3 K); high is 200m above → cooler (-1.3 K)
        assert ta_low > ta_high
        # total spread = lapse correction at 3000 minus correction at 3400
        expected_spread = LAPSE_RATE_K_PER_M * ((3000 - 3200) - (3400 - 3200))  # +2.6 K
        assert abs((ta_low - ta_high) - expected_spread) < 0.01

    def test_nwp_flag_in_each_file(self, tmp_path):
        stable_ts = _ts("2020-01-01T12:00")
        obs = ("2020-01-01T12:00", 0.0, 80.0, 2.0, 180.0, 0.0, 250.0, 0.0, 0.5)
        for name in ["cluster_0001", "cluster_0002", "cluster_0003"]:
            _make_cluster_smet(tmp_path, name, 3000.0, [obs])

        wrf_fc = self._make_wrf_hourly(stable_ts)
        extend_all_smets(tmp_path, wrf_fc, 3200.0, stable_ts)

        for name in ["cluster_0001", "cluster_0002", "cluster_0003"]:
            p = tmp_path / f"{name}.smet"
            assert NWP_FLAG in p.read_text(), f"{name}.smet missing NWP_FLAG"

    def test_row_count_matches_lead_hours(self, tmp_path):
        stable_ts = _ts("2020-01-01T12:00")
        obs = ("2020-01-01T12:00", 0.0, 80.0, 2.0, 180.0, 0.0, 250.0, 0.0, 0.5)
        _make_cluster_smet(tmp_path, "cluster_0001", 3000.0, [obs])

        lead = 24
        wrf_fc = self._make_wrf_hourly(stable_ts, n_hours=lead)
        counts = extend_all_smets(tmp_path, wrf_fc, 3200.0, stable_ts)
        assert counts["cluster_0001"] == lead


# ---------------------------------------------------------------------------
# Round-trip / regression
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _build_wrf_smet(self, tmp_path: Path, stable_ts: pd.Timestamp) -> Path:
        """3 × 6h ticks: stable_ts, stable_ts+6h, stable_ts+12h."""
        rows = []
        for i in range(3):
            ts = (stable_ts + pd.Timedelta(hours=i * 6)).strftime("%Y-%m-%dT%H:%M")
            rows.append((ts, -5.0 + i, 80.0, 2.0, 270.0, 0.0, 250.0, 0.0))
        return _make_wrf_smet(tmp_path, rows)

    def _build_cluster_smet(self, tmp_path: Path, stable_ts: pd.Timestamp) -> Path:
        obs = []
        for i in range(5):
            ts = (stable_ts - pd.Timedelta(hours=4 - i)).strftime("%Y-%m-%dT%H:%M")
            obs.append((ts, -6.0, 80.0, 2.0, 270.0, 0.0, 250.0, 0.0, 0.5))
        return _make_cluster_smet(tmp_path, "cluster_0001", 3000.0, obs)

    def test_output_ends_at_stable_plus_lead(self, tmp_path):
        stable_ts = _ts("2020-01-01T06:00")
        wrf_p = self._build_wrf_smet(tmp_path, stable_ts)
        cluster_p = self._build_cluster_smet(tmp_path, stable_ts)

        wrf_raw, wrf_alt = read_wrf_smet(
            wrf_p, stable_ts, stable_ts + pd.Timedelta(hours=48)
        )
        wrf_hourly = _resample_to_hourly(wrf_raw)
        extend_all_smets(tmp_path, wrf_hourly, wrf_alt, stable_ts)

        # Read all timestamps from cluster file
        timestamps = []
        in_data = False
        with open(cluster_p) as f:
            for line in f:
                stripped = line.strip()
                if stripped == "[DATA]":
                    in_data = True
                    continue
                if in_data and stripped and not stripped.startswith("#") and NWP_FLAG not in stripped:
                    try:
                        timestamps.append(pd.Timestamp(stripped.split()[0], tz="UTC"))
                    except Exception:
                        pass

        assert len(timestamps) > 0
        last_ts = max(timestamps)
        assert last_ts == stable_ts + pd.Timedelta(hours=12)

    def test_hs_is_minus999_in_nwp_rows(self, tmp_path):
        stable_ts = _ts("2020-01-01T06:00")
        wrf_p = self._build_wrf_smet(tmp_path, stable_ts)
        cluster_p = self._build_cluster_smet(tmp_path, stable_ts)

        wrf_raw, wrf_alt = read_wrf_smet(
            wrf_p, stable_ts, stable_ts + pd.Timedelta(hours=48)
        )
        wrf_hourly = _resample_to_hourly(wrf_raw)

        fields = ["timestamp", "TA", "RH", "VW", "DW", "ISWR", "ILWR", "MS_Snow", "HS"]
        hs_idx = fields.index("HS")

        extend_all_smets(tmp_path, wrf_hourly, wrf_alt, stable_ts)

        in_nwp = False
        with open(cluster_p) as f:
            for line in f:
                if NWP_FLAG in line:
                    in_nwp = True
                    continue
                if in_nwp and line.strip() and not line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) > hs_idx:
                        assert parts[hs_idx] == "-999", \
                            f"NWP row HS should be -999, got {parts[hs_idx]}"

    def test_idempotent_second_run(self, tmp_path):
        stable_ts = _ts("2020-01-01T06:00")
        wrf_p = self._build_wrf_smet(tmp_path, stable_ts)
        cluster_p = self._build_cluster_smet(tmp_path, stable_ts)

        wrf_raw, wrf_alt = read_wrf_smet(
            wrf_p, stable_ts, stable_ts + pd.Timedelta(hours=48)
        )
        wrf_hourly = _resample_to_hourly(wrf_raw)

        extend_all_smets(tmp_path, wrf_hourly, wrf_alt, stable_ts)
        text_after_first = cluster_p.read_text()

        extend_all_smets(tmp_path, wrf_hourly, wrf_alt, stable_ts)
        text_after_second = cluster_p.read_text()

        assert text_after_first == text_after_second, \
            "Second run of extend_all_smets should produce identical output (idempotent)"
