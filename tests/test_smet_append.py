"""
tests/test_smet_append.py — Unit tests for smet_append.py helpers.

Run with:  pytest tests/test_smet_append.py -v

Key risk: _format_smet_row() uses OPPOSITE unit conventions from nwp_ingest:
  TA  → Kelvin    (+273.15)   nwp_ingest writes Celsius
  RH  → fraction  (/100)      nwp_ingest writes percent
A wrong merge silently runs SNOWPACK at the wrong temperature.
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

from smet_append import (
    _format_smet_row,
    _read_smet_last_timestamp,
    _read_smet_last_hs,
    _propagate_hs_station_only,
    append_cluster_smet,
)
import smet_append as _sa_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDX6 = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")


def _make_smet(tmp_path: Path, data_rows: list[str], extra_header: str = "") -> Path:
    """Write a minimal cluster SMET with the given raw data lines."""
    p = tmp_path / "cluster_0001.smet"
    content = textwrap.dedent(f"""\
        SMET 1.1 ASCII
        [HEADER]
        station_id = cluster_0001
        altitude   = 3000.0
        fields     = timestamp TA RH VW HS
        {extra_header}
        [DATA]
    """)
    for row in data_rows:
        content += row + "\n"
    p.write_text(content)
    return p


def _make_smet_no_hs(tmp_path: Path, data_rows: list[str]) -> Path:
    """Write a SMET without an HS field."""
    p = tmp_path / "cluster_no_hs.smet"
    content = textwrap.dedent("""\
        SMET 1.1 ASCII
        [HEADER]
        station_id = cluster_no_hs
        fields     = timestamp TA RH VW
        [DATA]
    """)
    for row in data_rows:
        content += row + "\n"
    p.write_text(content)
    return p


def _read_all_hs(smet_path: Path) -> list[float]:
    """Return all HS values from a SMET data section in order."""
    import re
    fields = None
    hs_idx = None
    result = []
    in_data = False
    with open(smet_path) as f:
        for line in f:
            stripped = line.strip()
            if re.match(r"\s*fields\s*=", stripped):
                fields = stripped.split("=", 1)[1].split()
                hs_idx = fields.index("HS") if "HS" in fields else None
            if stripped == "[DATA]":
                in_data = True
                continue
            if in_data and stripped and not stripped.startswith("#"):
                if hs_idx is not None:
                    parts = stripped.split()
                    result.append(float(parts[hs_idx]))
    return result


# ---------------------------------------------------------------------------
# _format_smet_row — unit conversion
# ---------------------------------------------------------------------------

class TestFormatSmetRow:
    def _fields(self):
        return ["timestamp", "TA", "RH", "VW", "DW", "ISWR", "ILWR", "HS"]

    def test_ta_converted_to_kelvin(self):
        ts = pd.Timestamp("2020-01-01T06:00", tz="UTC")
        row = {"TA": 0.0, "RH": 80.0, "VW": 2.0, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "HS": 0.5}
        line = _format_smet_row(ts, row, self._fields())
        parts = line.split("\t")
        ta_idx = self._fields().index("TA")
        assert float(parts[ta_idx]) == pytest.approx(273.15, abs=0.01)

    def test_ta_negative_celsius_converted_correctly(self):
        ts = pd.Timestamp("2020-01-01T06:00", tz="UTC")
        row = {"TA": -10.0, "RH": 80.0, "VW": 2.0, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "HS": 0.5}
        line = _format_smet_row(ts, row, self._fields())
        parts = line.split("\t")
        ta_idx = self._fields().index("TA")
        assert float(parts[ta_idx]) == pytest.approx(263.15, abs=0.01)

    def test_rh_converted_to_fraction(self):
        ts = pd.Timestamp("2020-01-01T06:00", tz="UTC")
        row = {"TA": 0.0, "RH": 75.0, "VW": 2.0, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "HS": 0.5}
        line = _format_smet_row(ts, row, self._fields())
        parts = line.split("\t")
        rh_idx = self._fields().index("RH")
        assert float(parts[rh_idx]) == pytest.approx(0.75, abs=1e-5)

    def test_rh_100pct_becomes_1_0(self):
        ts = pd.Timestamp("2020-01-01T06:00", tz="UTC")
        row = {"TA": 0.0, "RH": 100.0, "VW": 2.0, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "HS": 0.5}
        line = _format_smet_row(ts, row, self._fields())
        parts = line.split("\t")
        rh_idx = self._fields().index("RH")
        assert float(parts[rh_idx]) == pytest.approx(1.0, abs=1e-5)

    def test_hs_written_in_metres_no_conversion(self):
        ts = pd.Timestamp("2020-01-01T06:00", tz="UTC")
        row = {"TA": 0.0, "RH": 80.0, "VW": 2.0, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "HS": 1.234}
        line = _format_smet_row(ts, row, self._fields())
        parts = line.split("\t")
        hs_idx = self._fields().index("HS")
        assert float(parts[hs_idx]) == pytest.approx(1.234, abs=1e-5)

    def test_nan_written_as_minus999(self):
        ts = pd.Timestamp("2020-01-01T06:00", tz="UTC")
        row = {"TA": float("nan"), "RH": float("nan"), "VW": float("nan"),
               "DW": float("nan"), "ISWR": float("nan"), "ILWR": float("nan"),
               "HS": float("nan")}
        line = _format_smet_row(ts, row, self._fields())
        parts = line.split("\t")
        for i, field in enumerate(self._fields()):
            if field == "timestamp":
                continue
            assert parts[i] == "-999", f"Expected -999 for NaN field {field}, got {parts[i]}"

    def test_timestamp_format(self):
        ts = pd.Timestamp("2026-03-15T09:00", tz="UTC")
        row = {"TA": 0.0, "RH": 80.0, "VW": 2.0, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "HS": 0.5}
        line = _format_smet_row(ts, row, self._fields())
        assert line.startswith("2026-03-15T09:00")

    def test_vw_written_as_four_decimal_float(self):
        ts = pd.Timestamp("2020-01-01T06:00", tz="UTC")
        row = {"TA": 0.0, "RH": 80.0, "VW": 3.14159, "DW": 270.0,
               "ISWR": 0.0, "ILWR": 250.0, "HS": 0.5}
        line = _format_smet_row(ts, row, self._fields())
        parts = line.split("\t")
        vw_idx = self._fields().index("VW")
        # Should be a valid float with ≤ 4 decimal places
        val = float(parts[vw_idx])
        assert val == pytest.approx(3.1416, abs=1e-4)

    def test_nwp_ingest_convention_differs(self):
        """Regression guard: smet_append stores K; nwp_ingest stores C. Must stay different."""
        from nwp_ingest import _format_nwp_row as nwp_fmt
        ts = pd.Timestamp("2020-01-01T06:00", tz="UTC")
        fields = ["timestamp", "TA", "RH", "HS"]
        row = {"TA": 0.0, "RH": 80.0, "HS": 0.5}

        smet_line = _format_smet_row(ts, row, fields)
        nwp_line = nwp_fmt(ts, row, fields)

        smet_parts = smet_line.split("\t")
        nwp_parts = nwp_line.split("\t")

        ta_idx = fields.index("TA")
        rh_idx = fields.index("RH")

        # TA: smet_append writes K (~273), nwp_ingest writes C (0.0)
        assert float(smet_parts[ta_idx]) != float(nwp_parts[ta_idx]), \
            "smet_append and nwp_ingest must use different TA units (K vs C)"
        # RH: smet_append writes fraction (~0.8), nwp_ingest writes percent (80.0)
        assert float(smet_parts[rh_idx]) != float(nwp_parts[rh_idx]), \
            "smet_append and nwp_ingest must use different RH units (frac vs %)"


# ---------------------------------------------------------------------------
# _read_smet_last_timestamp
# ---------------------------------------------------------------------------

class TestReadSmetLastTimestamp:
    def test_returns_last_utc_timestamp(self, tmp_path):
        p = _make_smet(tmp_path, [
            "2020-01-01T00:00\t273.15\t0.80\t2.0\t0.5",
            "2020-01-01T01:00\t273.10\t0.78\t1.8\t0.5",
            "2020-01-01T02:00\t272.90\t0.75\t2.2\t0.5",
        ])
        ts = _read_smet_last_timestamp(p)
        assert ts == pd.Timestamp("2020-01-01T02:00", tz="UTC")

    def test_returns_none_for_empty_data_section(self, tmp_path):
        p = _make_smet(tmp_path, [])
        assert _read_smet_last_timestamp(p) is None

    def test_returns_utc_aware_timestamp(self, tmp_path):
        p = _make_smet(tmp_path, ["2020-06-01T12:00\t280.0\t0.60\t3.0\t0.0"])
        ts = _read_smet_last_timestamp(p)
        assert ts.tzinfo is not None
        assert ts.tzname() == "UTC"

    def test_skips_comment_lines(self, tmp_path):
        p = _make_smet(tmp_path, [
            "2020-01-01T00:00\t273.15\t0.80\t2.0\t0.5",
            "# NWP_FORECAST CAIC_WRF",
            "2020-01-01T03:00\t274.00\t0.82\t2.5\t0.5",
        ])
        ts = _read_smet_last_timestamp(p)
        # Comment line should be skipped; last real timestamp is T+03:00
        assert ts == pd.Timestamp("2020-01-01T03:00", tz="UTC")

    def test_single_row(self, tmp_path):
        p = _make_smet(tmp_path, ["2026-01-18T12:00\t268.15\t0.90\t1.0\t0.8"])
        ts = _read_smet_last_timestamp(p)
        assert ts == pd.Timestamp("2026-01-18T12:00", tz="UTC")


# ---------------------------------------------------------------------------
# _read_smet_last_hs
# ---------------------------------------------------------------------------

class TestReadSmetLastHs:
    def test_returns_last_valid_hs(self, tmp_path):
        p = _make_smet(tmp_path, [
            "2020-01-01T00:00\t273.15\t0.80\t2.0\t0.50",
            "2020-01-01T01:00\t273.10\t0.78\t1.8\t0.55",
            "2020-01-01T02:00\t272.90\t0.75\t2.2\t0.60",
        ])
        assert _read_smet_last_hs(p) == pytest.approx(0.60)

    def test_returns_none_when_all_nodata(self, tmp_path):
        # Pre-fix files written by old smet_append had all -999 for HS
        p = _make_smet(tmp_path, [
            "2020-01-01T00:00\t273.15\t0.80\t2.0\t-999",
            "2020-01-01T01:00\t273.10\t0.78\t1.8\t-999",
        ])
        assert _read_smet_last_hs(p) is None

    def test_returns_none_for_empty_data_section(self, tmp_path):
        p = _make_smet(tmp_path, [])
        assert _read_smet_last_hs(p) is None

    def test_returns_none_when_no_hs_field(self, tmp_path):
        p = _make_smet_no_hs(tmp_path, ["2020-01-01T00:00\t273.15\t0.80\t2.0"])
        assert _read_smet_last_hs(p) is None

    def test_skips_nodata_returns_last_valid(self, tmp_path):
        # File has valid HS early, then -999 rows appended by old code
        p = _make_smet(tmp_path, [
            "2020-01-01T00:00\t273.15\t0.80\t2.0\t0.80",
            "2020-01-01T01:00\t273.10\t0.78\t1.8\t-999",
            "2020-01-01T02:00\t272.90\t0.75\t2.2\t-999",
        ])
        assert _read_smet_last_hs(p) == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# _propagate_hs_station_only
# ---------------------------------------------------------------------------

class TestPropagateHsStationOnly:
    def test_anchor_row_unchanged(self):
        stn = pd.Series([0.50], index=_IDX6[:1])
        result = _propagate_hs_station_only(1.20, stn)
        assert result.iloc[0] == pytest.approx(1.20)

    def test_snowfall_adds_station_delta(self):
        stn = pd.Series([0.50, 0.60], index=_IDX6[:2])  # +0.10
        result = _propagate_hs_station_only(1.20, stn)
        assert result.iloc[1] == pytest.approx(1.30)

    def test_settlement_applies_proportional_ratio(self):
        stn = pd.Series([0.60, 0.54], index=_IDX6[:2])  # ratio 0.9
        result = _propagate_hs_station_only(1.20, stn)
        assert result.iloc[1] == pytest.approx(1.08)

    def test_deep_cluster_stays_proportionally_deeper_after_settlement(self):
        stn = pd.Series([0.60, 0.54], index=_IDX6[:2])  # ratio 0.9
        deep = _propagate_hs_station_only(2.00, stn)
        shallow = _propagate_hs_station_only(0.50, stn)
        assert (deep.iloc[1] / shallow.iloc[1]) == pytest.approx(2.00 / 0.50, rel=1e-6)

    def test_result_never_negative(self):
        stn = pd.Series([1.00, 0.001], index=_IDX6[:2])  # near-zero station
        result = _propagate_hs_station_only(0.02, stn)
        assert (result.values >= 0).all()

    def test_nan_station_holds_last_cluster_value(self):
        stn = pd.Series([0.50, np.nan, 0.60], index=_IDX6[:3])
        result = _propagate_hs_station_only(1.00, stn)
        # i=1: stn_curr NaN → hold
        assert result.iloc[1] == pytest.approx(1.00)
        # i=2: stn_prev is NaN → hold again
        assert result.iloc[2] == pytest.approx(1.00)

    def test_station_near_zero_no_divide_by_zero(self):
        # stn_prev < 1e-6 → skip ratio, hold current value
        stn = pd.Series([0.0, 0.0], index=_IDX6[:2])
        result = _propagate_hs_station_only(0.50, stn)
        assert result.iloc[1] == pytest.approx(0.50)

    def test_multi_step_snowfall_then_settlement(self):
        # +0.10 snowfall then 0.9× settlement
        stn = pd.Series([0.50, 0.60, 0.54], index=_IDX6[:3])
        result = _propagate_hs_station_only(1.00, stn)
        assert result.iloc[1] == pytest.approx(1.10)
        assert result.iloc[2] == pytest.approx(1.10 * (0.54 / 0.60))

    def test_output_length_and_index_match_input(self):
        stn = pd.Series([0.5, 0.55, 0.6, 0.58], index=_IDX6[:4])
        result = _propagate_hs_station_only(1.00, stn)
        assert len(result) == len(stn)
        assert (result.index == stn.index).all()


# ---------------------------------------------------------------------------
# append_cluster_smet — HS fill integration
# (mocks _load_met_for_period to avoid parquet dependency in test env)
# ---------------------------------------------------------------------------

def _met_df(timestamps, hs_vals, vw_vals=None) -> pd.DataFrame:
    """Build a met DataFrame as _load_met_for_period would return."""
    idx = pd.DatetimeIndex(timestamps, tz="UTC")
    return pd.DataFrame({
        "TA": [-5.0] * len(timestamps),
        "RH": [80.0] * len(timestamps),
        "VW": vw_vals or [2.0] * len(timestamps),
        "DW": [270.0] * len(timestamps),
        "ISWR": [0.0] * len(timestamps),
        "ILWR": [250.0] * len(timestamps),
        "PSUM": [0.0] * len(timestamps),
        "HS": hs_vals,
    }, index=idx)


class TestAppendClusterSmetHs:
    def _patch_met(self, monkeypatch, met_df: pd.DataFrame):
        """Patch _load_met_for_period to return met_df (respecting include_anchor)."""
        def _fake_load(since, until, station_id=None, include_anchor=False):
            if include_anchor:
                return met_df[(met_df.index >= since) & (met_df.index <= until)]
            return met_df[(met_df.index > since) & (met_df.index <= until)]
        monkeypatch.setattr(_sa_mod, "_load_met_for_period", _fake_load)

    def test_appended_hs_are_real_not_nodata(self, tmp_path, monkeypatch):
        met_ts = pd.date_range("2026-01-10T00:00", periods=4, freq="h", tz="UTC")
        self._patch_met(monkeypatch, _met_df(met_ts, [0.80, 0.82, 0.84, 0.86]))
        smet_p = _make_smet(tmp_path, [
            "2026-01-10T00:00\t268.15\t0.85\t3.0\t0.80",
        ])

        n = append_cluster_smet(smet_p,
                                until=pd.Timestamp("2026-01-10T03:00", tz="UTC"))
        assert n == 3
        appended = _read_all_hs(smet_p)[1:]
        assert all(v >= 0 for v in appended), "HS should be non-negative"
        assert all(v != -999 for v in appended), "HS should not be -999 nodata"

    def test_hs_increases_during_snowfall(self, tmp_path, monkeypatch):
        met_ts = pd.date_range("2026-01-10T00:00", periods=4, freq="h", tz="UTC")
        # Station HS gains +0.05/hr
        self._patch_met(monkeypatch, _met_df(met_ts, [0.60, 0.65, 0.70, 0.75]))
        smet_p = _make_smet(tmp_path, [
            "2026-01-10T00:00\t268.15\t0.85\t3.0\t1.20",
        ])

        append_cluster_smet(smet_p,
                            until=pd.Timestamp("2026-01-10T03:00", tz="UTC"))

        appended = _read_all_hs(smet_p)[1:]
        assert appended[0] == pytest.approx(1.25, abs=1e-3)
        assert appended[1] == pytest.approx(1.30, abs=1e-3)
        assert appended[2] == pytest.approx(1.35, abs=1e-3)

    def test_hs_decreases_proportionally_during_settlement(self, tmp_path, monkeypatch):
        met_ts = pd.date_range("2026-01-10T00:00", periods=2, freq="h", tz="UTC")
        # Station settles: ratio 0.9
        self._patch_met(monkeypatch, _met_df(met_ts, [0.60, 0.54]))
        smet_p = _make_smet(tmp_path, [
            "2026-01-10T00:00\t268.15\t0.85\t3.0\t1.20",
        ])

        append_cluster_smet(smet_p,
                            until=pd.Timestamp("2026-01-10T01:00", tz="UTC"))

        appended = _read_all_hs(smet_p)[1:]
        assert appended[0] == pytest.approx(1.08, abs=1e-3)

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        met_ts = pd.date_range("2026-01-10T00:00", periods=4, freq="h", tz="UTC")
        self._patch_met(monkeypatch, _met_df(met_ts, [0.60] * 4))
        smet_p = _make_smet(tmp_path, [
            "2026-01-10T00:00\t268.15\t0.85\t3.0\t1.20",
        ])

        original_size = smet_p.stat().st_size
        n = append_cluster_smet(smet_p,
                                until=pd.Timestamp("2026-01-10T03:00", tz="UTC"),
                                dry_run=True)
        assert n == 0
        assert smet_p.stat().st_size == original_size

    def test_already_current_returns_zero(self, tmp_path, monkeypatch):
        met_ts = pd.date_range("2026-01-10T00:00", periods=2, freq="h", tz="UTC")
        self._patch_met(monkeypatch, _met_df(met_ts, [0.60, 0.65]))
        smet_p = _make_smet(tmp_path, [
            "2026-01-10T01:00\t268.15\t0.85\t3.0\t1.20",
        ])

        # until == last_ts in smet → nothing to append
        n = append_cluster_smet(smet_p,
                                until=pd.Timestamp("2026-01-10T01:00", tz="UTC"))
        assert n == 0
