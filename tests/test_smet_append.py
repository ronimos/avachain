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

from smet_append import _format_smet_row, _read_smet_last_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
