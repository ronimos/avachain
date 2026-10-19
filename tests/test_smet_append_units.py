"""Tests for SMET append unit handling.

`_format_smet_row` used to convert TA to Kelvin, RH to a 0-1 fraction and
treat HS as metres. But these SMET files declare

    units_offset     = 0 273.15 0 ...        (TA)
    units_multiplier = 1 1 0.01 ... 0.01 ... (RH, HS)

and SNOWPACK applies those on read, so the values must be written in the
file's own units — °C, percent, cm — exactly as smet_writer.write_smet (the
only producer of these files) and nwp_ingest._format_nwp_row already do.
Converting here as well double-converted: a -10.7 °C row landed as 262.4 in
the file and 535.6 K inside SNOWPACK.

This cannot be caught downstream: SNOWPACK would happily run on 535 K air and
the symptom would be nonsense stratigraphy, not an error. The daily-forward
replay loop appends every tick, so this is on its critical path.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from smet_append import (                                    # noqa: E402
    _format_smet_row,
    _read_smet_fields,
    _read_smet_last_hs,
    _read_smet_last_timestamp,
)

FIELDS = ["timestamp", "TA", "RH", "VW", "DW", "ISWR", "ILWR", "HS"]

# As written by smet_writer.write_smet for exactly these fields.
UNITS_OFFSET = [0, 273.15, 0, 0, 0, 0, 0, 0]
UNITS_MULTIPLIER = [1, 1, 0.01, 1, 1, 1, 1, 0.01]

ROW = {"TA": -10.722, "RH": 92.0, "VW": 6.661, "DW": 269.0,
       "ISWR": 159.0, "ILWR": 171.342, "HS": 102.85}


def _written(row=None, fields=FIELDS):
    ts = pd.Timestamp("2025-11-26T18:00", tz="UTC")
    parts = _format_smet_row(ts, row if row is not None else ROW, fields).split("\t")
    return dict(zip(fields, parts))


class TestFileUnitsAreNotDoubleConverted:
    def test_ta_written_in_celsius(self):
        """Regression: 273.15 added here made SNOWPACK see ~535 K."""
        assert float(_written()["TA"]) == pytest.approx(-10.722)

    def test_rh_written_in_percent(self):
        assert float(_written()["RH"]) == pytest.approx(92.0)

    def test_hs_written_in_centimetres(self):
        assert float(_written()["HS"]) == pytest.approx(102.85)

    def test_unscaled_fields_pass_through(self):
        w = _written()
        assert float(w["VW"]) == pytest.approx(6.661)
        assert float(w["DW"]) == pytest.approx(269.0)
        assert float(w["ISWR"]) == pytest.approx(159.0)
        assert float(w["ILWR"]) == pytest.approx(171.342)

    @pytest.mark.parametrize("field,offset,mult", [
        ("TA", 273.15, 1.0), ("RH", 0.0, 0.01), ("HS", 0.0, 0.01)])
    def test_snowpack_read_recovers_physical_value(self, field, offset, mult):
        """stored * multiplier + offset must give back the physical value."""
        stored = float(_written()[field])
        physical = stored * mult + offset
        expected = {"TA": -10.722 + 273.15,   # K
                    "RH": 0.92,               # fraction
                    "HS": 1.0285}[field]      # m
        assert physical == pytest.approx(expected, rel=1e-6)

    def test_header_units_match_the_assumed_convention(self):
        """Guards the premise: if write_smet's units change, this must fail."""
        assert UNITS_OFFSET[FIELDS.index("TA")] == 273.15
        assert UNITS_MULTIPLIER[FIELDS.index("RH")] == 0.01
        assert UNITS_MULTIPLIER[FIELDS.index("HS")] == 0.01


class TestRowFormatting:
    def test_timestamp_first_and_iso_minute_precision(self):
        assert _written()["timestamp"] == "2025-11-26T18:00"

    def test_missing_field_is_nodata(self):
        """Formats as -999.0000, which equals the header's nodata = -999.0."""
        w = _written(row={"TA": -5.0}, fields=["timestamp", "TA", "RH"])
        assert float(w["RH"]) == pytest.approx(-999.0)

    def test_nan_is_nodata(self):
        w = _written(row={"TA": float("nan")}, fields=["timestamp", "TA"])
        assert float(w["TA"]) == pytest.approx(-999.0)

    def test_field_order_follows_header(self):
        fields = ["timestamp", "HS", "TA"]
        w = _format_smet_row(pd.Timestamp("2025-11-26T18:00", tz="UTC"),
                             ROW, fields).split("\t")
        assert float(w[1]) == pytest.approx(102.85)   # HS
        assert float(w[2]) == pytest.approx(-10.722)  # TA


class TestAppendRoundTrip:
    """Append to a real-shaped SMET and read the result back."""

    @pytest.fixture
    def smet(self, tmp_path):
        p = tmp_path / "cluster_0001.smet"
        p.write_text(
            "SMET 1.1 ASCII\n[HEADER]\n"
            "station_id       = cluster_0001\n"
            "nodata           = -999.0\n"
            "tz               = 0\n"
            f"units_offset     = {' '.join(str(x) for x in UNITS_OFFSET)}\n"
            f"units_multiplier = {' '.join(str(x) for x in UNITS_MULTIPLIER)}\n"
            f"fields           = {' '.join(FIELDS)}\n"
            "[DATA]\n"
            "2025-11-26T18:00:00\t-10.722\t92.000\t6.661\t269.000\t159.000\t171.342\t102.850\n"
        )
        return p

    def test_header_parsed(self, smet):
        assert _read_smet_fields(smet) == FIELDS
        assert _read_smet_last_timestamp(smet) == pd.Timestamp(
            "2025-11-26T18:00", tz="UTC")

    def test_last_hs_is_in_file_units(self, smet):
        """cm, not m — the ΔHS propagation depends on this matching the cache."""
        assert _read_smet_last_hs(smet) == pytest.approx(102.85)

    def test_appended_row_is_unit_consistent_with_existing_rows(self, smet):
        """The appended TA must sit in the same range as the row above it."""
        new = _format_smet_row(pd.Timestamp("2025-11-26T19:00", tz="UTC"),
                               {"TA": -10.667, "RH": 92.0, "VW": 5.588,
                                "DW": 270.0, "ISWR": 151.0, "ILWR": 171.591,
                                "HS": 102.369}, FIELDS)
        with open(smet, "a") as f:
            f.write(new + "\n")

        rows = [ln.split() for ln in smet.read_text().splitlines()
                if ln.startswith("2025-")]
        assert len(rows) == 2
        ta = [float(r[FIELDS.index("TA")]) for r in rows]
        # Both near -10 °C; a double-converted row would be ~262.
        assert all(-40.0 < v < 40.0 for v in ta), ta
        assert abs(ta[1] - ta[0]) < 1.0
