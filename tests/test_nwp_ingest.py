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
    FC_FIELDS,
    LEAD_IN_HOURS,
    NWP_FLAG,
    LAPSE_RATE_K_PER_M,
    _apply_lapse_rate,
    _resample_to_hourly,
    get_stable_ts,
    prune_nwp_rows,
    read_wrf_smet,
    write_all_forecast_smets,
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

    def test_psum_is_held_backward_from_each_6h_rate(self):
        """The rate on a 6 h row covers the six hours BEFORE it. Held forward
        (the old behaviour) put WRF snowfall six hours late: against the file's
        own HNW24, r=0.66 forward vs 0.81 backward."""
        idx = pd.date_range("2020-01-01T00:00", periods=3, freq="6h", tz="UTC")
        df = pd.DataFrame({
            "TA": [0.0]*3, "RH": [80.0]*3, "VW": [2.0]*3, "DW": [180.0]*3,
            "ISWR": [0.0]*3, "ILWR": [250.0]*3,
            "MS_Snow": [0.2, 0.8, 0.0], "MS_Rain": [0.0, 0.1, 0.0],
        }, index=idx)
        out = _resample_to_hourly(df)
        for h in range(1, 7):          # (00, 06] takes the 06:00 rate
            assert out.loc[_ts(f"2020-01-01T{h:02d}:00"), "PSUM"] == pytest.approx(0.9)
        for h in range(7, 13):         # (06, 12] takes the 12:00 rate
            assert out.loc[_ts(f"2020-01-01T{h:02d}:00"), "PSUM"] == pytest.approx(0.0)
        assert "MS_Snow" not in out.columns and "MS_Rain" not in out.columns

    def test_psum_conserves_the_6h_totals(self):
        idx = pd.date_range("2020-01-01T00:00", periods=5, freq="6h", tz="UTC")
        rate = [0.0, 0.5, 1.5, 0.0, 0.3]
        df = pd.DataFrame({"TA": [0.0]*5, "MS_Snow": rate}, index=idx)
        out = _resample_to_hourly(df)
        assert out["PSUM"].iloc[1:].sum() == pytest.approx(6 * sum(rate[1:]))

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

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Forecast SMETs
# ---------------------------------------------------------------------------

class TestForecastSmets:
    """The forecast pass reads its own SMETs; observed SMETs are never touched."""

    def _hourly(self, start="2020-01-01T00:00", n=60):
        idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
        return pd.DataFrame({"TA": 1.0, "RH": 80.0, "VW": 2.0, "DW": 180.0,
                             "ISWR": 0.0, "ILWR": 250.0, "PSUM": 0.5}, index=idx)

    def _obs(self, tmp_path, name="cluster_0001", alt=3200.0):
        d = tmp_path / "obs"
        d.mkdir(exist_ok=True)
        p = d / f"{name}.smet"
        p.write_text(textwrap.dedent(f"""\
            SMET 1.1 ASCII
            [HEADER]
            station_id       = {name}
            latitude         = 39.6
            longitude        = -105.9
            altitude         = {alt}
            slope_angle      = 31.0
            slope_azi        = 120.0
            fields           = timestamp TA RH VW DW ISWR ILWR HS
            [DATA]
            2020-01-01T00:00\t-5\t80\t2\t180\t0\t250\t150
        """))
        return p

    def _read(self, p):
        head, data = p.read_text().split("[DATA]\n", 1)
        rows = [ln.split("\t") for ln in data.strip().splitlines()]
        return head, rows

    def test_observed_smet_is_untouched(self, tmp_path):
        obs = self._obs(tmp_path)
        before = obs.read_text()
        write_all_forecast_smets(obs.parent, tmp_path / "fc", self._hourly(),
                                 3200.0, _ts("2020-01-01T12:00"))
        assert obs.read_text() == before

    def test_layout_is_precipitation_driven(self, tmp_path):
        obs = self._obs(tmp_path)
        write_all_forecast_smets(obs.parent, tmp_path / "fc", self._hourly(),
                                 3200.0, _ts("2020-01-01T12:00"))
        head, rows = self._read(tmp_path / "fc" / "cluster_0001.smet")
        assert "fields           = " + " ".join(FC_FIELDS) in head
        assert "PSUM" in FC_FIELDS and FC_FIELDS[-1] == "HS"
        assert "slope_angle      = 31.0" in head and "altitude         = 3200.0" in head
        assert all(r[-1] == "-999" for r in rows)                  # HS nodata
        assert all(float(r[FC_FIELDS.index("PSUM")]) == 0.5 for r in rows)

    def test_rows_start_lead_in_hours_before_t_stable(self, tmp_path):
        """PSUM ACCUMULATE needs data before the restart step, or SNOWPACK stops
        with 'missing { precipitation }' at its first step."""
        obs = self._obs(tmp_path)
        write_all_forecast_smets(obs.parent, tmp_path / "fc", self._hourly(),
                                 3200.0, _ts("2020-01-01T12:00"))
        _, rows = self._read(tmp_path / "fc" / "cluster_0001.smet")
        assert rows[0][0] == (_ts("2020-01-01T12:00")
                              - pd.Timedelta(hours=LEAD_IN_HOURS)).strftime("%Y-%m-%dT%H:%M")

    def test_rewrite_replaces_the_previous_cycle(self, tmp_path):
        obs = self._obs(tmp_path)
        fc = tmp_path / "fc"
        write_all_forecast_smets(obs.parent, fc, self._hourly(), 3200.0, _ts("2020-01-01T12:00"))
        write_all_forecast_smets(obs.parent, fc, self._hourly("2020-01-02T00:00"),
                                 3200.0, _ts("2020-01-02T12:00"))
        _, rows = self._read(fc / "cluster_0001.smet")
        assert rows[0][0].startswith("2020-01-02T06:00")

    def test_lapse_rate_per_cluster(self, tmp_path):
        self._obs(tmp_path, "cluster_0001", 3200.0)
        self._obs(tmp_path, "cluster_0002", 3700.0)
        write_all_forecast_smets(tmp_path / "obs", tmp_path / "fc", self._hourly(),
                                 3200.0, _ts("2020-01-01T12:00"))
        ta = {c: float(self._read(tmp_path / "fc" / f"{c}.smet")[1][0][1])
              for c in ("cluster_0001", "cluster_0002")}
        assert ta["cluster_0001"] == pytest.approx(1.0)
        assert ta["cluster_0002"] == pytest.approx(1.0 + LAPSE_RATE_K_PER_M * 500, abs=1e-3)
