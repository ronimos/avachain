"""smet_append must replace yesterday's NWP forecast rows with observations.

A daily NWP cycle leaves each SMET as: observed rows up to T_stable, then an
NWP_FLAG comment, then 48 h of WRF forecast rows. prune_nwp_rows() existed but
had no caller. So on day 2 smet_append read the forecast end as the last row,
saw last_ts >= until, and appended nothing. nwp_ingest then cut at the new
T_stable, which kept the first 24 h of yesterday's forecast as if they were
observations. From day 2 on, no station observation would ever have reached
SNOWPACK.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import smet_append                                            # noqa: E402
from nwp_ingest import NWP_FLAG                               # noqa: E402

FIELDS = ["timestamp", "TA", "RH", "VW", "DW", "ISWR", "ILWR", "HS"]
T0 = pd.Timestamp("2026-02-05T12:00", tz="UTC")              # yesterday's T_stable


def _row(ts, ta, hs):
    return (f"{ts.strftime('%Y-%m-%dT%H:%M:%S')}\t{ta:.3f}\t80.000\t3.000"
            f"\t270.000\t0.000\t200.000\t{hs:.3f}\n")


@pytest.fixture
def smet(tmp_path):
    p = tmp_path / "cluster_0001.smet"
    obs = "".join(_row(T0 - pd.Timedelta(hours=h), -6.0, 180.0) for h in (2, 1, 0))
    # Forecast: obviously different values, so leaking rows are detectable.
    fc = "".join(_row(T0 + pd.Timedelta(hours=h), 99.0, 999.0) for h in range(1, 49))
    p.write_text(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = cluster_0001\nnodata = -999.0\n"
        "units_offset = 0 273.15 0 0 0 0 0 0\n"
        "units_multiplier = 1 1 0.01 1 1 1 1 0.01\n"
        f"fields = {' '.join(FIELDS)}\n[DATA]\n" + obs + NWP_FLAG + "\n" + fc)
    return p


@pytest.fixture
def station(monkeypatch):
    """24 h of observations after T0, station HS rising 1 cm/h."""
    idx = pd.date_range(T0, T0 + pd.Timedelta(hours=24), freq="h")
    met = pd.DataFrame({"TA": -7.0, "RH": 85.0, "VW": 2.0, "DW": 250.0,
                        "ISWR": 0.0, "ILWR": 210.0,
                        "HS": 150.0 + np.arange(len(idx))}, index=idx)

    def fake(since, until, station_id=None, include_anchor=False):
        m = (met.index >= since) if include_anchor else (met.index > since)
        return met[m & (met.index <= until)]
    monkeypatch.setattr(smet_append, "_load_met_for_period", fake)


def _data_rows(p):
    return [ln.split() for ln in p.read_text().splitlines()
            if ln[:4].isdigit()]


def test_forecast_rows_are_replaced_by_observations(smet, station):
    until = T0 + pd.Timedelta(hours=24)
    n = smet_append.append_cluster_smet(smet, until=until)
    assert n == 24
    rows = _data_rows(smet)
    assert pd.Timestamp(rows[-1][0], tz="UTC") == until
    assert NWP_FLAG not in smet.read_text()
    assert not any(float(r[1]) == 99.0 for r in rows), "forecast row survived"


def test_hs_continues_from_the_last_observed_row_not_the_forecast(smet, station):
    smet_append.append_cluster_smet(smet, until=T0 + pd.Timedelta(hours=1))
    first_new = _data_rows(smet)[3]
    # 180 cm observed + 1 cm station delta; the forecast row said 999.
    assert float(first_new[FIELDS.index("HS")]) == pytest.approx(181.0)


def test_dry_run_leaves_the_file_untouched(smet, station):
    before = smet.read_text()
    assert smet_append.append_cluster_smet(
        smet, until=T0 + pd.Timedelta(hours=24), dry_run=True) == 0
    assert smet.read_text() == before


def test_file_without_nwp_rows_appends_as_before(tmp_path, station):
    p = tmp_path / "cluster_0002.smet"
    p.write_text(
        "SMET 1.1 ASCII\n[HEADER]\nstation_id = cluster_0002\nnodata = -999.0\n"
        f"fields = {' '.join(FIELDS)}\n[DATA]\n" + _row(T0, -6.0, 180.0))
    assert smet_append.append_cluster_smet(p, until=T0 + pd.Timedelta(hours=3)) == 3
