"""Live station fetch: sql_util + aws_ingest against a stand-in CAIC database.

An in-memory SQLite database holds the CAIC observation tables (obsWX,
obsSnow, obsSolar, obsHydro, obsBattery: time, staname, raw columns). The
real database is MySQL on the pipeline host; the queries are plain SQL with
bound parameters, so SQLite exercises the same code.

Pinned here (2026-11-03, Ron: stations were hard-coded to A-Basin, and the
live fetch was a stub):
  - stations come only from the TOML; no built-in A-Basin fall-back;
  - each column comes from the station that lists it, in the
    weather_data.csv layout;
  - a fetched hour is converted exactly as the archived CSV is;
  - the fetch stops at the replay clock, replaces the overlap, and does not
    write its ILWR context rows;
  - a database error raises instead of looking like "no data".
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import aws_ingest  # noqa: E402
import sql_util  # noqa: E402
from smet_writer import convert_caic_frame, load_and_convert, station_config_from_cfg  # noqa: E402

SUMMIT_COLS = ["swin", "temp", "dewp", "rh", "wspd", "wdir"]
BASE_COLS = ["depth", "pcpac"]


def _cfg(stations=None):
    if stations is None:
        stations = [
            {"role": "summit", "sql_id": "TOP", "name": "Top", "sql_columns": SUMMIT_COLS},
            {"role": "base", "sql_id": "LOW", "name": "Low", "sql_columns": BASE_COLS},
        ]
    return SimpleNamespace(stations=stations, summit_id="TOP", summit_lat=39.64,
                           summit_lon=-105.87, summit_alt_m=3798.0,
                           base_id="LOW", base_lat=39.64, base_lon=-105.87,
                           base_alt_m=3554.0)


def _raw(times):
    """Raw CAIC values (tenths of F, mph, W/m2, inch) with a diurnal swin."""
    h = np.asarray(times.hour)
    sun = np.clip(np.sin((h - 14) / 10 * np.pi), 0, None)   # day ~14-24 UTC
    n = len(times)
    return {
        "TOP": pd.DataFrame({"swin": np.round(6000 * sun), "temp": 200 + np.arange(n) % 50,
                             "dewp": 100.0, "rh": 60 + np.arange(n) % 30,
                             "wspd": 150.0, "wdir": 270.0, "gust": 300.0}, index=times),
        # the base station also reports temp: it must NOT leak into the frame
        "LOW": pd.DataFrame({"depth": 300 + np.arange(n) // 24, "pcpac": 5.0,
                             "temp": -999.0}, index=times),
    }


def _engine(times, drop_hours=()):
    """SQLite stand-in: each station's columns spread over the CAIC tables."""
    eng = sa.create_engine("sqlite://")
    raw = _raw(times)
    tables = {"obsWX": ["temp", "dewp", "rh", "wspd", "wdir", "gust"],
              "obsSolar": ["swin"], "obsSnow": ["depth"], "obsHydro": ["pcpac"],
              "obsBattery": []}
    for t, cols in tables.items():
        rows = []
        for sid, df in raw.items():
            sub = df[[c for c in cols if c in df.columns]].copy()
            sub = sub[~sub.index.isin(pd.DatetimeIndex(drop_hours))]
            sub.insert(0, "staname", sid)
            rows.append(sub)
        out = pd.concat(rows).reset_index(names="time")
        out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out.to_sql(t, eng, index=False)
    return eng, raw


def _csv_frame(raw):
    """What sql_util.fetch_weather should return: weather_data.csv layout."""
    df = pd.concat([raw["TOP"][SUMMIT_COLS], raw["LOW"][BASE_COLS]], axis=1)
    df.index.name = "time"
    return df


TIMES = pd.date_range("2026-01-10 01:00", "2026-01-16 00:00", freq="h")


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(aws_ingest, "WEATHER_CACHE_DIR", tmp_path)
    monkeypatch.delenv("AVACHAIN_AS_OF", raising=False)
    return tmp_path


def _seed(cfg, raw, until):
    full = _csv_frame(raw)
    seed = convert_caic_frame(full[full.index <= until],
                              station=station_config_from_cfg(cfg, "summit"))
    seed = seed.reset_index().rename(columns={"time": "timestamp"})
    seed["timestamp"] = pd.to_datetime(seed["timestamp"], utc=True)
    aws_ingest._write_cache("TOP", seed)


def _reference(cfg, raw):
    ref = convert_caic_frame(_csv_frame(raw), station=station_config_from_cfg(cfg, "summit"))
    ref.index = ref.index.tz_localize("UTC")
    return ref


# --------- stations ---------

def test_no_builtin_station_list():
    assert not hasattr(sql_util, "WEATHER_STATIONS")
    assert not hasattr(aws_ingest, "STATIONS")


def test_stations_from_cfg_requires_toml_stations():
    with pytest.raises(ValueError, match="no \\[\\[stations\\]\\]"):
        sql_util.stations_from_cfg(_cfg(stations=[]))
    with pytest.raises(ValueError, match="sql_columns"):
        sql_util.stations_from_cfg(_cfg(stations=[{"role": "summit", "sql_id": "TOP"}]))
    assert sql_util.stations_from_cfg(_cfg()) == {"TOP": SUMMIT_COLS, "LOW": BASE_COLS}


def test_engine_without_credentials_names_the_remedy(monkeypatch, tmp_path):
    for k in sql_util.ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AVACHAIN_DB_ENV", str(tmp_path / "empty.env"))
    (tmp_path / "empty.env").write_text("")
    monkeypatch.setattr(sql_util, "ROOT_PATH", tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="db.env"):
        sql_util.get_engine()


# --------- sql_util.fetch_weather ---------

def test_fetch_weather_layout_and_window():
    eng, raw = _engine(TIMES)
    df = sql_util.fetch_weather(sql_util.stations_from_cfg(_cfg()),
                                "2026-01-12 00:00:00", "2026-01-13 00:00:00", engine=eng)
    assert list(df.columns) == SUMMIT_COLS + BASE_COLS     # base temp left out
    assert df.index.min() == pd.Timestamp("2026-01-12 01:00")   # start exclusive
    assert df.index.max() == pd.Timestamp("2026-01-13 00:00")   # end inclusive
    exp = _csv_frame(raw).loc[df.index]
    pd.testing.assert_frame_equal(df.astype(float), exp.astype(float), check_freq=False)


def test_database_error_raises():
    eng = sa.create_engine("sqlite://")        # no tables at all
    with pytest.raises((sa.exc.OperationalError, pd.errors.DatabaseError)):
        sql_util.fetch_weather({"TOP": ["temp"]}, "2026-01-12", engine=eng)


# --------- aws_ingest.fetch_and_cache ---------

def test_fetch_matches_whole_season_conversion(cache_dir):
    """Rows the fetch writes equal the archived-CSV conversion, ILWR included."""
    cfg = _cfg()
    eng, raw = _engine(TIMES)
    _seed(cfg, raw, "2026-01-12 06:00")
    n = aws_ingest.fetch_and_cache(cfg, until=pd.Timestamp("2026-01-15 12:00", tz="UTC"),
                                   engine=eng)
    got = aws_ingest.load_cache("TOP").set_index("timestamp")
    ref = _reference(cfg, raw)
    assert got.index.max() == pd.Timestamp("2026-01-15 12:00", tz="UTC")
    assert n == len(got.loc["2026-01-11 06:00":].iloc[1:])     # overlap + new
    cols = ["TA", "RH", "VW", "DW", "ISWR", "HS", "ILWR"]
    pd.testing.assert_frame_equal(got[cols], ref.loc[got.index, cols], check_freq=False)


def test_context_rows_matter(cache_dir):
    """Without CONTEXT_H the first night of the window gets the 0.5 fallback
    cloudiness; this is why the fetch reads it (fails if CONTEXT_H = 0)."""
    cfg = _cfg()
    eng, raw = _engine(TIMES)
    _seed(cfg, raw, "2026-01-13 01:00")
    aws_ingest.fetch_and_cache(cfg, until=pd.Timestamp("2026-01-15 00:00", tz="UTC"),
                               engine=eng)
    got = aws_ingest.load_cache("TOP").set_index("timestamp")
    ref = _reference(cfg, raw)
    np.testing.assert_allclose(got["ILWR"], ref.loc[got.index, "ILWR"], rtol=0, atol=1e-9)
    st = station_config_from_cfg(cfg, "summit")
    full = _csv_frame(raw)
    bare = convert_caic_frame(full[(full.index > "2026-01-12 01:00")
                                   & (full.index <= "2026-01-15 00:00")], station=st)
    assert not np.allclose(bare["ILWR"].values,
                           ref.loc[bare.index.tz_localize("UTC"), "ILWR"].values)


def test_stops_at_replay_clock(cache_dir, monkeypatch):
    cfg = _cfg()
    eng, raw = _engine(TIMES)
    _seed(cfg, raw, "2026-01-12 06:00")
    monkeypatch.setenv("AVACHAIN_AS_OF", "2026-01-13T12:00:00")
    aws_ingest.fetch_and_cache(cfg, engine=eng)
    got = aws_ingest.load_cache("TOP")
    assert got["timestamp"].max() == pd.Timestamp("2026-01-13 12:00", tz="UTC")


def test_overlap_replaces_late_rows(cache_dir):
    """An hour missing at the last fetch and recorded since is filled in."""
    cfg = _cfg()
    late = pd.Timestamp("2026-01-12 04:00")
    eng0, raw = _engine(TIMES, drop_hours=[late])
    _seed(cfg, raw, "2026-01-11 06:00")
    aws_ingest.fetch_and_cache(cfg, until=pd.Timestamp("2026-01-12 06:00", tz="UTC"),
                               engine=eng0)
    assert late.tz_localize("UTC") not in set(aws_ingest.load_cache("TOP")["timestamp"])
    eng1, _ = _engine(TIMES)                       # the database has it now
    aws_ingest.fetch_and_cache(cfg, until=pd.Timestamp("2026-01-13 06:00", tz="UTC"),
                               engine=eng1)
    got = aws_ingest.load_cache("TOP")
    assert late.tz_localize("UTC") in set(got["timestamp"])
    assert got["timestamp"].is_unique and got["timestamp"].is_monotonic_increasing


def test_missing_cache_asks_for_a_seed(cache_dir):
    eng, _ = _engine(TIMES)
    with pytest.raises(FileNotFoundError, match="--from-csv"):
        aws_ingest.fetch_and_cache(_cfg(), until=pd.Timestamp("2026-01-13", tz="UTC"),
                                   engine=eng)


def test_since_starts_an_empty_cache(cache_dir):
    cfg = _cfg()
    eng, raw = _engine(TIMES)
    aws_ingest.fetch_and_cache(cfg, since=pd.Timestamp("2026-01-12", tz="UTC"),
                               until=pd.Timestamp("2026-01-14", tz="UTC"), engine=eng)
    got = aws_ingest.load_cache("TOP")
    assert got["timestamp"].min() == pd.Timestamp("2026-01-12 01:00", tz="UTC")
    assert got["timestamp"].max() == pd.Timestamp("2026-01-14 00:00", tz="UTC")


def test_csv_and_frame_conversion_agree(tmp_path):
    """load_and_convert is convert_caic_frame on the read CSV."""
    cfg = _cfg()
    _, raw = _engine(TIMES)
    p = tmp_path / "w.csv"
    _csv_frame(raw).to_csv(p)
    st = station_config_from_cfg(cfg, "summit")
    a = load_and_convert(str(p), station=st)
    b = convert_caic_frame(pd.read_csv(p, parse_dates=["time"], index_col="time"), station=st)
    pd.testing.assert_frame_equal(a, b)
