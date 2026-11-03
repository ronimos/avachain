"""
sql_util.py — read CAIC station observations from the CAIC MySQL database.

The database runs on the same host as the pipeline, so this connects to it
directly. The SSH-tunnel fallback was removed on 2026-11-03 (Ron): it was for
running from a laptop, and it hid connection failures behind a second attempt.

Stations come only from the slope TOML's [[stations]] entries (sql_id and
sql_columns). There is no built-in station list: a slope without stations is
an error, not a silent fall-back to A-Basin.

The frame returned by fetch_weather() has the layout of the archived
weather_data.csv: a `time` index (UTC, naive, as stored) and the raw CAIC
columns (tenths of °F, tenths of mph, ...), each taken from the station that
lists it. smet_writer.convert_caic_frame() turns it into SMET units.

Credentials (db_user, db_password, db_name, local_host, db_port) are read,
first file found wins, from:
  1. $AVACHAIN_DB_ENV
  2. ~/.config/avachain/db.env
  3. the repository's .env (the dev checkout's legacy location)
Variables already set in the environment take precedence over all of them.

CLI (writes the archived-format CSV, e.g. to re-seed a station cache):
    python sql_util.py --toml slopes/<slope>/slope_config.toml \
        [--start "2025-10-01 00:00:00"] [--end ...] [--out weather_data.csv]
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).resolve().parents[2]
TABLES = ["obsBattery", "obsHydro", "obsSnow", "obsSolar", "obsWX"]
ENV_VARS = ("db_user", "db_password", "db_name", "local_host")


# --------- Stations ---------

def stations_from_cfg(cfg) -> dict:
    """{sql_id: [columns]} from the slope TOML's [[stations]] entries.

    Raises ValueError when the TOML lists no stations, or a station has no
    sql_id or no sql_columns.
    """
    stations = list(getattr(cfg, "stations", None) or [])
    if not stations:
        raise ValueError("the slope TOML has no [[stations]] entries; "
                         "station data cannot be fetched")
    out = {}
    for s in stations:
        sid = s.get("sql_id")
        cols = s.get("sql_columns")
        if not sid or not cols:
            raise ValueError(f"[[stations]] entry {s.get('name') or s.get('role')!r} "
                             f"needs both sql_id and sql_columns")
        out[sid] = list(cols)
    return out


# --------- Connection ---------

def _env_file() -> Optional[Path]:
    cands = [os.getenv("AVACHAIN_DB_ENV"),
             Path.home() / ".config" / "avachain" / "db.env",
             ROOT_PATH / ".env"]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def get_engine() -> sqlalchemy.engine.Engine:
    """Engine for the CAIC database; raises with the remedy when unconfigured."""
    from dotenv import load_dotenv
    env = _env_file()
    if env is not None:
        load_dotenv(env, override=False)
    missing = [k for k in ENV_VARS if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"CAIC database credentials missing ({', '.join(missing)}). Put "
            f"db_user, db_password, db_name, local_host and db_port in "
            f"$AVACHAIN_DB_ENV or ~/.config/avachain/db.env (mode 600)"
            + (f"; read {env}" if env else "; no credentials file found"))
    url = sqlalchemy.engine.URL.create(
        "mysql+pymysql", username=os.getenv("db_user"),
        password=os.getenv("db_password"), host=os.getenv("local_host"),
        port=int(os.getenv("db_port", 3306)), database=os.getenv("db_name"))
    return create_engine(url, connect_args={"connect_timeout": 20})


# --------- Queries ---------

def _fetch_table_data(engine, tables, start_time, st_id, end_time=None):
    """One frame per table holding rows for st_id with start < time <= end.

    A table with no rows is skipped. A connection or SQL error raises: it
    used to be logged and skipped, so a database outage looked like a
    station with no data.
    """
    frames = []
    q = "SELECT * FROM {t} WHERE time > :start AND staname = :sid"
    params = {"start": start_time, "sid": st_id}
    if end_time is not None:
        q += " AND time <= :end"
        params["end"] = end_time
    for table in tables:
        df = pd.read_sql(text(q.format(t=table)), con=engine, params=params)
        if df.empty:
            logger.info(f"no rows in {table} for {st_id}")
            continue
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df.drop(columns=["staname"], errors="ignore").set_index("time"))
    return frames


def get_weather_from_sql_db(st_id: str,
                            start_time: Optional[datetime.datetime] = None,
                            end_time: Optional[datetime.datetime] = None,
                            engine=None) -> pd.DataFrame:
    """All observation columns for one station, start < time <= end.

    start defaults to 7 days ago. Times are the database's (UTC, naive).
    """
    if start_time is None:
        start_time = datetime.datetime.utcnow() - datetime.timedelta(hours=168)
    engine = engine or get_engine()
    frames = _fetch_table_data(engine, TABLES, _ts(start_time), st_id,
                               _ts(end_time) if end_time is not None else None)
    if not frames:
        logger.warning(f"no data retrieved for station {st_id}")
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def fetch_weather(stations: dict, start_time, end_time=None, engine=None) -> pd.DataFrame:
    """The archived weather_data.csv layout for a slope's stations.

    @param stations: {sql_id: [columns]} from stations_from_cfg
    @returns: DataFrame indexed by `time`, one column per requested column,
             taken from the station that lists it; empty when no rows
    """
    engine = engine or get_engine()
    parts = []
    for sid, cols in stations.items():
        df = get_weather_from_sql_db(sid, start_time, end_time, engine=engine)
        if df.empty:
            logger.warning(f"station {sid}: no rows between {start_time} and "
                           f"{end_time or 'now'}")
            continue
        keep = df.columns.tolist() if cols == ["*"] else [c for c in cols if c in df.columns]
        absent = [c for c in cols if c != "*" and c not in df.columns]
        if absent:
            logger.warning(f"station {sid}: columns not in the database: {absent}")
        parts.append(df[keep])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "time"
    return out


def season_start(toml_path) -> Optional[str]:
    """[snowpack] season_start from a slope TOML as 'YYYY-MM-DD 00:00:00', or None."""
    import tomllib
    with open(toml_path, "rb") as f:
        v = tomllib.load(f).get("snowpack", {}).get("season_start")
    return f"{v} 00:00:00" if v else None


def _ts(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")


# --------- CLI ---------

def parse_args():
    p = argparse.ArgumentParser(description="Fetch a slope's station data from the "
                                            "CAIC database into the archived CSV layout.")
    p.add_argument("--toml", required=True, help="slope_config.toml (stations and paths)")
    p.add_argument("--start", default=None,
                   help="start, 'YYYY-MM-DD HH:MM:SS' UTC (default: the TOML's season_start)")
    p.add_argument("--end", default=None, help="end, inclusive (default: now)")
    p.add_argument("--out", default=None,
                   help="output CSV (default: the TOML's weather_csv)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ProjectConfig
    args = parse_args()
    cfg = ProjectConfig.from_toml(Path(args.toml))
    start = args.start or season_start(args.toml)
    if start is None:
        raise SystemExit("no --start and no [snowpack] season_start in the TOML")
    df = fetch_weather(stations_from_cfg(cfg), start, args.end)
    if df.empty:
        raise SystemExit("no data collected from any station; CSV not written")
    out = Path(args.out) if args.out else Path(cfg.weather_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    logger.info(f"wrote {len(df)} rows to {out}")
