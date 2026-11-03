"""
aws_ingest.py — Pull latest hourly records from AWS (CAIC stations) and
append them to a local weather cache (Parquet).

This is the first step in the daily forward mode (see
docs/operational_mode_design_concept.md §2.1):

    Ingest new weather data → Extend SMET files → Run SNOWPACK → ...

The cache is a single Parquet file per station (outputs/weather/<station_id>.parquet).
Only rows whose timestamp is newer than the last cached record are fetched,
so the fetch is always incremental regardless of how long the station history is.

Usage (standalone):
    python aws_ingest.py --toml T                      # fetch since the cache's end
    python aws_ingest.py --toml T --since 2026-01-01   # refetch from a date
    python aws_ingest.py --toml T --dry-run            # fetch, don't write
    python aws_ingest.py --toml T --from-csv           # seed from the archived CSV

Called by run_daily_<slope>.sh as the first step before smet_append.py.

The fetch reads the CAIC MySQL database on this host (sql_util), for the
stations in the slope TOML's [[stations]] only. Both stations' raw columns
are joined into the weather_data.csv layout and converted by
smet_writer.convert_caic_frame, the same conversion that seeds the cache
from the archived CSV. So a live row and a seeded row of the same hour are
identical.

The cache is one file per slope, named after the summit station
(<weather_cache_dir>/<summit_id>.parquet), because smet_append reads that one
file. Each fetch re-reads the last OVERLAP_H hours and replaces those cached
rows, so an hour the database recorded late, or filled in afterwards, still
lands. Rows already appended to the SMET files are not rewritten.

It also reads CONTEXT_H hours before that window and does not write them.
ILWR fills night-time cloudiness by interpolating from the daytime before,
so a window converted without them gets up to 10 night rows wrong by up to
6 W/m2 (measured on 2026-01-15). With them, a fetch reproduces the
whole-season conversion of the archived CSV exactly (2026-01-14..20, all
seven columns).

The fetch never runs past `now`: the replay clock ($AVACHAIN_AS_OF) when set,
otherwise the wall clock.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Hours re-read and replaced on every fetch, and hours read before them for
# the ILWR interpolation only (see the module docstring)
OVERLAP_H = 24
CONTEXT_H = 24

# Expected columns in the cache (SMET-compatible names + timestamp).
# Units are exactly smet_writer.load_and_convert's output, which is also what
# gets written into the SMET files unconverted — so HS is cm, not m.
CACHE_COLS = [
    "timestamp",   # UTC, timezone-aware
    "TA",          # air temperature (°C)
    "RH",          # relative humidity (%)
    "VW",          # wind speed (m/s)
    "DW",          # wind direction (°)
    "ISWR",        # incoming shortwave radiation (W/m²)
    "ILWR",        # incoming longwave radiation (W/m²)
    "PSUM",        # precipitation sum (mm/h)
    "HS",          # snow height (cm) — SMET units_multiplier=0.01 gives m
]

WEATHER_CACHE_DIR = Path(__file__).parent.parent.parent / "outputs" / "weather"


def cache_path(station_id: str) -> Path:
    return WEATHER_CACHE_DIR / f"{station_id}.parquet"


def load_cache(station_id: str) -> pd.DataFrame:
    p = cache_path(station_id)
    if not p.exists():
        return pd.DataFrame(columns=CACHE_COLS)
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def last_cached_timestamp(station_id: str) -> pd.Timestamp | None:
    df = load_cache(station_id)
    if df.empty:
        return None
    return df["timestamp"].max()


def append_to_cache(station_id: str, new_rows: pd.DataFrame) -> int:
    """Append new_rows to the station cache; return count of rows written."""
    if new_rows.empty:
        return 0
    WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_cache(station_id)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    combined.to_parquet(cache_path(station_id), index=False)
    return len(new_rows)


def seed_cache_from_csv(station_id: str, csv_path: Path, cfg,
                        role: str = "summit", dry_run: bool = False) -> int:
    """Seed the station cache from the archived CAIC weather CSV.

    This is how an operational replay gets station forcing (convert the whole
    archived season once, then gate it with `smet_append --until`), and one
    way to start a season's cache before the live fetch extends it. Conversion is delegated to
    smet_writer.load_and_convert so the cache carries exactly the units the
    SMET files are written in (TA °C, RH %, HS cm) — hand-rolling it here is
    how a 100x HS error gets in.
    """
    from smet_writer import load_and_convert, station_config_from_cfg

    station = station_config_from_cfg(cfg, role=role)
    df = load_and_convert(str(csv_path), tz_output="UTC", station=station)
    df = df.reset_index().rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    span = f"{df['timestamp'].min()} → {df['timestamp'].max()}"
    print(f"    [{station_id}] {len(df)} rows from {csv_path.name}  ({span})")
    if dry_run:
        print(f"    [{station_id}] dry-run: not written")
        return 0

    _write_cache(station_id, df)
    print(f"    [{station_id}] wrote {cache_path(station_id)}")
    return len(df)


def _write_cache(station_id: str, df: pd.DataFrame) -> None:
    # Write through a temp file and rename, so a crash never leaves a
    # half-written cache for smet_append to read
    WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = cache_path(station_id)
    tmp = dst.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(dst)


def fetch_and_cache(cfg, since: pd.Timestamp | None = None,
                    until: pd.Timestamp | None = None,
                    dry_run: bool = False, engine=None) -> int:
    """Fetch the slope's station rows from the CAIC database into its cache.

    @param cfg: ProjectConfig (stations, summit station metadata)
    @param since: refetch from here; default: the last cached hour minus
                  OVERLAP_H. With no cache, the cache must be seeded first
                  (--from-csv or --since).
    @param until: last hour to take; default: the replay clock or now (UTC)
    @param engine: SQLAlchemy engine (tests); default: sql_util.get_engine()
    @returns: number of cached rows added or replaced
    """
    import sql_util
    from config import replay_as_of
    from smet_writer import convert_caic_frame, station_config_from_cfg

    sid = cfg.summit_id
    if not sid:
        raise ValueError("the slope TOML has no summit station (role = \"summit\")")
    stations = sql_util.stations_from_cfg(cfg)
    if until is None:
        as_of = replay_as_of()
        until = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    until = _utc(until).floor("h")
    last = last_cached_timestamp(sid)
    if since is None:
        if last is None:
            raise FileNotFoundError(
                f"no station cache at {cache_path(sid)}; seed it first with "
                f"`aws_ingest.py --toml <slope_config.toml> --from-csv` or "
                f"`--since YYYY-MM-DD`")
        since = last - pd.Timedelta(hours=OVERLAP_H)
    since = _utc(since)
    if since >= until:
        print(f"    [{sid}] cache already current ({last})")
        return 0

    print(f"    [{sid}] fetching {since:%Y-%m-%d %H:%M} -> {until:%Y-%m-%d %H:%M} UTC "
          f"from {', '.join(stations)}")
    # The database stores naive UTC; the window's start is exclusive. Read
    # CONTEXT_H more for the ILWR interpolation and keep only rows > since.
    start = since - pd.Timedelta(hours=CONTEXT_H)
    raw = sql_util.fetch_weather(stations, start.tz_localize(None),
                                 until.tz_localize(None), engine=engine)
    if raw.empty:
        print(f"    [{sid}] no new rows in the database")
        return 0
    new = convert_caic_frame(raw, tz_output="UTC",
                             station=station_config_from_cfg(cfg, role="summit"))
    new = new.reset_index().rename(columns={"time": "timestamp"})
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
    new = new.dropna(subset=["timestamp"])
    new = new[new["timestamp"] > since].reset_index(drop=True)
    if new.empty:
        print(f"    [{sid}] no new rows in the database")
        return 0
    print(f"    [{sid}] {len(new)} rows, {new['timestamp'].min()} -> "
          f"{new['timestamp'].max()}")
    if dry_run:
        print(f"    [{sid}] dry-run: not written")
        return 0

    cached = load_cache(sid)
    # fetched rows win over cached ones of the same hour (the overlap)
    combined = pd.concat([cached, new], ignore_index=True) \
        .drop_duplicates(subset=["timestamp"], keep="last") \
        .sort_values("timestamp").reset_index(drop=True)
    _write_cache(sid, combined)
    print(f"    [{sid}] cached {len(new)} rows (total: {len(combined)})")
    return len(new)


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and cache AWS station data for the daily pipeline")
    parser.add_argument("--since", default=None,
                        help="Force refetch from this date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be fetched without writing")
    parser.add_argument("--toml", default=None,
                        help="Slope config TOML; supplies stations and weather cache")
    parser.add_argument("--from-csv", nargs="?", const=True, default=None,
                        help="Seed the cache from an archived CAIC weather CSV "
                             "(default: cfg.weather_csv). Used to stage station "
                             "forcing for an operational replay.")
    parser.add_argument("--project-dir", default=None,
                        help="Project directory; resolves slope_config.toml automatically")
    parser.add_argument("--slope-name", default=None,
                        help="Slope name (used with --project-dir to select config)")
    args = parser.parse_args()

    global WEATHER_CACHE_DIR
    toml_path = Path(args.toml) if args.toml else None
    if toml_path is None and args.project_dir and args.slope_name:
        toml_path = Path(args.project_dir) / "slopes" / args.slope_name / "slope_config.toml"
    if not (toml_path and toml_path.exists()):
        raise SystemExit("aws_ingest needs the slope TOML (--toml): the stations "
                         "and the cache location come from it")
    from config import ProjectConfig
    cfg = ProjectConfig.from_toml(toml_path)
    WEATHER_CACHE_DIR = cfg.weather_cache_dir

    if args.from_csv:
        csv_path = Path(args.from_csv) if args.from_csv is not True else cfg.weather_csv
        n = seed_cache_from_csv(cfg.summit_id, csv_path, cfg,
                                role="summit", dry_run=args.dry_run)
        print(f"\n  Seeded {n} rows for {cfg.summit_id}")
        return

    since_ts = pd.Timestamp(args.since, tz="UTC") if args.since else None
    n = fetch_and_cache(cfg, since=since_ts, dry_run=args.dry_run)
    print(f"\n  Rows cached: {n}")

if __name__ == "__main__":
    main()
