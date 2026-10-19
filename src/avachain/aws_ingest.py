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
    python aws_ingest.py                       # fetch all configured stations
    python aws_ingest.py --station CAABT       # single station
    python aws_ingest.py --since 2026-01-01    # force refetch from a date
    python aws_ingest.py --dry-run             # print what would be fetched

Called by run_daily.sh as the first step before smet_append.py.

TODO: replace the stub _fetch_station() with the real CAIC API / CDOT RWIS
      call once the endpoint and auth token are confirmed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# --- Station registry ---
# Default stations for little_prof.  For other slopes the pipeline passes
# stations built via stations_from_cfg(cfg) instead of this module-level dict.
STATIONS = {
    "CAABT": {
        "name":    "Abasin Top",
        "lat":     39.6368,
        "lon":    -105.8719,
        "elev_m":  3780,
    },
    "CAABM": {
        "name":    "Abasin Mid",
        "lat":     39.6381,
        "lon":    -105.8732,
        "elev_m":  3530,
    },
}


def stations_from_cfg(cfg) -> dict:
    """Build a STATIONS-format dict from a ProjectConfig."""
    return {
        cfg.summit_id: {"name": cfg.summit_id,
                        "lat":    cfg.summit_lat,
                        "lon":    cfg.summit_lon,
                        "elev_m": cfg.summit_alt_m},
        cfg.base_id:   {"name": cfg.base_id,
                        "lat":    cfg.base_lat,
                        "lon":    cfg.base_lon,
                        "elev_m": cfg.base_alt_m},
    }

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


# ---------------------------------------------------------------------------
# Data fetch — stub, replace with real CAIC / CDOT RWIS API call
# ---------------------------------------------------------------------------

def _fetch_station(station_id: str, since: pd.Timestamp | None,
                   until: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    TODO: implement real fetch from CAIC API or CDOT RWIS.

    Expected return: DataFrame with columns matching CACHE_COLS, timestamps
    UTC-aware, no duplicates, sorted ascending by timestamp.

    For now returns an empty frame so the pipeline skeleton can run end-to-end
    without a live data source.
    """
    print(f"    [aws_ingest] STUB: would fetch {station_id} "
          f"since {since} until {until or 'now'}")
    return pd.DataFrame(columns=CACHE_COLS)


def seed_cache_from_csv(station_id: str, csv_path: Path, cfg,
                        role: str = "summit", dry_run: bool = False) -> int:
    """Seed the station cache from the archived CAIC weather CSV.

    The live fetch is still a stub, so this is how an operational replay gets
    station forcing: convert the whole archived season once, then gate it with
    `smet_append --until`. Conversion is delegated to
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

    WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path(station_id), index=False)
    print(f"    [{station_id}] wrote {cache_path(station_id)}")
    return len(df)


def fetch_and_cache(station_id: str,
                    since: pd.Timestamp | None = None,
                    dry_run: bool = False) -> int:
    """Fetch new records for station_id and append to cache.

    If `since` is None, resumes from the last cached timestamp (or fetches
    all available history if the cache is empty).

    Returns the number of new rows cached (0 on dry_run or no new data).
    """
    last = since if since is not None else last_cached_timestamp(station_id)
    until = pd.Timestamp(datetime.now(timezone.utc))

    if last is not None and last >= until:
        print(f"    [{station_id}] cache already current ({last})")
        return 0

    print(f"    [{station_id}] fetching {last} → {until.strftime('%Y-%m-%d %H:%M')} UTC")
    new_rows = _fetch_station(station_id, since=last, until=until)

    if dry_run:
        print(f"    [{station_id}] dry-run: {len(new_rows)} rows would be cached")
        return 0

    n = append_to_cache(station_id, new_rows)
    print(f"    [{station_id}] cached {n} new rows "
          f"(total: {len(load_cache(station_id))})")
    return n


def fetch_all(stations: dict | None = None,
              since: pd.Timestamp | None = None,
              dry_run: bool = False) -> dict[str, int]:
    """Fetch all stations.  Pass stations=stations_from_cfg(cfg) for a non-default slope."""
    active = stations if stations is not None else STATIONS
    return {sid: fetch_and_cache(sid, since=since, dry_run=dry_run)
            for sid in active}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and cache AWS station data for the daily pipeline")
    parser.add_argument("--station", default=None,
                        help="Single station ID (default: all stations for the slope)")
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

    # Resolve station registry and weather cache from config.
    global WEATHER_CACHE_DIR
    active_stations = STATIONS
    toml_path = Path(args.toml) if args.toml else None
    if toml_path is None and args.project_dir and args.slope_name:
        toml_path = Path(args.project_dir) / "slopes" / args.slope_name / "slope_config.toml"
    if toml_path and toml_path.exists():
        from config import ProjectConfig
        cfg = ProjectConfig.from_toml(toml_path)
        active_stations = stations_from_cfg(cfg)
        WEATHER_CACHE_DIR = cfg.weather_cache_dir

    since_ts = pd.Timestamp(args.since, tz="UTC") if args.since else None

    if args.from_csv:
        if not (toml_path and toml_path.exists()):
            raise SystemExit("--from-csv needs --toml (for station metadata)")
        csv_path = Path(args.from_csv) if args.from_csv is not True else cfg.weather_csv
        n = seed_cache_from_csv(cfg.summit_id, csv_path, cfg,
                                role="summit", dry_run=args.dry_run)
        print(f"\n  Seeded {n} rows for {cfg.summit_id}")
        return

    if args.station:
        fetch_and_cache(args.station, since=since_ts, dry_run=args.dry_run)
    else:
        counts = fetch_all(stations=active_stations, since=since_ts, dry_run=args.dry_run)
        total = sum(counts.values())
        print(f"\n  Total new rows cached: {total}")


if __name__ == "__main__":
    main()
