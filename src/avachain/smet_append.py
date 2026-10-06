"""
smet_append.py — Append new hourly weather records to per-cluster SMET files.

This replaces the survey-mode behaviour of step_smet (which rewrites all
~6,636 files from scratch) with an append-only update that:

  1. Reads the last timestamp written in each existing SMET file.
  2. Pulls new rows from the AWS weather cache (outputs/weather/<station>.parquet)
     for the period after that timestamp.
  3. Appends those rows to the SMET, leaving everything before intact.

Between UAS surveys the spatial HS distribution stays frozen at the last
survey — only the meteorological forcing columns (TA, RH, VW, DW, ISWR,
ILWR, PSUM) update. HS evolves from the last-observed cluster value using
station ΔHS: positive increments (snowfall) are added uniformly; negative
increments (settlement/melt) are applied as a proportional ratio, preserving
the cluster's relative rank within the snowpack. This mirrors
spatial_model.gap_fill_station_only() for the scalar (single-cluster) case.

Design doc: docs/operational_mode_design_concept.md §2.1 and §6.1

Usage:
    python smet_append.py                      # append to all clusters
    python smet_append.py --cluster 3178       # single cluster
    python smet_append.py --until 2026-01-18   # stop at a specific date
    python smet_append.py --dry-run            # report gaps without writing

Called by run_daily.sh after aws_ingest.py confirms new data is available.
"""

from __future__ import annotations

import argparse
import re
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd


SMET_DIR_DEFAULT = Path(__file__).parent.parent.parent / "snowpack" / "little_prof" / "input"
WEATHER_CACHE_DIR = Path(__file__).parent.parent.parent / "outputs" / "weather"

# Primary met station used for per-cluster forcing (all clusters share the same
# met station; the spatial HS distribution comes from the UAS surveys, not from
# the station). If a second station is used for redundancy / gap-fill, add it
# here and handle merging in _load_met_for_period().
PRIMARY_STATION = "CAABT"


# ---------------------------------------------------------------------------
# SMET helpers
# ---------------------------------------------------------------------------

def _read_smet_last_timestamp(smet_path: Path) -> pd.Timestamp | None:
    """Return the last DATA timestamp in a SMET file, UTC-aware."""
    last_ts = None
    in_data = False
    with open(smet_path) as f:
        for line in f:
            line = line.strip()
            if line == "[DATA]":
                in_data = True
                continue
            if in_data and line and not line.startswith("#"):
                ts_str = line.split()[0]
                try:
                    ts = pd.Timestamp(ts_str, tz="UTC")
                    last_ts = ts
                except Exception:
                    pass
    return last_ts


def _read_smet_fields(smet_path: Path) -> list[str]:
    """Return the list of field names from the SMET header's `fields =` line."""
    with open(smet_path) as f:
        for line in f:
            m = re.match(r"\s*fields\s*=\s*(.+)", line)
            if m:
                return m.group(1).split()
    return []


def _read_smet_last_hs(smet_path: Path) -> float | None:
    """Return the last valid HS value (metres) written in a SMET file."""
    fields = _read_smet_fields(smet_path)
    if "HS" not in fields:
        return None
    hs_idx = fields.index("HS")
    last_hs = None
    in_data = False
    with open(smet_path) as f:
        for line in f:
            line = line.strip()
            if line == "[DATA]":
                in_data = True
                continue
            if in_data and line and not line.startswith("#"):
                parts = line.split()
                if len(parts) > hs_idx:
                    try:
                        val = float(parts[hs_idx])
                        if val >= 0:  # -999 nodata is negative
                            last_hs = val
                    except ValueError:
                        pass
    return last_hs


def _propagate_hs_station_only(hs_start: float,
                                stn_hs: pd.Series) -> pd.Series:
    """
    Propagate a cluster's HS forward using station ΔHS only.

    Mirrors spatial_model.gap_fill_station_only() for the scalar case:
    - Positive ΔHS (snowfall): add uniformly to cluster HS.
    - Negative ΔHS (settlement/melt): proportional scaling, preserving
      the cluster's relative rank within the snowpack.

    stn_hs must include the anchor row at last_ts as its first element so
    the first delta can be computed. Returns a Series indexed identically
    to stn_hs.
    """
    stn_vals = stn_hs.values
    current = float(hs_start)
    result = []

    for i in range(len(stn_vals)):
        if i == 0:
            result.append(current)
            continue
        stn_prev = stn_vals[i - 1]
        stn_curr = stn_vals[i]
        if np.isnan(stn_prev) or np.isnan(stn_curr):
            result.append(current)
            continue
        dhs = stn_curr - stn_prev
        if dhs >= 0:
            current = max(current + dhs, 0.0)
        else:
            if stn_prev > 1e-6:
                current = max(current * max(stn_curr / stn_prev, 0.0), 0.0)
        result.append(current)

    return pd.Series(result, index=stn_hs.index)


def _format_smet_row(ts: pd.Timestamp, row: dict, fields: list[str]) -> str:
    """Format one hourly row as a SMET data line."""
    parts = []
    for col in fields:
        if col == "timestamp":
            parts.append(ts.strftime("%Y-%m-%dT%H:%M"))
        else:
            val = row.get(col, -999)
            if pd.isna(val):
                parts.append("-999")
            elif col in ("TA",):
                # SMET stores temperature in Kelvin
                parts.append(f"{val + 273.15:.2f}")
            elif col in ("RH",):
                # SMET stores RH as fraction 0-1
                parts.append(f"{val / 100.0:.4f}")
            elif col in ("HS",):
                # SMET stores HS in metres
                parts.append(f"{val:.4f}")
            else:
                parts.append(f"{val:.4f}")
    return "\t".join(parts)


# ---------------------------------------------------------------------------
# Met data loader
# ---------------------------------------------------------------------------

def _load_met_for_period(since: pd.Timestamp,
                         until: pd.Timestamp,
                         station_id: str | None = None,
                         include_anchor: bool = False) -> pd.DataFrame:
    """Load met records from the AWS cache for (since, until].

    When include_anchor=True the row at exactly `since` is included so
    callers can compute the first ΔHS delta against the anchor station value.
    """
    sid = station_id or PRIMARY_STATION
    cache = WEATHER_CACHE_DIR / f"{sid}.parquet"
    if not cache.exists():
        print(f"    WARNING: weather cache not found at {cache} — "
              f"run aws_ingest.py first")
        return pd.DataFrame()

    df = pd.read_parquet(cache)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if include_anchor:
        mask = (df["timestamp"] >= since) & (df["timestamp"] <= until)
    else:
        mask = (df["timestamp"] > since) & (df["timestamp"] <= until)
    return df[mask].set_index("timestamp").sort_index()


# ---------------------------------------------------------------------------
# Core append logic
# ---------------------------------------------------------------------------

def append_cluster_smet(smet_path: Path,
                        until: pd.Timestamp,
                        dry_run: bool = False,
                        station_id: str | None = None) -> int:
    """Append new met rows to one cluster's SMET file.

    Returns the number of rows appended (0 if already current or dry-run).
    """
    last_ts = _read_smet_last_timestamp(smet_path)
    if last_ts is None:
        print(f"    WARNING: could not read last timestamp from {smet_path.name}")
        return 0

    if last_ts >= until:
        return 0  # already current

    fields = _read_smet_fields(smet_path)
    if not fields:
        print(f"    WARNING: could not parse fields from {smet_path.name}")
        return 0

    # Load met including the anchor row at last_ts so we can compute the
    # first ΔHS delta against the station value at that timestamp.
    met_full = _load_met_for_period(last_ts, until, station_id=station_id,
                                    include_anchor=True)
    met = met_full[met_full.index > last_ts]
    if met.empty:
        return 0

    if dry_run:
        print(f"    {smet_path.name}: would append {len(met)} rows "
              f"({last_ts} → {until})")
        return 0

    # Propagate cluster HS forward using station ΔHS (mirrors
    # spatial_model.gap_fill_station_only for the scalar case).
    # Falls back to the frozen last value when station HS is unavailable.
    last_hs = _read_smet_last_hs(smet_path)
    if last_hs is not None and "HS" in met_full.columns:
        hs_series = _propagate_hs_station_only(last_hs, met_full["HS"])
        hs_lookup = hs_series[hs_series.index > last_ts].to_dict()
    else:
        hs_lookup = {ts: last_hs for ts in met.index}

    lines = []
    for ts, row in met.iterrows():
        row_dict = row.to_dict()
        if "HS" in fields:
            row_dict["HS"] = hs_lookup.get(ts, last_hs)
        lines.append(_format_smet_row(ts, row_dict, fields))

    with open(smet_path, "a") as f:
        f.write("\n".join(lines) + "\n")

    return len(lines)


def append_all(smet_dir: Path,
               until: pd.Timestamp | None = None,
               dry_run: bool = False,
               station_id: str | None = None) -> dict[str, int]:
    """Append new rows to all SMET files in smet_dir."""
    if until is None:
        until = pd.Timestamp("now", tz="UTC").floor("h")

    smet_files = sorted(smet_dir.glob("cluster_*.smet"))
    if not smet_files:
        print(f"    No SMET files found in {smet_dir}")
        return {}

    total = 0
    updated = 0
    counts = {}
    for p in smet_files:
        n = append_cluster_smet(p, until=until, dry_run=dry_run, station_id=station_id)
        counts[p.stem] = n
        total += n
        if n:
            updated += 1

    print(f"    SMET append: {updated}/{len(smet_files)} files updated, "
          f"{total} total rows written")
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append new met records to per-cluster SMET files")
    parser.add_argument("--smet-dir", type=Path, default=SMET_DIR_DEFAULT,
                        help="Directory containing cluster_*.smet files")
    parser.add_argument("--cluster", type=int, default=None,
                        help="Single cluster ID to update")
    parser.add_argument("--until", default=None,
                        help="End timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM); "
                             "default: current hour UTC")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report gaps without writing")
    parser.add_argument("--project-dir", default=None,
                        help="Project directory; resolves slope_config.toml for station ID")
    parser.add_argument("--slope-name", default=None,
                        help="Slope name (used with --project-dir)")
    args = parser.parse_args()

    until = (pd.Timestamp(args.until, tz="UTC") if args.until
             else pd.Timestamp("now", tz="UTC").floor("h"))

    # Resolve primary station from config when slope is specified
    active_station_id = None
    if args.project_dir and args.slope_name:
        _toml = Path(args.project_dir) / "slopes" / args.slope_name / "slope_config.toml"
        if _toml.exists():
            import sys as _sys
            _sys.path.insert(0, str(Path(args.project_dir) / "src" / "avachain"))
            from config import ProjectConfig as _PC
            _cfg = _PC.from_toml(_toml)
            active_station_id = _cfg.summit_id or None

    if args.cluster is not None:
        pattern = f"cluster_{args.cluster:04d}_cluster_{args.cluster:04d}.smet"
        p = args.smet_dir / pattern
        if not p.exists():
            # try without zero-padding
            candidates = list(args.smet_dir.glob(f"*{args.cluster}*.smet"))
            if not candidates:
                raise SystemExit(f"No SMET file found for cluster {args.cluster} "
                                 f"in {args.smet_dir}")
            p = candidates[0]
        n = append_cluster_smet(p, until=until, dry_run=args.dry_run,
                                station_id=active_station_id)
        print(f"  {p.name}: {n} rows {'would be ' if args.dry_run else ''}appended")
    else:
        append_all(args.smet_dir, until=until, dry_run=args.dry_run,
                   station_id=active_station_id)


if __name__ == "__main__":
    main()
