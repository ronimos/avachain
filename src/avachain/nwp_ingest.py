"""
nwp_ingest.py — Extend cluster SMET files with CAIC WRF forecast rows.

NWP forecast mode (docs/operational_mode_design_concept.md §3):
after the daily forward step has advanced to today (T+0, observed data),
this module appends CAIC WRF forecast rows to the cluster SMET files so
SNOWPACK can run through T+48h and produce stability snapshots at T+0, T+1,
T+2 for the scenario ensemble.

Data source: the closest WRF grid point is a SNOWPACK-output SMET file updated
every 6 hours at /ssd/snowpack/fcst/<season>/<zone>/<grid_id>/<grid_id>2_res.smet.
The fields we extract are: TA (°C), RH (%), VW (m/s), DW (°), ISWR (W/m²),
ILWR (W/m²), MS_Snow (kg/m²/h).  All already in the file's stored units —
no multiplier/offset conversion needed because they match the cluster SMET
convention exactly.

T_stable = last 6h WRF tick ≤ now.  Cluster SMETs are truncated to T_stable,
then WRF rows (T_stable+1h … T_stable+forecast_hours) are appended flagged
with NWP_FLAG so they can be pruned and replaced on the next daily cycle
(rolling convergence, §3.3).

Lapse rate: WRF grid cell is at a fixed altitude; each cluster has its own
altitude read from its SMET header.  A dry-adiabatic lapse rate of
−6.5 K/km is applied to TA for the altitude difference.

HS in forecast rows is written as -999 (SNOWPACK nodata).  SNOWPACK source
(Snowpack.cc:1500) confirms that when hs == nodata, the previous simulated
mH is kept — no modification of the .ini file is needed.

Usage:
    python nwp_ingest.py --toml ../../slopes/little_prof/slope_config.toml
    python nwp_ingest.py --toml ... --lead-hours 48 --dry-run
    python nwp_ingest.py --toml ... --cluster 42

Called by run_<slope>.sh after the observed-data SMET append (smet_append.py).

TODO (future ensemble work):
  - Add HRRR / NAM / NBM as secondary NWP sources alongside CAIC WRF.
  - Weight ensemble members by past forecast skill.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Sentinel comment written before forecast rows so prune_nwp_rows() can find them.
NWP_FLAG = "# NWP_FORECAST CAIC_WRF"

# Lapse rate: −6.5 K per 1000 m
LAPSE_RATE_K_PER_M = -6.5e-3

# WRF forcing columns we extract (stored units match cluster SMET convention)
WRF_COLS = ["TA", "RH", "VW", "DW", "ISWR", "ILWR", "MS_Snow"]


# ---------------------------------------------------------------------------
# WRF SMET helpers
# ---------------------------------------------------------------------------

def get_stable_ts(wrf_smet_path: Path) -> pd.Timestamp | None:
    """Return the last timestamp in the WRF SMET file that is ≤ now (UTC)."""
    now = pd.Timestamp("now", tz="UTC")
    last: pd.Timestamp | None = None
    in_data = False
    with open(wrf_smet_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[DATA]":
                in_data = True
                continue
            if in_data and stripped and not stripped.startswith("#"):
                ts_str = stripped.split()[0]
                try:
                    ts = pd.Timestamp(ts_str, tz="UTC")
                    if ts <= now:
                        last = ts
                except Exception:
                    pass
    return last


def _parse_smet_header(smet_path: Path) -> dict:
    """Return dict with keys: altitude, fields, units_offset, units_multiplier."""
    info: dict = {}
    in_header = False
    with open(smet_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[HEADER]":
                in_header = True
                continue
            if stripped == "[DATA]":
                break
            if not in_header:
                continue
            m = re.match(r"(\w+)\s*=\s*(.+)", stripped)
            if not m:
                continue
            key, val = m.group(1).strip(), m.group(2).strip()
            if key == "altitude":
                info["altitude"] = float(val)
            elif key == "fields":
                info["fields"] = val.split()
            elif key == "units_offset":
                info["units_offset"] = [float(x) for x in val.split()]
            elif key == "units_multiplier":
                info["units_multiplier"] = [float(x) for x in val.split()]
    return info


def read_wrf_smet(wrf_smet_path: Path,
                  start_ts: pd.Timestamp,
                  end_ts: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """
    Read the WRF SMET file and return (DataFrame, wrf_altitude_m).

    DataFrame columns: TA (°C), RH (%), VW (m/s), DW (°), ISWR (W/m²),
    ILWR (W/m²), MS_Snow (kg/m²/h), indexed by UTC timestamp.
    The data span [start_ts, end_ts] inclusive.

    Units are exactly as stored in the file — no multiplier/offset applied —
    because the WRF file uses the same conventions as the cluster SMETs.
    """
    header = _parse_smet_header(wrf_smet_path)
    fields = header.get("fields", [])
    wrf_alt = header.get("altitude", float("nan"))

    wanted = {col: fields.index(col) for col in WRF_COLS if col in fields}
    ts_idx = fields.index("timestamp") if "timestamp" in fields else 0

    rows = []
    in_data = False
    with open(wrf_smet_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[DATA]":
                in_data = True
                continue
            if not in_data or not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                ts = pd.Timestamp(parts[ts_idx], tz="UTC")
            except Exception:
                continue
            if ts < start_ts or ts > end_ts:
                continue
            row: dict = {"timestamp": ts}
            for col, idx in wanted.items():
                raw = parts[idx] if idx < len(parts) else "-999"
                row[col] = float(raw)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, wrf_alt
    df = df.set_index("timestamp").sort_index()
    # Replace nodata (-999) with NaN for interpolation
    df = df.replace(-999.0, float("nan"))
    return df, wrf_alt


def _resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 6-hourly WRF data to 1-hourly.

    TA, RH, VW, ISWR, ILWR: linear interpolation.
    DW: circular interpolation via U/V components.
    MS_Snow: step (constant rate within each 6 h window).
    """
    if df.empty:
        return df

    # Build a full hourly index from first to last timestamp
    start = df.index[0]
    end = df.index[-1]
    hourly_idx = pd.date_range(start, end, freq="h", tz="UTC")

    # Re-index to hourly (inserts NaN for new hours)
    out = df.reindex(hourly_idx)

    # DW: circular interpolation via U/V unit vectors
    if "DW" in out.columns and "VW" in df.columns:
        dw_rad = np.radians(df["DW"].dropna())
        ux = np.sin(dw_rad)
        uy = np.cos(dw_rad)
        ux_s = pd.Series(ux.values, index=dw_rad.index).reindex(hourly_idx)
        uy_s = pd.Series(uy.values, index=dw_rad.index).reindex(hourly_idx)
        ux_i = ux_s.interpolate(method="time", limit_direction="forward")
        uy_i = uy_s.interpolate(method="time", limit_direction="forward")
        dw_out = np.degrees(np.arctan2(ux_i, uy_i)) % 360
        out["DW"] = dw_out

    # Linear interpolation for scalar met vars
    for col in ["TA", "RH", "VW", "ISWR", "ILWR"]:
        if col in out.columns:
            out[col] = out[col].interpolate(method="time", limit_direction="forward")

    # MS_Snow: hold constant within each 6 h block (forward-fill then back-fill)
    if "MS_Snow" in out.columns:
        ms = df["MS_Snow"].reindex(hourly_idx)
        out["MS_Snow"] = ms.ffill().bfill()

    # Physical bounds
    if "ISWR" in out.columns:
        out["ISWR"] = out["ISWR"].clip(lower=0.0)
    if "RH" in out.columns:
        out["RH"] = out["RH"].clip(0.0, 100.0)
    if "VW" in out.columns:
        out["VW"] = out["VW"].clip(lower=0.0)
    if "MS_Snow" in out.columns:
        out["MS_Snow"] = out["MS_Snow"].clip(lower=0.0)

    return out


def _apply_lapse_rate(df: pd.DataFrame,
                      wrf_alt_m: float,
                      cluster_alt_m: float) -> pd.DataFrame:
    """Apply −6.5 K/km lapse rate to TA for the altitude difference."""
    if "TA" not in df.columns or np.isnan(wrf_alt_m) or np.isnan(cluster_alt_m):
        return df
    delta_t = LAPSE_RATE_K_PER_M * (cluster_alt_m - wrf_alt_m)
    out = df.copy()
    out["TA"] = out["TA"] + delta_t
    return out


# ---------------------------------------------------------------------------
# SMET NWP-row management (truncate + append)
# ---------------------------------------------------------------------------

def _read_smet_altitude(smet_path: Path) -> float:
    """Read altitude (m) from SMET header; returns NaN if not found."""
    with open(smet_path) as f:
        for line in f:
            m = re.match(r"\s*altitude\s*=\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return float("nan")


def _read_smet_fields(smet_path: Path) -> list[str]:
    """Return field list from SMET header."""
    with open(smet_path) as f:
        for line in f:
            m = re.match(r"\s*fields\s*=\s*(.+)", line)
            if m:
                return m.group(1).split()
    return []


def _truncate_smet_to(smet_path: Path, at_ts: pd.Timestamp) -> int:
    """
    Keep all SMET rows with timestamp ≤ at_ts (inclusive) and discard the rest.
    Also strips any existing NWP_FLAG comment lines.
    Returns the number of data rows removed.
    """
    with open(smet_path) as f:
        lines = f.readlines()

    header_lines: list[str] = []
    data_lines: list[str] = []
    in_data = False
    for line in lines:
        stripped = line.strip()
        if stripped == "[DATA]":
            in_data = True
            header_lines.append(line)
            continue
        if not in_data:
            header_lines.append(line)
        else:
            data_lines.append(line)

    kept: list[str] = []
    removed = 0
    for line in data_lines:
        stripped = line.strip()
        if not stripped or NWP_FLAG in stripped or stripped.startswith("#"):
            # Drop NWP flags and standalone comments after [DATA]; blank lines dropped
            if NWP_FLAG in stripped:
                continue
            if stripped.startswith("#"):
                continue
            continue
        ts_str = stripped.split()[0]
        try:
            ts = pd.Timestamp(ts_str, tz="UTC")
        except Exception:
            kept.append(line)
            continue
        if ts <= at_ts:
            kept.append(line)
        else:
            removed += 1

    with open(smet_path, "w") as f:
        f.writelines(header_lines)
        f.writelines(kept)
    return removed


def _format_nwp_row(ts: pd.Timestamp, row: dict, fields: list[str]) -> str:
    """
    Format one forecast row as a SMET data line.

    Units written to file:
      TA  → Celsius (cluster SMET: units_offset=273.15, mult=1)
      RH  → percent  (cluster SMET: units_multiplier=0.01)
      VW, DW, ISWR, ILWR, MS_Snow → no conversion needed
      HS  → -999 (nodata; SNOWPACK keeps previous simulated mH)
      All other fields → -999
    """
    parts: list[str] = []
    for col in fields:
        if col == "timestamp":
            parts.append(ts.strftime("%Y-%m-%dT%H:%M"))
            continue
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            parts.append("-999")
            continue
        if col == "TA":
            parts.append(f"{val:.3f}")
        elif col == "RH":
            parts.append(f"{val:.3f}")
        elif col in ("VW", "DW", "ISWR", "ILWR", "MS_Snow"):
            parts.append(f"{val:.4f}")
        elif col == "HS":
            parts.append("-999")
        else:
            parts.append("-999")
    return "\t".join(parts)


def prune_nwp_rows(smet_path: Path) -> int:
    """Remove previously appended NWP forecast rows (identified by NWP_FLAG comment)."""
    with open(smet_path) as f:
        lines = f.readlines()

    clean: list[str] = []
    in_nwp_block = False
    removed = 0
    for line in lines:
        if NWP_FLAG in line:
            in_nwp_block = True
            continue
        if in_nwp_block:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                removed += 1
                continue
            else:
                in_nwp_block = False
        clean.append(line)

    if removed:
        with open(smet_path, "w") as f:
            f.writelines(clean)
    return removed


def append_nwp_rows(smet_path: Path,
                    forecast: pd.DataFrame,
                    dry_run: bool = False) -> int:
    """Append NWP forecast rows to one SMET file, flagged with NWP_FLAG."""
    if forecast.empty:
        return 0

    fields = _read_smet_fields(smet_path)
    if not fields:
        return 0

    # Find last existing timestamp so we don't write duplicate rows
    last_ts: pd.Timestamp | None = None
    in_data = False
    with open(smet_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[DATA]":
                in_data = True
                continue
            if in_data and stripped and not stripped.startswith("#"):
                try:
                    last_ts = pd.Timestamp(stripped.split()[0], tz="UTC")
                except Exception:
                    pass

    fc = forecast.sort_index()
    if last_ts is not None:
        fc = fc[fc.index > last_ts]
    if fc.empty:
        return 0

    if dry_run:
        print(f"    {smet_path.name}: would append {len(fc)} WRF rows "
              f"({fc.index[0]} → {fc.index[-1]})")
        return 0

    lines = [f"{NWP_FLAG}\n"]
    for ts, row in fc.iterrows():
        lines.append(_format_nwp_row(ts, row.to_dict(), fields) + "\n")

    with open(smet_path, "a") as f:
        f.writelines(lines)
    return len(fc)


def extend_all_smets(smet_dir: Path,
                     wrf_forecast_raw: pd.DataFrame,
                     wrf_alt_m: float,
                     stable_ts: pd.Timestamp,
                     dry_run: bool = False) -> dict[str, int]:
    """
    For each cluster SMET:
      1. Truncate to stable_ts (removes post-stable observed rows and stale NWP rows).
      2. Apply per-cluster lapse rate correction to TA.
      3. Append WRF forecast rows from stable_ts+1h … end of wrf_forecast_raw.

    Returns {smet_stem: n_rows_appended}.
    """
    smet_files = sorted(smet_dir.glob("cluster_*.smet"))
    if not smet_files:
        print(f"    No SMET files in {smet_dir}")
        return {}

    counts: dict[str, int] = {}
    total_appended = 0
    for p in smet_files:
        cluster_alt = _read_smet_altitude(p)
        if np.isnan(cluster_alt):
            print(f"    WARNING: no altitude in {p.name} — skipping lapse rate")
            fc = wrf_forecast_raw.copy()
        else:
            fc = _apply_lapse_rate(wrf_forecast_raw, wrf_alt_m, cluster_alt)

        # Filter to rows strictly after stable_ts
        fc = fc[fc.index > stable_ts]

        if not dry_run:
            _truncate_smet_to(p, stable_ts)
        n = append_nwp_rows(p, fc, dry_run=dry_run)
        counts[p.stem] = n
        total_appended += n

    action = "would append" if dry_run else "appended"
    print(f"    NWP extend: {len(smet_files)} SMETs — "
          f"{action} ~{total_appended // max(len(smet_files), 1)} rows each "
          f"(T_stable={stable_ts})")
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend cluster SMET files with CAIC WRF forecast rows")
    parser.add_argument("--toml", default=None,
                        help="Path to slope_config.toml (resolves WRF SMET path and smet_dir)")
    parser.add_argument("--smet-dir", type=Path, default=None,
                        help="Override directory containing cluster_*.smet files")
    parser.add_argument("--wrf-smet", type=Path, default=None,
                        help="Override WRF SMET file path")
    parser.add_argument("--lead-hours", type=int, default=48,
                        help="Forecast horizon in hours beyond T_stable (default: 48)")
    parser.add_argument("--cluster", type=int, default=None,
                        help="Single cluster ID to update (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without modifying files")
    args = parser.parse_args()

    # Resolve paths from TOML config when provided
    cfg = None
    if args.toml:
        toml_path = Path(args.toml)
        sys.path.insert(0, str(toml_path.parent.parent.parent / "src" / "avachain"))
        from config import ProjectConfig
        cfg = ProjectConfig.from_toml(toml_path)

    wrf_smet_path = (
        Path(args.wrf_smet) if args.wrf_smet
        else Path(cfg.wrf_smet_file) if cfg and cfg.wrf_smet_file
        else None
    )
    if wrf_smet_path is None or not wrf_smet_path.exists():
        print(f"ERROR: WRF SMET file not found: {wrf_smet_path}")
        print("  Provide --toml or --wrf-smet, or confirm /ssd is mounted.")
        sys.exit(1)

    smet_dir = (
        args.smet_dir if args.smet_dir
        else cfg.smet_dir if cfg
        else None
    )
    if smet_dir is None:
        print("ERROR: no SMET directory — provide --toml or --smet-dir")
        sys.exit(1)

    # Determine T_stable
    stable_ts = get_stable_ts(wrf_smet_path)
    if stable_ts is None:
        print("ERROR: could not determine T_stable from WRF SMET (no timestamps ≤ now)")
        sys.exit(1)
    print(f"  T_stable = {stable_ts}")

    # Read WRF data from T_stable to T_stable + lead_hours
    end_ts = stable_ts + pd.Timedelta(hours=args.lead_hours)
    wrf_raw, wrf_alt = read_wrf_smet(wrf_smet_path, stable_ts, end_ts)
    if wrf_raw.empty:
        print(f"  WARNING: no WRF data in [{stable_ts}, {end_ts}] — nothing to do")
        return

    wrf_hourly = _resample_to_hourly(wrf_raw)
    print(f"  WRF: {len(wrf_raw)} 6h ticks → {len(wrf_hourly)} hourly rows "
          f"({wrf_alt:.0f} m altitude)")

    if args.cluster is not None:
        candidates = list(smet_dir.glob(f"*{args.cluster:04d}*.smet"))
        if not candidates:
            candidates = list(smet_dir.glob(f"*{args.cluster}*.smet"))
        if not candidates:
            print(f"ERROR: no SMET file found for cluster {args.cluster} in {smet_dir}")
            sys.exit(1)
        p = candidates[0]
        cluster_alt = _read_smet_altitude(p)
        fc = _apply_lapse_rate(wrf_hourly, wrf_alt, cluster_alt)
        fc = fc[fc.index > stable_ts]
        if not args.dry_run:
            _truncate_smet_to(p, stable_ts)
        n = append_nwp_rows(p, fc, dry_run=args.dry_run)
        action = "would write" if args.dry_run else "wrote"
        print(f"  {p.name}: {action} {n} rows")
    else:
        extend_all_smets(smet_dir, wrf_hourly, wrf_alt, stable_ts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
