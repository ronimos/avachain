"""
nwp_ingest.py — Per-cluster NWP forecast forcing from the CAIC WRF run.

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

T_stable = last 6h WRF tick ≤ now. For each cluster a SEPARATE forecast SMET,
<slope_dir>/input/smet_forecast/<cid>.smet, is written from T_stable to
T_stable + lead hours. The observed SMETs are never touched: they used to be
truncated at T_stable and extended with flagged WRF rows. That threw away the
observations after T_stable, and it is how yesterday's forecast came to be
read as today's observations (0d83a1e).

The forecast pass runs SNOWPACK precipitation-driven
(slopes/<slope>/config/master_config_forecast.ini sets
ENFORCE_MEASURED_SNOW_HEIGHTS = FALSE), with PSUM from WRF. The WRF file is a
SNOWPACK result file: MS_Snow and MS_Rain are its solid and liquid
precipitation RATES (kg m-2 h-1). On each 6 h row the rate covers the
PRECEDING six hours. Integrated that way it matches the file's own HNW24
(Nov-Apr: r=0.81, wet-day ratio 1.01, MAE 0.9 mm). Held forward, as this
module used to do, gives r=0.66 and places snowfall six hours late.

Lapse rate: WRF grid cell is at a fixed altitude; each cluster has its own
altitude read from its SMET header.  A dry-adiabatic lapse rate of
−6.5 K/km is applied to TA for the altitude difference.

HS in forecast rows is nodata. Under the observed config
(ENFORCE_MEASURED_SNOW_HEIGHTS = TRUE) SNOWPACK stops at the first such step
with "[E] ... missing { HS }" while still printing FINISHED. That is why the
forecast pass never ran before, and why it needs its own config. An earlier
note here claimed nodata HS needed no .ini change; that was wrong. MS_Snow
was also written into the cluster SMETs' MS_Snow column, which SNOWPACK reads
as an output field, so no forecast precipitation ever reached the model.

Usage:
    python nwp_ingest.py --toml ../../slopes/little_prof/slope_config.toml
    python nwp_ingest.py --toml ... --lead-hours 48 --dry-run
    python nwp_ingest.py --toml ... --cluster 42

Called by run_<slope>.sh / run_daily_<slope>.sh after smet_append.py.

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

from config import replay_as_of


# Sentinel comment written before forecast rows so prune_nwp_rows() can find them.
NWP_FLAG = "# NWP_FORECAST CAIC_WRF"

# Lapse rate: −6.5 K per 1000 m
LAPSE_RATE_K_PER_M = -6.5e-3

# WRF forcing columns we extract (stored units match cluster SMET convention)
WRF_COLS = ["TA", "RH", "VW", "DW", "ISWR", "ILWR", "MS_Snow", "MS_Rain"]

# Forecast SMET layout: what SNOWPACK reads in precipitation-driven mode.
# Forecast rows before T_stable, so the first step's PSUM accumulation has data.
LEAD_IN_HOURS = 6
FC_FIELDS = ["timestamp", "TA", "RH", "VW", "DW", "ISWR", "ILWR", "PSUM", "HS"]
FC_OFFSET = [0, 273.15, 0, 0, 0, 0, 0, 0, 0]          # TA in degC
FC_MULT   = [1, 1, 0.01, 1, 1, 1, 1, 1, 0.01]         # RH in %, HS in cm


# ---------------------------------------------------------------------------
# WRF SMET helpers
# ---------------------------------------------------------------------------

def get_stable_ts(wrf_smet_path: Path, now=None) -> pd.Timestamp | None:
    """Return the last timestamp in the WRF SMET file that is ≤ now (UTC).

    `now` defaults to the replay clock ($AVACHAIN_AS_OF) when set, else the
    wall clock. Replay caveat: this season has no per-cycle WRF archive, only
    one rolling file, so rows after a historical T_stable were written later
    than that date and are not the forecast that was actually issued then.
    """
    now = replay_as_of(now)
    now = pd.Timestamp(now, tz="UTC") if now else pd.Timestamp("now", tz="UTC")
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
    PSUM: kg m-2 per hour = (MS_Snow + MS_Rain) rate, held BACKWARD: the
          rate on a 6 h row covers the six hours before it.
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

    # PSUM per hour from the precipitation rates, each 6 h rate applied to the
    # hours (t-6h, t] that precede it, i.e. back-filled from the next row.
    rate = sum(df[c].fillna(0.0) for c in ("MS_Snow", "MS_Rain") if c in df.columns)
    if not isinstance(rate, int):
        out["PSUM"] = rate.reindex(hourly_idx).bfill().clip(lower=0.0)
    out = out.drop(columns=[c for c in ("MS_Snow", "MS_Rain") if c in out.columns])

    # Physical bounds
    if "ISWR" in out.columns:
        out["ISWR"] = out["ISWR"].clip(lower=0.0)
    if "RH" in out.columns:
        out["RH"] = out["RH"].clip(0.0, 100.0)
    if "VW" in out.columns:
        out["VW"] = out["VW"].clip(lower=0.0)
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
# Forecast SMETs
# ---------------------------------------------------------------------------

_HEADER_KEYS = ("station_id", "station_name", "latitude", "longitude", "altitude",
                "slope_angle", "slope_azi", "easting", "northing", "epsg")


def _read_smet_header_kv(smet_path: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    with open(smet_path) as f:
        for line in f:
            if line.strip() == "[DATA]":
                break
            m = re.match(r"\s*(\w+)\s*=\s*(.+)", line)
            if m:
                kv[m.group(1)] = m.group(2).strip()
    return kv


def _read_smet_altitude(smet_path: Path) -> float:
    """Read altitude (m) from SMET header; returns NaN if not found."""
    v = _read_smet_header_kv(smet_path).get("altitude")
    return float(v) if v is not None else float("nan")


def _format_fc_row(ts: pd.Timestamp, row: dict) -> str:
    parts = [ts.strftime("%Y-%m-%dT%H:%M")]
    for col in FC_FIELDS[1:]:
        v = row.get(col)
        if col == "HS" or v is None or (isinstance(v, float) and np.isnan(v)):
            parts.append("-999")
        else:
            parts.append(f"{v:.4f}")
    return "\t".join(parts)


def write_forecast_smet(obs_smet: Path, out_path: Path, fc: pd.DataFrame) -> int:
    """Write one cluster's forecast SMET: its station header, WRF rows, HS nodata."""
    kv = _read_smet_header_kv(obs_smet)
    head = ["SMET 1.1 ASCII", "[HEADER]"]
    head += [f"{k:<16} = {kv[k]}" for k in _HEADER_KEYS if k in kv]
    head += ["nodata           = -999", "tz               = 0",
             "units_offset     = " + " ".join(str(x) for x in FC_OFFSET),
             "units_multiplier = " + " ".join(str(x) for x in FC_MULT),
             "fields           = " + " ".join(FC_FIELDS),
             "comment          = NWP forecast forcing (CAIC WRF); precipitation-driven",
             "[DATA]"]
    rows = [_format_fc_row(ts, r.to_dict()) for ts, r in fc.sort_index().iterrows()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(head + rows) + "\n")
    return len(rows)


def write_all_forecast_smets(obs_dir: Path, fc_dir: Path, wrf_hourly: pd.DataFrame,
                             wrf_alt_m: float, stable_ts: pd.Timestamp,
                             only: list[Path] | None = None) -> int:
    """One forecast SMET per observed SMET, lapse-rate corrected.

    Rows start LEAD_IN_HOURS before T_stable. PSUM is resampled with ACCUMULATE
    over the 15 min before each step, so the restart step itself needs earlier
    precipitation data. Starting exactly at T_stable made SNOWPACK stop with
    "missing { precipitation }" at its first step. The lead-in rows carry the
    T_stable rate, which in the backward convention is exactly the rate for
    those hours. Stale files from an earlier cycle are replaced, not appended
    to.
    """
    files = only if only is not None else sorted(obs_dir.glob("cluster_*.smet"))
    if not files:
        print(f"    No SMET files in {obs_dir}")
        return 0
    fc_all = wrf_hourly[wrf_hourly.index >= stable_ts - pd.Timedelta(hours=LEAD_IN_HOURS)]
    n_rows = 0
    for p in files:
        alt = _read_smet_altitude(p)
        fc = fc_all if np.isnan(alt) else _apply_lapse_rate(fc_all, wrf_alt_m, alt)
        n_rows = write_forecast_smet(p, fc_dir / p.name, fc)
    print(f"    NWP forecast SMETs: {len(files)} written to {fc_dir} "
          f"({n_rows} hourly rows each from T_stable={stable_ts})")
    return len(files)


def prune_nwp_rows(smet_path: Path) -> int:
    """Remove NWP rows an older version appended to an OBSERVED SMET."""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write per-cluster NWP forecast SMETs from the CAIC WRF run")
    parser.add_argument("--toml", default=None,
                        help="Path to slope_config.toml (resolves WRF SMET path and smet_dir)")
    parser.add_argument("--smet-dir", type=Path, default=None,
                        help="Override directory containing cluster_*.smet files")
    parser.add_argument("--forecast-dir", type=Path, default=None,
                        help="Override output directory for forecast SMETs")
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
    wrf_raw, wrf_alt = read_wrf_smet(
        wrf_smet_path, stable_ts - pd.Timedelta(hours=LEAD_IN_HOURS), end_ts)
    if wrf_raw.empty:
        print(f"  WARNING: no WRF data in [{stable_ts}, {end_ts}] — nothing to do")
        return

    wrf_hourly = _resample_to_hourly(wrf_raw)
    print(f"  WRF: {len(wrf_raw)} 6h ticks → {len(wrf_hourly)} hourly rows "
          f"({wrf_alt:.0f} m altitude)")

    fc_dir = args.forecast_dir or (cfg.smet_forecast_dir if cfg else None)
    if fc_dir is None:
        print("ERROR: no forecast SMET directory — provide --toml or --forecast-dir")
        sys.exit(1)
    if args.cluster is not None:
        only = (list(smet_dir.glob(f"cluster_{args.cluster:04d}.smet"))
                or list(smet_dir.glob(f"*{args.cluster}*.smet")))
        if not only:
            print(f"ERROR: no SMET file found for cluster {args.cluster} in {smet_dir}")
            sys.exit(1)
    else:
        only = None
    if args.dry_run:
        n = len(only) if only is not None else len(list(smet_dir.glob("cluster_*.smet")))
        print(f"  would write {n} forecast SMETs to {fc_dir}")
        return
    write_all_forecast_smets(smet_dir, fc_dir, wrf_hourly, wrf_alt, stable_ts, only=only)

if __name__ == "__main__":
    main()
