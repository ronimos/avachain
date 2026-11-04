#!/usr/bin/env python3
"""wrf_archive.py — keep every WRF-SNOWPACK forecast cycle for a slope's grid cell.

    python wrf_archive.py --toml slopes/Professor_PTC/slope_config.toml \
                          --archive-dir /home/avachain/work/weather/wrf_archive

The CAIC WRF-SNOWPACK run (/ssd/snowpack/fcst_snowpack.sh, about 07:40 local
each day) deletes /ssd/snowpack/fcst/<season>/ and rebuilds it. Each cell's
*_res.smet holds the season so far plus about 3.5 days of forecast, so the
forecast a cycle actually issued is overwritten by the next one. 2025-26 has
no per-cycle archive, so no forecast-skill claim is possible for it
(CLAUDE.md, "Replay clock"). This keeps one snapshot per cycle:

    <archive-dir>/<season>/<zone>/<cell>/<YYYYMMDDTHHMMZ>/   all files of the cell
                                                          + manifest.json

The stamp is the UTC mtime of the cell's newest *_res.smet, i.e. when that
cycle was written. A run does nothing when:
  - the newest file is younger than --min-age-min (the rebuild may still be
    writing; it takes ~15 min for all zones), or
  - the *_res.smet files are byte-identical to the latest snapshot.
So it is safe to run as often as hourly. Stdlib only, no project imports, so
a pinned copy can run from cron before a full deploy.

The cell comes from the slope TOML's [wrf] fcst_base_dir / wrf_zone /
wrf_grid_id (install.py writes them), in the newest season folder that has
it; --cell-dir names it directly instead.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

STAMP_FMT = "%Y%m%dT%H%MZ"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_dir_from_toml(toml_path: Path) -> Path:
    """The slope's grid cell in the newest season folder that has it."""
    wrf = tomllib.loads(Path(toml_path).read_text()).get("wrf", {})
    base, zone, grid = wrf.get("fcst_base_dir"), wrf.get("wrf_zone"), wrf.get("wrf_grid_id")
    if not (base and zone and grid):
        raise ValueError(f"{toml_path}: [wrf] needs fcst_base_dir, wrf_zone and "
                         f"wrf_grid_id (run install.py)")
    seasons = sorted((d for d in Path(base).iterdir()
                      if d.name.isdigit() and (d / zone / str(grid)).is_dir()),
                     key=lambda d: int(d.name))
    if not seasons:
        raise FileNotFoundError(f"no season folder under {base} has {zone}/{grid}")
    return seasons[-1] / zone / str(grid)


def _data_span(smet: Path):
    """(first, last) data timestamps of a SMET file, or (None, None)."""
    first = last = None
    in_data = False
    with open(smet) as f:
        for line in f:
            if not in_data:
                in_data = line.strip() == "[DATA]"
                continue
            t = line.split(None, 1)[0] if line.strip() else None
            if t:
                first = first or t
                last = t
    return first, last


def archive_cell(cell: Path, archive_dir: Path, min_age_min: float = 10.0,
                 now: dt.datetime | None = None) -> Path | None:
    """Snapshot one cell. Returns the new snapshot dir, or None if skipped."""
    cell = Path(cell)
    smets = sorted(cell.glob("*_res.smet"))
    if not smets:
        raise FileNotFoundError(f"no *_res.smet in {cell}")
    now = now or dt.datetime.now(dt.timezone.utc)
    newest = max(p.stat().st_mtime for p in cell.iterdir() if p.is_file())
    age_min = (now.timestamp() - newest) / 60.0
    if age_min < min_age_min:
        print(f"{cell}: written {age_min:.1f} min ago, may still be in progress — skipped")
        return None

    season, zone, grid = cell.parts[-3:]
    dest_root = Path(archive_dir) / season / zone / grid
    shas = {p.name: _sha256(p) for p in smets}
    prev = sorted(d for d in dest_root.glob("*Z") if (d / "manifest.json").exists()) \
        if dest_root.exists() else []
    if prev:
        last = json.loads((prev[-1] / "manifest.json").read_text())
        if {k: v["sha256"] for k, v in last["files"].items() if k.endswith("_res.smet")} == shas:
            print(f"{cell}: unchanged since {prev[-1].name} — skipped")
            return None

    stamp = dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in smets),
                                      dt.timezone.utc).strftime(STAMP_FMT)
    dest = dest_root / stamp
    if dest.exists():
        print(f"{cell}: {dest} already exists — skipped")
        return None
    tmp = dest_root / f".{stamp}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files = {}
    for p in sorted(cell.iterdir()):
        if p.is_file():
            shutil.copy2(p, tmp / p.name)
            files[p.name] = {"bytes": p.stat().st_size, "sha256": _sha256(tmp / p.name)}
    first, last = _data_span(smets[0])
    manifest = {
        "source": str(cell),
        "cycle_written_utc": stamp,
        "archived_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_first": first, "data_last": last,
        "files": files,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.rename(tmp, dest)                      # a reader never sees a partial snapshot
    print(f"{cell}: archived {len(files)} files -> {dest} (data to {last})")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--toml", nargs="+", type=Path, help="slope TOML(s)")
    src.add_argument("--cell-dir", nargs="+", type=Path, help="grid-cell folder(s)")
    ap.add_argument("--archive-dir", type=Path, required=True)
    ap.add_argument("--min-age-min", type=float, default=10.0)
    args = ap.parse_args(argv)
    rc = 0
    cells = list(args.cell_dir or [])
    for t in args.toml or []:
        try:
            cells.append(cell_dir_from_toml(t))
        except (OSError, ValueError) as e:
            print(f"ERROR {t}: {e}", file=sys.stderr)
            rc = 1
    for cell in dict.fromkeys(cells):          # two slopes may share a cell
        try:
            archive_cell(cell, args.archive_dir, args.min_age_min)
        except (OSError, ValueError) as e:
            print(f"ERROR {cell}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
