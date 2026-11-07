"""dem_check.py — check a new ground DEM before the pipeline is pointed at it.

    python src/avachain/dem_check.py --toml slopes/Professor_PTC/slope_config.toml \
           --dem NEW.tif [--compare OLD.tif ...] [--path-dem RUNOUT.tif] \
           [--road ROAD.shp] [--json out.json]

Written 2026-11-05 for the fall-2026 GCP snow-free DEM (GO_LIVE_CHECKLIST §1),
which replaces both last season's snowpack DEM (251110) and runout DEM (260605).
One surface serves both parts, so it must cover the snowpack domain and the
whole path to the road. Everything is resampled (average) onto one 1 m grid in
EPSG:6342 over the union of the inputs, and checked there:

  FAIL  horizontal CRS is not EPSG:6342 (config.RASTER_EPSG). A compound
        CRS (6342 + a vertical) is read by its horizontal part.
  FAIL  coverage below --min-coverage (default 0.995) of valid cells in the
        boundary KML, the start-zone KML, the runout path footprint (the
        valid cells of --path-dem, default the runout profile's terrain.dem)
        or the road centreline inside that footprint (sampled every 1 m).
  WARN  the heights are not documented as ellipsoidal. Since 2026-11-07
        (Ron, option 1) the whole DEM series is NAD83(2011) ellipsoidal
        heights with no geoid: a 3DEP check put 261018 and 260605 at
        −12.24 / −12.32 m from NAVD88, where GEOID18 N = −12.42 m. A
        conforming DEM is tagged horizontal-only EPSG:6342 with a
        VERTICAL_REFERENCE metadata tag saying "ellipsoidal". Any vertical
        CRS in the tag (last season's files say NAVD88, falsely) warns.
  WARN  a --compare DEM differs by more than VERTICAL_MISMATCH_M (median):
        about the geoid separation here, so the two are in different
        vertical frames, and snow height between them would be ~12 m off.
  info  sha256, size, resolution, nodata; per --compare DEM the elevation
        difference new − old (median, p05, p95) and the median |slope
        difference| in the start zone and the boundary. Last season's two
        surfaces differ by a median ~1.1 m; the report's DEM caveat quotes
        that number, so this is what to rewrite it from.

Exit 0 when nothing FAILs, 1 otherwise. Reads only; writes the --json report.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from pyproj import CRS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RASTER_EPSG  # noqa: E402

GRID_M = 1.0
VERTICAL_MISMATCH_M = 5.0   # GEOID18 N at the site is -12.42 m
MIN_COVERAGE = 0.995


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def crs_parts(crs) -> dict:
    """Horizontal EPSG and vertical datum name of a (possibly compound) CRS."""
    c = CRS.from_wkt(crs.to_wkt())
    horiz, vert = c, None
    if c.is_compound:
        horiz = next((s for s in c.sub_crs_list if not s.is_vertical), c)
        vert = next((s for s in c.sub_crs_list if s.is_vertical), None)
    return {"name": c.name, "horizontal_epsg": horiz.to_epsg(),
            "vertical": vert.name if vert is not None else None}


def describe(path: Path) -> dict:
    with rasterio.open(path) as s:
        out = {"path": str(path), "bytes": path.stat().st_size,
               "shape": list(s.shape), "res_m": [float(s.res[0]), float(s.res[1])],
               "dtype": s.dtypes[0], "nodata": s.nodata,
               "bounds": [float(b) for b in s.bounds]}
        out.update(crs_parts(s.crs))
        out["vertical_reference"] = s.tags().get("VERTICAL_REFERENCE")
    return out


def union_grid(paths: list[Path], res: float = GRID_M):
    """A res-metre grid over the union of the rasters' bounds."""
    bs = []
    for p in paths:
        with rasterio.open(p) as s:
            bs.append(s.bounds)
    left = np.floor(min(b.left for b in bs) / res) * res
    top = np.ceil(max(b.top for b in bs) / res) * res
    right = np.ceil(max(b.right for b in bs) / res) * res
    bottom = np.floor(min(b.bottom for b in bs) / res) * res
    shape = (int(round((top - bottom) / res)), int(round((right - left) / res)))
    return shape, from_origin(left, top, res, res)


def to_grid(path: Path, shape, transform) -> np.ndarray:
    """The DEM averaged onto the grid, NaN where it has no data."""
    out = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(path) as s:
        # Warp in the horizontal CRS: a compound source with a 2-D target
        # would ask PROJ for a vertical transformation it does not need.
        src_crs = CRS.from_epsg(crs_parts(s.crs)["horizontal_epsg"] or RASTER_EPSG)
        reproject(rasterio.band(s, 1), out, src_nodata=s.nodata, dst_nodata=np.nan,
                  src_crs=src_crs.to_wkt(), dst_transform=transform,
                  dst_crs=f"EPSG:{RASTER_EPSG}", resampling=Resampling.average)
    return out


def slope_deg(z: np.ndarray, res: float = GRID_M) -> np.ndarray:
    gy, gx = np.gradient(z, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def coverage(valid: np.ndarray, mask: np.ndarray) -> float | None:
    n = int(mask.sum())
    return None if n == 0 else float(valid[mask].sum() / n)


def road_points(road_path: Path, step: float = GRID_M) -> np.ndarray:
    """(x, y) every `step` metres along the road centreline, EPSG:6342."""
    import geopandas as gpd
    g = gpd.read_file(road_path).to_crs(epsg=RASTER_EPSG)
    pts = []
    for geom in g.geometry:
        for line in getattr(geom, "geoms", [geom]):
            for d in np.arange(0.0, line.length + step / 2, step):
                p = line.interpolate(min(d, line.length))
                pts.append((p.x, p.y))
    return np.array(pts).reshape(-1, 2)


def cells_of(xy: np.ndarray, transform, shape) -> tuple[np.ndarray, np.ndarray]:
    inv = ~transform
    cols, rows = inv * (xy[:, 0], xy[:, 1])
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return rows[ok], cols[ok]


def diff_stats(new: np.ndarray, old: np.ndarray, mask: np.ndarray) -> dict:
    both = mask & np.isfinite(new) & np.isfinite(old)
    if not both.any():
        return {"cells": 0}
    d = (new - old)[both]
    ds = np.abs(slope_deg(new) - slope_deg(old))[both]
    ds = ds[np.isfinite(ds)]
    return {"cells": int(both.sum()),
            "dz_median_m": round(float(np.median(d)), 3),
            "dz_p05_m": round(float(np.percentile(d, 5)), 3),
            "dz_p95_m": round(float(np.percentile(d, 95)), 3),
            "slope_absdiff_median_deg": round(float(np.median(ds)), 2) if ds.size else None}


def check(dem: Path, boundary_kml: Path | None, start_zone_kml: Path | None,
          path_dem: Path | None = None, road: Path | None = None,
          compare: list[Path] = (), min_coverage: float = MIN_COVERAGE,
          with_sha: bool = True) -> dict:
    from snowpack_io import kml_to_mask

    rep = {"dem": describe(dem), "fail": [], "warn": [], "coverage": {},
           "compare": {}}
    if with_sha:
        rep["dem"]["sha256"] = sha256(dem)
    if rep["dem"]["horizontal_epsg"] != RASTER_EPSG:
        rep["fail"].append(f"horizontal CRS is EPSG:{rep['dem']['horizontal_epsg']}, "
                           f"not EPSG:{RASTER_EPSG}")
        return rep  # nothing below is meaningful on another grid
    v, note = rep["dem"]["vertical"], rep["dem"]["vertical_reference"]
    if v is not None:
        rep["warn"].append(f"tagged vertical CRS {v}; the series is NAD83(2011) "
                           "ellipsoidal heights (no geoid). Check it against 3DEP: a "
                           "real geoid surface sits ~12.4 m above, a mislabel ~0")
    elif not note or "ellipsoid" not in note.lower():
        rep["warn"].append("no VERTICAL_REFERENCE tag saying the heights are "
                           "ellipsoidal (the series' frame since 2026-11-07)")

    others = [p for p in [path_dem, *compare] if p is not None]
    shape, tf = union_grid([dem, *others])
    z = to_grid(dem, shape, tf)
    valid = np.isfinite(z)
    rep["grid"] = {"res_m": GRID_M, "shape": list(shape)}

    masks = {}
    for name, kml in (("boundary", boundary_kml), ("start_zone", start_zone_kml)):
        if kml is not None:
            masks[name] = kml_to_mask(Path(kml), shape, tf)
            rep["coverage"][name] = coverage(valid, masks[name])
    path_valid = None
    if path_dem is not None:
        path_valid = np.isfinite(to_grid(path_dem, shape, tf))
        rep["coverage"]["path_footprint"] = coverage(valid, path_valid)
        rep["path_dem"] = str(path_dem)
    if road is not None:
        r, c = cells_of(road_points(road), tf, shape)
        if path_valid is not None:  # only the road the path reaches
            keep = path_valid[r, c]
            r, c = r[keep], c[keep]
        rep["coverage"]["road"] = None if r.size == 0 else float(valid[r, c].mean())
        rep["road_points"] = int(r.size)
    for name, cov in rep["coverage"].items():
        if cov is None:
            rep["fail"].append(f"{name}: no cells to check (empty mask?)")
        elif cov < min_coverage:
            rep["fail"].append(f"{name}: {cov:.4f} of cells have data "
                               f"(< {min_coverage})")

    for p in compare:
        zo = to_grid(p, shape, tf)
        rep["compare"][Path(p).name] = {
            name: diff_stats(z, zo, m) for name, m in masks.items()}
        meds = [s["dz_median_m"] for s in rep["compare"][Path(p).name].values()
                if s.get("cells")]
        if meds and max(abs(m) for m in meds) > VERTICAL_MISMATCH_M:
            rep["warn"].append(f"{Path(p).name}: median dz {meds[0]:+.2f} m, more than "
                               f"{VERTICAL_MISMATCH_M:g} m: different vertical frames "
                               "(geoid vs ellipsoid?)")
    return rep


def runout_profile_dem(toml_path: Path) -> Path | None:
    """terrain.dem of the slope's runout path profile, if it has one."""
    import tomllib
    import yaml
    rc = tomllib.loads(Path(toml_path).read_text()).get("runout") or {}
    if not rc.get("path_profile"):
        return None
    prof = (Path(__file__).resolve().parents[2] / "runout" / "config" / "paths"
            / f"{rc['path_profile']}.yaml")
    if not prof.exists():
        return None
    dem = (yaml.safe_load(prof.read_text()).get("terrain") or {}).get("dem")
    return Path(dem) if dem else None


def runout_profile_road(toml_path: Path) -> Path | None:
    import tomllib
    import yaml
    rc = tomllib.loads(Path(toml_path).read_text()).get("runout") or {}
    prof = (Path(__file__).resolve().parents[2] / "runout" / "config" / "paths"
            / f"{rc.get('path_profile', '')}.yaml")
    if not rc.get("path_profile") or not prof.exists():
        return None
    road = (yaml.safe_load(prof.read_text()).get("infrastructure") or {}).get("road")
    return Path(road) if road else None


def print_report(rep: dict) -> None:
    d = rep["dem"]
    print(f"DEM      {d['path']}")
    print(f"  sha256 {d.get('sha256', '-')}")
    print(f"  {d['shape'][0]} x {d['shape'][1]} cells at {d['res_m'][0]:g} m, "
          f"{d['dtype']}, nodata {d['nodata']}, {d['bytes'] / 1e6:.0f} MB")
    print(f"  CRS    {d['name']} (horizontal EPSG:{d['horizontal_epsg']}, "
          f"vertical {d['vertical']})")
    if d.get("vertical_reference"):
        print(f"  height {d['vertical_reference']}")
    if rep["coverage"]:
        print("Coverage (fraction of cells with data, 1 m grid):")
        for k, v in rep["coverage"].items():
            extra = f" ({rep['road_points']} points)" if k == "road" else ""
            print(f"  {k:15s} {'-' if v is None else f'{v:.4f}'}{extra}")
    for name, regions in rep["compare"].items():
        print(f"Against {name} (new − old):")
        for reg, s in regions.items():
            if not s.get("cells"):
                print(f"  {reg:15s} no overlap")
                continue
            print(f"  {reg:15s} dz median {s['dz_median_m']:+.3f} m "
                  f"[p05 {s['dz_p05_m']:+.3f}, p95 {s['dz_p95_m']:+.3f}], "
                  f"|d slope| median {s['slope_absdiff_median_deg']}°, "
                  f"{s['cells']} cells")
    for w in rep["warn"]:
        print(f"WARN  {w}")
    for f in rep["fail"]:
        print(f"FAIL  {f}")
    print("RESULT", "FAIL" if rep["fail"] else "PASS")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--toml", required=True, type=Path)
    ap.add_argument("--dem", required=True, type=Path, help="the DEM to check")
    ap.add_argument("--compare", nargs="*", type=Path, default=[],
                    help="DEMs to difference against (e.g. 251110 and 260605)")
    ap.add_argument("--path-dem", type=Path, default=None,
                    help="raster whose valid cells are the runout path footprint "
                         "(default: the runout profile's terrain.dem)")
    ap.add_argument("--road", type=Path, default=None,
                    help="road centreline (default: the runout profile's road)")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    from config import ProjectConfig
    cfg = ProjectConfig.from_toml(a.toml)
    path_dem = a.path_dem or runout_profile_dem(a.toml)
    road = a.road or runout_profile_road(a.toml)
    rep = check(a.dem, cfg.boundary_kml, cfg.start_zone_kml, path_dem, road,
                a.compare, a.min_coverage)
    print_report(rep)
    if a.json:
        a.json.write_text(json.dumps(rep, indent=2) + "\n")
    return 1 if rep["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
