"""
cluster_carry.py — carry last season's cluster map onto this season's DEM.

    python cluster_carry.py --toml T --from-map <last season>/cluster_map.tif
                            [--max-fill-m 3]

Ron, 2026-11-03: a new season starts from last season's clusters rather than
from its own first surveys. On the 2025-26 surveys, a map built on the late
season fitted the early-season surveys almost as well as a map built on them
(cell HS R2 0.919 against 0.935), and cluster_update splits clusters that
turn out heterogeneous. A re-cluster on this season's surveys is planned
once there are enough of them.

The TOML's [paths] dem must already be the new surface. This:
  1. builds the 1 m reference DEM from it (as `forcing_pipeline resample`
     does) and the domain mask (as `cluster` does: boundary KML, slope);
  2. resamples the old cluster IDs onto that grid, nearest neighbour, in the
     map frame (both rasters carry their own transform and CRS);
  3. drops IDs on cells outside the new domain, and gives cells inside it
     with no ID the nearest cell's ID when that cell is within --max-fill-m.
     Cells farther than that are left out of the domain and reported: a strip
     of new terrain is not part of any old cluster's snow history;
  4. writes cluster_map.npy/.tif and domain_mask.npy to analysis_dir, and
     cluster_carry_report.json beside them.

Cluster IDs are kept, so last season's and this season's outputs for a
cluster can be compared. IDs that end up with fewer than min_cluster_size
cells are reported, not merged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig


def carry_ids(old_ids: np.ndarray, old_transform, old_crs,
              dst_shape: tuple, dst_transform, dst_crs,
              domain: np.ndarray, max_fill_m: float,
              res_m: float = 1.0) -> tuple:
    """Old cluster IDs on the new grid, restricted to and filling `domain`.

    Returns (cluster_map int32, domain_out bool, stats dict). domain_out is
    `domain` minus the cells left without a cluster.
    """
    carried = np.zeros(dst_shape, dtype=np.int32)
    reproject(source=old_ids.astype(np.int32), destination=carried,
              src_transform=old_transform, src_crs=old_crs, src_nodata=0,
              dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=0,
              resampling=Resampling.nearest)
    old_cells = int((carried > 0).sum())
    outside = (carried > 0) & ~domain
    carried[outside] = 0

    need = domain & (carried == 0)
    dist, (ri, ci) = distance_transform_edt(carried == 0, sampling=res_m,
                                            return_indices=True)
    fill = need & (dist <= max_fill_m)
    carried[fill] = carried[ri[fill], ci[fill]]
    left_out = need & ~fill
    domain_out = domain & (carried > 0)

    old_set = set(np.unique(old_ids[old_ids > 0]).tolist())
    ids, counts = np.unique(carried[carried > 0], return_counts=True)
    stats = {
        "old_clusters": len(old_set),
        "clusters": int(len(ids)),
        "lost_clusters": sorted(old_set - set(ids.tolist())),
        "domain_cells": int(domain.sum()),
        "cells_carried": int(old_cells - int(outside.sum())),
        "cells_dropped_outside_domain": int(outside.sum()),
        "cells_filled_from_nearest": int(fill.sum()),
        "cells_left_out": int(left_out.sum()),
        "max_fill_m": max_fill_m,
        "fill_distance_m_max": float(dist[fill].max()) if fill.any() else 0.0,
        "cluster_cells_min": int(counts.min()) if len(counts) else 0,
        "cluster_cells_median": float(np.median(counts)) if len(counts) else 0.0,
        "cluster_cells_max": int(counts.max()) if len(counts) else 0,
    }
    return carried, domain_out, stats


def carry(cfg: ProjectConfig, from_map: Path, max_fill_m: float = 3.0) -> dict:
    from clustering import build_domain_mask
    from pipeline_io import load_dem
    from spatial_model import create_reference_dem

    cfg.ensure_dirs()
    ref = cfg.resampled_dir / "dem_1m.tif"
    if not ref.exists():
        print(f"Creating 1 m reference DEM from {cfg.dem_path.name}")
        create_reference_dem(str(cfg.dem_path), str(ref), cfg.target_resolution_m)
    dem, transform, crs = load_dem(cfg)
    domain = build_domain_mask(dem, transform, crs, kml_path=str(cfg.boundary_kml),
                               min_slope_deg=cfg.min_slope_deg,
                               resolution=cfg.target_resolution_m)
    with rasterio.open(from_map) as src:
        old = src.read(1)
        old_t, old_crs = src.transform, src.crs

    cmap, domain_out, stats = carry_ids(old, old_t, old_crs, dem.shape, transform,
                                        crs, domain, max_fill_m,
                                        cfg.target_resolution_m)
    ids, counts = np.unique(cmap[cmap > 0], return_counts=True)
    stats["small_clusters"] = [int(i) for i, c in zip(ids, counts)
                               if c < cfg.min_cluster_size]
    stats.update({"from_map": str(from_map), "dem": str(cfg.dem_path)})

    out = cfg.analysis_dir
    np.save(out / "cluster_map.npy", cmap)
    np.save(out / "domain_mask.npy", domain_out)
    profile = {"driver": "GTiff", "dtype": "int32", "compress": "lzw",
               "height": dem.shape[0], "width": dem.shape[1], "count": 1,
               "crs": crs, "transform": transform, "nodata": 0}
    with rasterio.open(out / "cluster_map.tif", "w", **profile) as dst:
        dst.write(cmap, 1)
    (out / "cluster_carry_report.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Carried {stats['old_clusters']} clusters -> {stats['clusters']} on "
          f"{cfg.dem_path.name}")
    print(f"  domain {stats['domain_cells']} cells: {stats['cells_carried']} carried, "
          f"{stats['cells_filled_from_nearest']} filled from the nearest cluster "
          f"(<= {max_fill_m} m), {stats['cells_left_out']} left out")
    print(f"  {stats['cells_dropped_outside_domain']} old cells fell outside the new "
          f"domain; lost clusters: {len(stats['lost_clusters'])}; under "
          f"{cfg.min_cluster_size} cells: {len(stats['small_clusters'])}")
    print(f"  wrote {out / 'cluster_map.tif'} and cluster_carry_report.json")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", required=True)
    ap.add_argument("--from-map", required=True, type=Path,
                    help="last season's cluster_map.tif (with its transform and CRS)")
    ap.add_argument("--max-fill-m", type=float, default=3.0)
    args = ap.parse_args()
    cfg = ProjectConfig.from_toml(Path(args.toml))
    if cfg.cluster_map_path.exists():
        sys.exit(f"ERROR: {cfg.cluster_map_path} exists; this would overwrite it. "
                 f"Archive last season's analysis_dir first.")
    carry(cfg, args.from_map, args.max_fill_m)


if __name__ == "__main__":
    main()
