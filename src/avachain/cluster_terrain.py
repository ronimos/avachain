"""
cluster_terrain.py — clusters from terrain alone, for a slope with no survey
history (its first season).

    python cluster_terrain.py --toml T [--cells-per-cluster 10] [--compactness 0.5]

`forcing_pipeline cluster` groups cells by their HS over a season of surveys,
so it cannot run before a slope has been surveyed. This builds the cluster
map from the DEM instead: SLIC superpixels (skimage.segmentation.slic) on
standardised slope, sin/cos aspect, curvature, relative height and
elevation. Each superpixel is one spatially connected cluster.

Why this is a reasonable start (2025-26 Little Professor surveys, 11-03):
at the same number of units the terrain superpixels fitted each survey's
cell HS and the HS change between consecutive surveys at least as well as
the production HS-evolution clustering. Median R2 was 0.928 / 0.894 against
0.918 / 0.886. The number of units moved the scores more than the method
did. That says nothing about the release model, which grows release areas
cluster by cluster; check that before relying on it.

Aspect enters as (sin, cos), and every feature is standardised over the
domain, so a reflected or rotated aspect convention gives the same
distances and the same clusters.

Cells come from the domain mask (boundary KML and slope, as `cluster`
builds it). When the slope already has resampled surveys, cells with no
valid HS on any of them are dropped, as `cluster` drops them. Writes
cluster_map.npy/.tif, domain_mask.npy and cluster_terrain_report.json to
analysis_dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, laplace, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig

FEATURES = ("slope", "aspect_sin", "aspect_cos", "curvature", "tpi15", "elevation")


def terrain_features(dem: np.ndarray, res_m: float = 1.0) -> dict:
    """Per-cell terrain features (unstandardised). NaN DEM cells get the mean."""
    f = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(f, res_m)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    asp = np.arctan2(-dx, dy)
    return {"slope": slope, "aspect_sin": np.sin(asp), "aspect_cos": np.cos(asp),
            "curvature": -laplace(gaussian_filter(f, 2.0 / res_m)),
            "tpi15": f - uniform_filter(f, max(1, int(round(15 / res_m)))),
            "elevation": f}


def terrain_clusters(dem: np.ndarray, domain: np.ndarray, cells_per_cluster: float,
                     compactness: float = 0.5, res_m: float = 1.0) -> np.ndarray:
    """Cluster IDs 1..n on the domain, 0 elsewhere (int32)."""
    from skimage.segmentation import slic
    feats = terrain_features(dem, res_m)
    img = []
    for k in FEATURES:
        a = feats[k]
        sd = a[domain].std()
        img.append((a - a[domain].mean()) / (sd if sd > 0 else 1.0))
    n = max(1, int(round(domain.sum() / cells_per_cluster)))
    lab = slic(np.dstack(img), n_segments=n, compactness=compactness, mask=domain,
               channel_axis=-1, start_label=1, enforce_connectivity=True)
    lab = np.where(domain, lab, 0)
    ids = np.unique(lab[lab > 0])
    remap = np.zeros(lab.max() + 1, dtype=np.int32)
    remap[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return remap[lab]


def build(cfg: ProjectConfig, cells_per_cluster: float = 10.0,
          compactness: float = 0.5) -> dict:
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
    surveys = sorted(cfg.resampled_dir.glob("hs_*.npy"))
    n_nodata = 0
    if surveys:
        seen = np.zeros(domain.shape, bool)
        for p in surveys:
            seen |= np.isfinite(np.load(p))
        n_nodata = int((domain & ~seen).sum())
        domain &= seen
    cmap = terrain_clusters(dem, domain, cells_per_cluster, compactness,
                            cfg.target_resolution_m)
    ids, counts = np.unique(cmap[cmap > 0], return_counts=True)
    stats = {"dem": str(cfg.dem_path), "cells_per_cluster_target": cells_per_cluster,
             "compactness": compactness, "features": list(FEATURES),
             "domain_cells": int(domain.sum()), "cells_dropped_no_survey_data": n_nodata,
             "clusters": int(len(ids)), "cluster_cells_min": int(counts.min()),
             "cluster_cells_median": float(np.median(counts)),
             "cluster_cells_max": int(counts.max()),
             "small_clusters": int((counts < cfg.min_cluster_size).sum())}
    out = cfg.analysis_dir
    np.save(out / "cluster_map.npy", cmap)
    np.save(out / "domain_mask.npy", domain)
    profile = {"driver": "GTiff", "dtype": "int32", "compress": "lzw",
               "height": dem.shape[0], "width": dem.shape[1], "count": 1,
               "crs": crs, "transform": transform, "nodata": 0}
    with rasterio.open(out / "cluster_map.tif", "w", **profile) as dst:
        dst.write(cmap, 1)
    (out / "cluster_terrain_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Terrain clusters: {stats['clusters']} on {stats['domain_cells']} cells "
          f"(size min/median/max {stats['cluster_cells_min']}/"
          f"{stats['cluster_cells_median']:.0f}/{stats['cluster_cells_max']}; "
          f"{stats['small_clusters']} under {cfg.min_cluster_size}); "
          f"{n_nodata} cells dropped with no survey data")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", required=True)
    ap.add_argument("--cells-per-cluster", type=float, default=10.0)
    ap.add_argument("--compactness", type=float, default=0.5)
    args = ap.parse_args()
    cfg = ProjectConfig.from_toml(Path(args.toml))
    if cfg.cluster_map_path.exists():
        sys.exit(f"ERROR: {cfg.cluster_map_path} exists; this would overwrite it.")
    build(cfg, args.cells_per_cluster, args.compactness)


if __name__ == "__main__":
    main()
