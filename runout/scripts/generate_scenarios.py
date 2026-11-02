"""
Scenario Generator

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Writes synthetic scenario folders in the same layout the snowpack chain
    produces, so the simulate chain runs them unchanged. Each scenario is
    one release polygon at one uniform depth and one density: release
    polygon, depth raster on the DEM grid, density.json, params.json with
    equal weights, and metadata.json. Used by sweep mode and by calibrate
    mode for a scenario built from an observed release. Depths are
    slope-normal.
"""

# --------- Load Libraries --------
import datetime
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from pyproj import CRS as PyprojCRS

try:
    from . import avalanche_helpers as h
except ImportError:
    import avalanche_helpers as h


def _write_prj(asc_path, crs):
    # Write the ESRI WKT1 sidecar an .asc needs to carry a CRS
    # @param asc_path: Path to the .asc
    # @param crs: rasterio CRS
    # @returns: None
    wkt = PyprojCRS.from_user_input(crs).to_wkt(version='WKT1_ESRI')
    asc_path.with_suffix('.prj').write_text(wkt)


def generate_scenarios(out_dir, release_path, dem_path, depths, densities,
                       scour_depth_m=0.0, mu=0.155, xi=1500.0,
                       release_label=None, start_index=1):
    # Write one scenario folder per (depth, density) for one release polygon
    # @param out_dir: Path to write scenario_### folders into
    # @param release_path: Path to the release polygon (one or more features,
    #                      dissolved into a single release area)
    # @param dem_path: Path to the DEM whose grid the depth raster follows
    # @param depths: list of uniform depths (m, slope-normal)
    # @param densities: list of densities (kg/m3)
    # @param scour_depth_m: value for params.json scour_depth_m (0 lets the
    #                       pipeline fall back to its config default)
    # @param mu: Voellmy mu written to params.json (inert under samosAT)
    # @param xi: Voellmy xi written to params.json (inert under samosAT)
    # @param release_label: name for the release polygon in params.json
    #                       (default: the file stem)
    # @param start_index: first scenario number
    # @returns: int - number of scenario folders written
    # @raises ValueError: if the release polygon is empty or off the DEM
    # scenario_probability is equal across the set, so the stack is
    #       unweighted; the sweep products read depth_m and density_kgm3

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    release_label = release_label or Path(release_path).stem

    with rasterio.open(dem_path) as dem:
        dem_crs = dem.crs
        dem_profile = dem.profile.copy()
        dem_shape = (dem.height, dem.width)
        dem_transform = dem.transform

    gdf = gpd.read_file(release_path)
    if gdf.crs is not None and dem_crs is not None and gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)
    elif gdf.crs is None:
        gdf = gdf.set_crs(dem_crs)
    geom = gdf.geometry.unary_union.buffer(0)
    if geom.is_empty:
        # throw error if the release file has no usable geometry
        raise ValueError(f"release polygon is empty: {release_path}")

    # Release area mask on the DEM grid (1 inside, 0 outside)
    area_mask = h.polygon_to_raster_mask(geom, dem_shape, dem_transform, 1.0)
    n_cells = int(area_mask.sum())
    if n_cells == 0:
        # throw error if the polygon misses the DEM entirely
        raise ValueError(f"release polygon has no cells on the DEM: {release_path}")
    cell_area = abs(dem_transform.a * dem_transform.e)
    area_m2 = float(geom.area)

    combos = [(d, rho) for d in depths for rho in densities]
    weight = 1.0 / len(combos)
    n = 0
    for i, (depth, rho) in enumerate(combos, start=start_index):
        sid = f"scenario_{i:03d}"
        inp = out_dir / sid / 'inputs'
        inp.mkdir(parents=True, exist_ok=True)

        gdf_out = gpd.GeoDataFrame({'name': [release_label]}, geometry=[geom], crs=dem_crs)
        gdf_out.to_file(inp / 'release.geojson', driver='GeoJSON')

        depth_arr = (area_mask * float(depth)).astype('float32')
        prof = dem_profile.copy()
        prof.update(nodata=0.0)
        h.write_raster(inp / 'depth.asc', depth_arr, prof, driver='AAIGrid')
        _write_prj(inp / 'depth.asc', dem_crs)

        (inp / 'density.json').write_text(json.dumps({'mean_kgm3': float(rho)}, indent=2))

        params = {
            'mu': mu,
            'xi': xi,
            'rho_kgm3': float(rho),
            'scour_depth_m': float(scour_depth_m),
            'scour_depth_mid_m': float(scour_depth_m),
            'hand_hardness_value': None,
            'scenario_probability': weight,
            'mean_depth_m': float(depth),
            'release_area_m2': area_m2,
            'total_volume_m3': float(depth) * n_cells * cell_area,
            'depth_m': float(depth),
            'density_kgm3': float(rho),
            'release_label': release_label,
        }
        (inp / 'params.json').write_text(json.dumps(params, indent=2))

        meta = {
            'generator': 'scripts.generate_scenarios',
            'generated': datetime.datetime.now().isoformat(timespec='seconds'),
            'release_file': str(release_path),
            'dem_file': str(dem_path),
            'depth_convention': 'slope_normal',
        }
        (inp / 'metadata.json').write_text(json.dumps(meta, indent=2))
        n += 1
        logging.debug(f"{sid}: depth={depth} m rho={rho} kg/m3 "
                      f"volume={params['total_volume_m3']:.0f} m3")

    logging.info(f"generated {n} scenario(s) from {Path(release_path).name}: "
                 f"{len(depths)} depth(s) x {len(densities)} density(ies), "
                 f"release area {area_m2:.0f} m2")
    return n
