#!/usr/bin/env python3
"""
release_volume.py

Author: Valerie Foley
Last Updated: 7/2026

Description:
    Compute avalanche release volume by zonal integration of a release-depth
    raster over a release-area polygon:

        V = sum( depth_i * pixel_area_i )  over raster cells in the polygon

    Correct when release depth varies spatially. Also reports the uniform
    mean-depth * area estimate as a cross-check.

    Assumptions:
      * Depth raster and release polygon share the same projected CRS in
        meters (checked, warns on mismatch).
      * Raster values are snow depth in meters unless --depth-units is set.
      * SWE rasters need --depth-is-swe and --rho to convert to geometric
        depth (com1DFA wants geometric volume).
      * NoData and negative values are excluded; --min-depth sets a floor.

Usage:
    python release_volume.py --depth depth.tif --release release.shp
    python release_volume.py --depth depth.tif --release release.shp --depth-units cm
    python release_volume.py --depth depth.tif --release release.shp --depth-is-swe --rho 250

Requirements:
    - rasterio, fiona, shapely, numpy
"""

import argparse
import sys
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import fiona
from shapely.geometry import shape
from shapely.ops import unary_union


WATER_DENSITY = 1000.0  # kg/m3, for SWE -> depth conversion


def parse_args():
    # Parse command-line arguments
    # @param: None
    # @returns: argparse.Namespace with parsed arguments

    p = argparse.ArgumentParser(description="Compute avalanche release volume.")
    p.add_argument("--depth", required=True, help="Release depth raster (GeoTIFF).")
    p.add_argument("--release", required=True, help="Release area polygon (shapefile/geojson).")
    p.add_argument("--depth-units", default="m", choices=["m", "cm", "ft"],
                   help="Units of the depth raster values (default: m).")
    p.add_argument("--depth-is-swe", action="store_true",
                   help="Treat raster as SWE; convert to geometric depth using --rho.")
    p.add_argument("--rho", type=float, default=250.0,
                   help="Snow density kg/m3 for SWE->depth conversion (default 250).")
    p.add_argument("--min-depth", type=float, default=0.0,
                   help="Ignore cells with depth below this (in meters, post-conversion).")
    return p.parse_args()


def unit_to_meters(arr, units):
    # Convert a depth array to meters
    # @param arr: numpy array of depth values
    # @param units: 'm', 'cm', or 'ft'
    # @returns: numpy array in meters
    # @raises ValueError: if units is not recognized

    if units == "m":
        return arr
    if units == "cm":
        return arr / 100.0
    if units == "ft":
        return arr * 0.3048
    raise ValueError(units)


def main():
    # Compute and print the zonal release volume
    # @param: None (args from command line)
    # @returns: None (exits with status code)

    args = parse_args()

    # --- load polygon geometries ---
    with fiona.open(args.release) as src:
        vec_crs = src.crs
        geoms = [shape(f["geometry"]) for f in src if f["geometry"] is not None]
    if not geoms:
        sys.exit("No geometries found in release file.")
    release_geom = unary_union(geoms)  # dissolve to single (multi)polygon
    geom_list = [release_geom.__geo_interface__]

    # --- open raster, CRS sanity check ---
    with rasterio.open(args.depth) as rds:
        ras_crs = rds.crs
        px_w, px_h = rds.res  # in CRS units
        pixel_area = abs(px_w * px_h)

        print("=" * 60)
        print("CRS check")
        print(f"  raster CRS : {ras_crs.to_string() if ras_crs else 'UNKNOWN'}")
        print(f"  vector CRS : {vec_crs}")
        if ras_crs is None:
            print("  !! raster has no CRS; assuming it matches the vector.")
        elif vec_crs and ras_crs.to_string() != rasterio.crs.CRS(vec_crs).to_string():
            print("  !! WARNING: raster and vector CRS differ. Reproject the")
            print("     polygon to the raster CRS before trusting this number.")
        if ras_crs and ras_crs.is_geographic:
            print("  !! WARNING: raster CRS is geographic (degrees). Pixel area")
            print("     is NOT in m^2 - reproject to a projected CRS (EPSG:6342).")
        print(f"  pixel size : {px_w} x {px_h} (CRS units)")
        print(f"  pixel area : {pixel_area:.4f} (CRS units^2 -> m^2 if projected)")

        # --- clip raster to polygon ---
        out_img, out_transform = rio_mask(rds, geom_list, crop=True, filled=True)
        band = out_img[0].astype("float64")
        nodata = rds.nodata

    # --- mask out nodata / invalid ---
    valid = np.ones(band.shape, dtype=bool)
    if nodata is not None:
        valid &= (band != nodata)
    valid &= np.isfinite(band)

    depth_m = unit_to_meters(band, args.depth_units)

    if args.depth_is_swe:
        # SWE (m of water) -> geometric snow depth: d = SWE * rho_water / rho_snow
        depth_m = depth_m * (WATER_DENSITY / args.rho)
        print(f"\nConverted SWE -> depth using rho_snow = {args.rho} kg/m3")

    valid &= (depth_m > args.min_depth)

    depths = depth_m[valid]
    if depths.size == 0:
        sys.exit("No valid depth cells inside the polygon. Check CRS / nodata / units.")

    # --- the actual integration ---
    volume_m3 = float(np.sum(depths) * pixel_area)

    n_cells = depths.size
    covered_area = n_cells * pixel_area
    mean_depth = float(np.mean(depths))
    max_depth = float(np.max(depths))
    min_depth = float(np.min(depths))
    poly_area = float(release_geom.area)  # polygon area in CRS units^2

    print("\n" + "=" * 60)
    print("RELEASE VOLUME")
    print("=" * 60)
    print(f"  valid cells inside polygon : {n_cells:,}")
    print(f"  polygon area (vector)      : {poly_area:,.1f} m^2")
    print(f"  area covered by valid cells: {covered_area:,.1f} m^2")
    print(f"  mean release depth         : {mean_depth:.3f} m")
    print(f"  min / max release depth    : {min_depth:.3f} / {max_depth:.3f} m")
    print("-" * 60)
    print(f"  >> RELEASE VOLUME (zonal)  : {volume_m3:,.1f} m^3")
    print(f"  (cross-check mean*area)    : {mean_depth * poly_area:,.1f} m^3")
    print("=" * 60)

    # --- interpret against samosAT calibration buckets ---
    print("\nsamosAT default calibration buckets (release volume):")
    print(f"  small  : < 25,000 m^3")
    print(f"  medium : < 60,000 m^3")
    print(f"  (both also assume runout altitude < 1600 m msl)")
    if volume_m3 < 25000:
        bucket = "SMALL (< 25,000 m^3)"
    elif volume_m3 < 60000:
        bucket = "MEDIUM (< 60,000 m^3)"
    else:
        bucket = "LARGE (>= 60,000 m^3) -> default samosAT, no small/medium"
    print(f"  -> this event falls in: {bucket}")
    print("  NB: the small/medium calibrations also assume runout altitude")
    print("      < 1600 m msl; check this against your site.")


if __name__ == "__main__":
    main()
