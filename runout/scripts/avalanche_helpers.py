"""
avalanche_helpers.py

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Shared utilities for the avalanche runout pipeline:
        - Raster I/O (read, write, format conversion, COG output)
        - Vector I/O (GeoJSON <-> Shapefile)
        - CRS validation across mixed inputs
        - Pixel <-> world coordinate helpers
        - Statistics helpers (Wilson CI)
        - Entrainment area auto-detection (alpha-beta corridor + fallback)

    No command line. Imported by the other pipeline modules.

Requirements:
    - rasterio, fiona, geopandas, shapely, pyproj, numpy, scipy
"""

# --------- Load Libraries --------
import os
import json
import shutil
import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes
from pyproj import CRS as PyprojCRS
import fiona
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, mapping, shape as shp_shape
from scipy import stats as scipy_stats




def read_raster(path):
    # Read first band of a raster + key metadata
    # @param path: Path to raster file (.tif, .asc, etc.)
    # @returns: tuple (data, meta) where:
    #   - data: 2D numpy array
    #   - meta: dict with transform, crs, nodata, shape, dtype, profile
    
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'nodata': src.nodata,
            'shape': data.shape,
            'dtype': data.dtype,
            'profile': src.profile.copy()
        }
    
    logging.debug(f"Read raster: {Path(path).name}")
    return data, meta


def write_raster(path, data, profile, driver='GTiff', cog=True):
    # Write a raster to disk, optionally as Cloud-Optimized GeoTIFF
    # @param path: Output file path
    # @param data: 2D numpy array to write
    # @param profile: rasterio profile dict (will be modified for driver)
    # @param driver: 'GTiff' or 'AAIGrid'
    # @param cog: If True and driver=GTiff, write tiled+compressed
    # @returns: Path to written file
    # #   - AAIGrid (.asc) doesn't support compression or tiling - those
    #     options get stripped automatically
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build clean output profile based on driver
    out_profile = profile.copy()
    out_profile.update(driver=driver, count=1, dtype=data.dtype)
    
    if driver == 'GTiff' and cog:
        # COG-friendly settings
        out_profile.update(
            compress='deflate', tiled=True,
            blockxsize=256, blockysize=256
        )
    else:
        # AAIGrid rejects these keys - strip them
        for key in ('compress', 'tiled', 'blockxsize', 'blockysize',
                    'interleave', 'cellsize'):
            out_profile.pop(key, None)
    
    with rasterio.open(path, 'w', **out_profile) as dst:
        dst.write(data, 1)
    
    return path


def convert_raster_format(src_path, dst_path):
    # Convert a raster between formats based on file extensions
    # @param src_path: Source raster (.tif or .asc)
    # @param dst_path: Destination raster (extension determines format)
    # @returns: Path to written file
    # #   - If src and dst are both same format, copies file directly
    #   - .prj sidecars only get copied for .asc files. GeoTIFF embeds CRS
    #     internally; copying a .prj alongside a .tif is unnecessary and
    #     would re-trigger AvaFrame's buggy remeshedRasters glob.
    #   - .tif <-> .asc conversion goes through rasterio
    
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    
    # Same format - just copy
    if src_path.suffix.lower() == dst_path.suffix.lower():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        # Copy .prj sidecar only for .asc files (GeoTIFF embeds CRS internally)
        if dst_path.suffix.lower() == '.asc':
            prj = src_path.with_suffix('.prj')
            if prj.exists():
                shutil.copy2(prj, dst_path.with_suffix('.prj'))
        return dst_path
    
    # Format conversion via rasterio
    data, meta = read_raster(src_path)
    driver = 'AAIGrid' if dst_path.suffix.lower() == '.asc' else 'GTiff'
    
    # Force float32 for AAIGrid compatibility
    return write_raster(
        dst_path, data.astype('float32'), meta['profile'],
        driver=driver, cog=(driver == 'GTiff')
    )


def resample_to_reference_grid(src_path, ref_path, dst_path,
                               resampling_method='bilinear',
                               fill_value=0.0):
    # Resample a raster onto another raster's exact grid (cell size, origin, extent)
    # @param src_path: Source raster to resample
    # @param ref_path: Reference raster whose grid to align to
    # @param dst_path: Output raster path
    # @param resampling_method: 'bilinear' (default), 'nearest', or 'cubic'
    # @param fill_value: Value assigned to cells outside the source extent
    # @returns: Path to written raster
    # @raises ValueError: if resampling_method is not recognized
    # bilinear suits continuous fields; use 'nearest' for binary data
    
    from rasterio.warp import reproject, Resampling
    
    # Map method strings to rasterio Resampling enum
    methods = {
        'bilinear': Resampling.bilinear,
        'nearest': Resampling.nearest,
        'cubic': Resampling.cubic,
    }
    if resampling_method not in methods:
        # throw error if invalid method
        raise ValueError(
            f"resampling_method must be one of {list(methods.keys())}, "
            f"got '{resampling_method}'"
        )
    
    # Open reference to get target grid
    with rasterio.open(ref_path) as ref:
        target_profile = ref.profile.copy()
        target_transform = ref.transform
        target_crs = ref.crs
        target_shape = (ref.height, ref.width)
    
    # Open source and reproject onto target grid
    with rasterio.open(src_path) as src:
        # Output array, pre-filled with fill_value for cells outside source
        out_data = np.full(target_shape, fill_value, dtype='float32')
        
        reproject(
            source=rasterio.band(src, 1),
            destination=out_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=methods[resampling_method],
            dst_nodata=fill_value
        )
    
    # Write with reference grid's profile (driver determined by dst extension)
    dst_path = Path(dst_path)
    driver = 'AAIGrid' if dst_path.suffix.lower() == '.asc' else 'GTiff'
    target_profile.update(
        driver=driver, count=1, dtype='float32', nodata=fill_value
    )
    
    return write_raster(
        dst_path, out_data, target_profile,
        driver=driver, cog=(driver == 'GTiff')
    )


def compute_slope_deg(dem_path):
    # Compute per-cell slope in degrees from a DEM
    # @param dem_path: Path to DEM raster
    # @returns: tuple (slope_deg, meta) where slope_deg is a 2D float32 array
    #           on the DEM grid and meta is the standard read_raster meta dict
    # #   - Uses centered finite differences (np.gradient). Edge rows/cols use
    #     one-sided differences so the output has the same shape as the DEM.
    #   - Assumes square cells (transform.a == -transform.e); reads cell size
    #     from transform.a.

    elev, meta = read_raster(dem_path)
    cell_size = abs(meta['transform'].a)

    # np.gradient returns (dy, dx) in array order (row, col)
    dz_dy, dz_dx = np.gradient(elev.astype('float32'), cell_size)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad).astype('float32')

    return slope_deg, meta


def mosaic_dem_with_priority(primary_path, fallback_path, out_path,
                             feather_cells=5, overlap_check=True,
                             overlap_stats_out=None,
                             vertical_correction_m=0.0):
    # Mosaic two DEMs: primary (high-res, smaller extent) wins where it has
    # data; fallback covers everywhere else. Feathered blend across the seam.
    # @param primary_path: Path to the high-resolution / bare-ground DEM/DSM
    # @param fallback_path: Path to the wider-extent fallback DEM (e.g. 3DEP)
    # @param out_path: Path to the merged DEM
    # @param feather_cells: int >= 0. Width (in cells) of the linear blend
    #                       ramp from primary to fallback at the seam. 0 = hard edge.
    # @param overlap_check: If True, compute vertical offset stats over the
    #                       primary's footprint and log them.
    # @param overlap_stats_out: Optional Path to write the offset stats as JSON
    # @param vertical_correction_m: Constant (m) added to every valid cell
    #                       of the primary DEM before stats and merging.
    #                       Compensates a known bulk vertical offset
    #                       (e.g. PPK long-baseline error). Negative = down.
    # @returns: Path to the merged DEM
    # @raises ValueError: if the primary DEM has no valid cells over the
    #                     fallback extent
    # primary is resampled onto the fallback grid (bilinear); the
    #       seam is blended linearly across feather_cells

    from scipy import ndimage
    import json

    primary_path = Path(primary_path)
    fallback_path = Path(fallback_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: resample primary onto fallback's grid
    tmp_resampled = out_path.parent / '_tmp_primary_on_fallback_grid.tif'
    resample_to_reference_grid(
        src_path=primary_path,
        ref_path=fallback_path,
        dst_path=tmp_resampled,
        resampling_method='bilinear',
        fill_value=np.nan
    )

    primary_data, primary_meta = read_raster(tmp_resampled)
    fallback_data, fallback_meta = read_raster(fallback_path)

    # Validity mask: primary contributes where it has finite, non-nodata values.
    # Compute this BEFORE applying the vertical correction so the validity test
    # uses the raw values (correction may shift nodata sentinels into valid range).
    valid = np.isfinite(primary_data)
    if primary_meta.get('nodata') is not None:
        valid &= (primary_data != primary_meta['nodata'])

    # Apply vertical correction to valid cells only (don't shift nodata).
    if vertical_correction_m != 0.0:
        logging.info(
            f'Applying vertical correction of {vertical_correction_m:+.3f} m '
            f'to primary DEM ({primary_path.name}).'
        )
        primary_data = np.where(valid, primary_data + float(vertical_correction_m), primary_data)

    if not valid.any():
        tmp_resampled.unlink(missing_ok=True)
        raise ValueError(
            f"Primary DEM {primary_path.name} has no valid cells over the "
            f"fallback DEM extent. Check CRS and footprint."
        )

    # Step 2: overlap stats (informational)
    stats = None
    if overlap_check:
        diff = primary_data[valid] - fallback_data[valid]
        stats = {
            'primary_file': str(primary_path),
            'fallback_file': str(fallback_path),
            'vertical_correction_applied_m': float(vertical_correction_m),
            'n_overlap_cells': int(valid.sum()),
            'mean_offset_m': float(np.nanmean(diff)),
            'median_offset_m': float(np.nanmedian(diff)),
            'std_offset_m': float(np.nanstd(diff)),
            'p05_offset_m': float(np.nanpercentile(diff, 5)),
            'p95_offset_m': float(np.nanpercentile(diff, 95)),
        }
        logging.info(
            f"DEM overlap stats ({stats['n_overlap_cells']} cells): "
            f"mean={stats['mean_offset_m']:.3f} m, "
            f"median={stats['median_offset_m']:.3f} m, "
            f"std={stats['std_offset_m']:.3f} m"
        )
        if abs(stats['mean_offset_m']) > 0.2:
            logging.warning(
                f"DEM mean vertical offset = {stats['mean_offset_m']:.3f} m "
                f"exceeds 0.2 m. Surfaces may be on different vertical datums "
                f"or have systematic registration error. Consider applying a "
                f"constant correction before merging."
            )
        if stats['std_offset_m'] > 0.5:
            logging.warning(
                f"DEM offset std = {stats['std_offset_m']:.3f} m exceeds 0.5 m. "
                f"Suggests local registration error or genuine surface "
                f"differences (e.g. canopy returns in fallback)."
            )
        if overlap_stats_out is not None:
            overlap_stats_out = Path(overlap_stats_out)
            overlap_stats_out.parent.mkdir(parents=True, exist_ok=True)
            with open(overlap_stats_out, 'w') as f:
                json.dump(stats, f, indent=2)

    # Step 3: build the blend weight raster
    # weight = 1.0 where primary fully wins, 0.0 where fallback fully wins,
    # linear ramp across feather_cells width at the seam
    if feather_cells > 0:
        # distance_transform_edt: distance (in cells) from each invalid cell
        # to the nearest valid cell, and vice versa.
        dist_outside = ndimage.distance_transform_edt(~valid).astype('float32')
        dist_inside = ndimage.distance_transform_edt(valid).astype('float32')

        # Cells inside primary, far from edge -> weight 1
        # Cells inside primary, within feather_cells of edge -> ramps down
        # Cells outside primary, within feather_cells of edge -> ramps up from 0
        # Cells outside primary, far from edge -> weight 0
        weight = np.zeros_like(primary_data, dtype='float32')

        # Inside: full primary weight, ramp down within feather_cells of edge
        inside_full = valid & (dist_inside >= feather_cells)
        inside_ramp = valid & (dist_inside < feather_cells)
        weight[inside_full] = 1.0
        weight[inside_ramp] = 0.5 + 0.5 * (dist_inside[inside_ramp] / feather_cells)

        # Outside but within feather_cells of seam: ramps from 0 toward 0.5 at seam
        outside_ramp = (~valid) & (dist_outside < feather_cells)
        weight[outside_ramp] = 0.5 - 0.5 * (dist_outside[outside_ramp] / feather_cells)

        # For the blend, we need primary values in the outside ramp band too.
        # Fill primary with fallback there so the blend math doesn't propagate NaN.
        primary_filled = np.where(valid, primary_data, fallback_data)
        merged = weight * primary_filled + (1.0 - weight) * fallback_data
    else:
        # Hard edge: primary where valid, fallback elsewhere
        merged = np.where(valid, primary_data, fallback_data).astype('float32')

    # Step 4: write merged DEM to disk
    write_raster(
        out_path,
        merged.astype('float32'),
        fallback_meta['profile'],
        driver='GTiff',
        cog=True
    )

    tmp_resampled.unlink(missing_ok=True)

    logging.info(
        f"Merged DEM written: {out_path.name} "
        f"(primary cells: {int(valid.sum())}, "
        f"fallback cells: {int((~valid).sum())}, "
        f"feather: {feather_cells} cells)"
    )

    return out_path



def geojson_to_shapefile(src_path, dst_path, attributes=None, fill_holes=False,
                         clip_to_raster=None, explode_multipart=False,
                         min_part_area_m2=1.0):
    # Convert a GeoJSON to ESRI Shapefile (com1DFA expects .shp)
    # @param src_path: Path to .geojson (or any geopandas-readable file)
    # @param dst_path: Path to .shp
    # @param attributes: Optional dict of {field_name: value} added to every
    #                    feature (e.g. 'thickness' required by AvaFrame)
    # @param fill_holes: If True, drop interior rings. com1DFA rejects any
    #                    area feature with holes; filling errs conservative.
    # @param clip_to_raster: Optional raster Path - clip features to its
    #                        valid-data footprint. com1DFA aborts on features
    #                        outside the DEM.
    # @param explode_multipart: If True, split multipart geometries into one
    #                           simple polygon per row (com1DFA requirement
    #                           for area features)
    # @param min_part_area_m2: drop exploded parts smaller than this
    # @returns: Path to written shapefile
    # @raises ValueError: if clipping or exploding removes every feature
    # shapefile field names are truncated to 10 chars
    
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    gdf = gpd.read_file(src_path)
    
    # Clip to the raster valid-data footprint, shrunk by a 3-cell margin.
    # AvaFrame checks cell-center bounds and rejects edge-touching features.
    if clip_to_raster is not None:
        from shapely.geometry import box, shape as _shape
        from shapely.ops import unary_union as _unary_union
        with rasterio.open(clip_to_raster) as src:
            cell = max(abs(src.transform.a), abs(src.transform.e))
            margin = 3 * cell
            target_crs = src.crs
            src_transform = src.transform
            l, b, r, t = src.bounds
            # Clip to the valid-data footprint, not the bounding box.
            # For masked DEMs, NoData fills the corners and com1DFA rejects
            # features sitting over NoData.
            band1 = src.read(1, masked=True)
            if np.ma.is_masked(band1):
                valid = (~band1.mask).astype(np.uint8)
            else:
                # No mask flagged: fall back to nodata value if present
                nod = src.nodata
                arr = src.read(1)
                if nod is not None:
                    valid = (arr != nod).astype(np.uint8)
                else:
                    valid = np.ones(arr.shape, dtype=np.uint8)

        # Rectangular DEM: footprint == bbox. Otherwise vectorize the mask.
        if valid.all():
            clip_geom = box(l + margin, b + margin, r - margin, t - margin)
        else:
            shapes_iter = shapes(
                valid, mask=valid.astype(bool), transform=src_transform)
            polys = [_shape(geom) for geom, val in shapes_iter if val == 1]
            footprint = _unary_union(polys)
            # Shrink inward so no feature touches the edge.
            # buffer(0) first heals tiny topology issues.
            clip_geom = footprint.buffer(0).buffer(-margin)
            if clip_geom.is_empty:
                # margin ate the whole footprint (tiny DEM); fall back to
                # the un-shrunk footprint rather than clipping to nothing
                clip_geom = footprint.buffer(0)

        # Clip in the raster's CRS - reproject if needed, then return to original
        original_crs = gdf.crs
        if original_crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        gdf['geometry'] = gdf.geometry.intersection(clip_geom)
        
        # Drop empty geometries that fell entirely outside
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
        if original_crs != target_crs:
            gdf = gdf.to_crs(original_crs)
        
        if len(gdf) == 0:
            # throw error if clipping removed everything
            raise ValueError(
                f"Clipping {src_path.name} to {Path(clip_to_raster).name} "
                "valid-data footprint removed all features. Check that your "
                "shapefile actually overlaps the DEM's valid (non-NoData) area."
            )
    
    # Explode multipart / collection geometries into one simple polygon per
    # row. com1DFA rejects any area feature that is not a single polygon.
    # Runs BEFORE hole filling so parts produced by clipping (including
    # GeometryCollection fragments) also get their holes filled below.
    if explode_multipart:
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        # Drop empty / non-polygon / sliver parts
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        gdf = gdf[gdf.geometry.geom_type == 'Polygon']
        if min_part_area_m2 and min_part_area_m2 > 0:
            gdf = gdf[gdf.geometry.area >= min_part_area_m2]
        gdf = gdf.reset_index(drop=True)
        if len(gdf) == 0:
            # throw error if exploding removed everything
            raise ValueError(
                f"After exploding multipart and dropping slivers, "
                f"{src_path.name} has no valid polygons left. Check the "
                "source geometry and min_part_area_m2."
            )

    # Fill holes: rebuild each polygon from its exterior ring only
    if fill_holes:
        from shapely.geometry import MultiPolygon

        def _fill(geom):
            if geom is None or geom.is_empty:
                return geom
            if geom.geom_type == 'Polygon':
                return Polygon(geom.exterior.coords)
            if geom.geom_type == 'MultiPolygon':
                return MultiPolygon(
                    [Polygon(p.exterior.coords) for p in geom.geoms]
                )
            if geom.geom_type == 'GeometryCollection':
                # Keep only polygon parts (clipping can emit collections)
                parts = [_fill(g) for g in geom.geoms
                         if g.geom_type in ('Polygon', 'MultiPolygon')]
                if not parts:
                    return None
                flat = []
                for p in parts:
                    flat.extend(p.geoms if p.geom_type == 'MultiPolygon'
                                else [p])
                return flat[0] if len(flat) == 1 else MultiPolygon(flat)
            return geom

        gdf = gdf.set_geometry([_fill(g) for g in gdf.geometry])
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        gdf = gdf.reset_index(drop=True)

    # Inject required attributes (e.g. thickness for AvaFrame validation)
    if attributes:
        for field, value in attributes.items():
            gdf[field] = value
    
    # Truncate column names to 10 chars for shapefile compliance
    gdf.columns = [c[:10] for c in gdf.columns]
    gdf.to_file(dst_path, driver='ESRI Shapefile')
    
    return dst_path


def get_crs_of(path):
    # Get the CRS of any geospatial file (raster or vector)
    # @param path: Path to .tif/.asc/.shp/.geojson/.gpkg
    # @returns: rasterio CRS object, or None if no CRS set
    # @raises ValueError: if the file extension is not recognized
    
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix in ('.tif', '.tiff', '.asc'):
        with rasterio.open(path) as src:
            return src.crs
    
    if suffix in ('.shp', '.geojson', '.gpkg', '.json'):
        with fiona.open(path) as src:
            if src.crs:
                return CRS.from_user_input(src.crs)
        return None
    
    # throw error if file extension not recognized
    raise ValueError(f"Unknown geospatial extension: {suffix}")


def horizontal_crs(crs):
    # Extract the horizontal (2D) sub-CRS from a possibly-compound CRS
    # @param crs: rasterio/pyproj CRS object (may be compound)
    # @returns: pyproj CRS object representing only the horizontal part
    # DEMs often carry a compound CRS (horizontal + vertical datum)
    #       while vector layers carry only the horizontal part. Comparing
    #       horizontal sub-CRS avoids false mismatches.
    
    if crs is None:
        return None
    pp = PyprojCRS.from_user_input(crs)
    if pp.is_compound and pp.sub_crs_list:
        return pp.sub_crs_list[0]
    return pp


def validate_crs(paths_and_labels, reference_crs):
    # Validate that all input files share the reference CRS (horizontal only)
    # @param paths_and_labels: list of (path, label) tuples for error messages
    # @param reference_crs: rasterio CRS to compare against (compound OK)
    # @returns: tuple (success, mismatches) where:
    #   - success: Boolean
    #   - mismatches: list of human-readable mismatch descriptions
    # #   - Compares horizontal sub-CRS only. A DEM with a compound CRS
    #     (horizontal + NAVD88 vertical) matches a vector layer with only the
    #     horizontal part, since both place coordinates the same way in 2D.
    
    ref_horiz = horizontal_crs(reference_crs)
    mismatches = []
    for path, label in paths_and_labels:
        c = get_crs_of(path)
        if c is None:
            mismatches.append(f"{label} ({path}) has no CRS")
            continue
        c_horiz = horizontal_crs(c)
        if c_horiz != ref_horiz:
            # Use to_authority for a compact error string when possible
            c_str = c_horiz.to_authority() or c_horiz.to_string()
            ref_str = ref_horiz.to_authority() or ref_horiz.to_string()
            mismatches.append(
                f"{label} ({path}) horizontal CRS = {c_str}, "
                f"expected {ref_str}"
            )
    
    return (len(mismatches) == 0), mismatches


def pixels_to_world(rows, cols, transform):
    # Convert pixel (row, col) to world (x, y) coordinates - cell centers
    # @param rows: array of row indices
    # @param cols: array of column indices
    # @param transform: rasterio Affine transform
    # @returns: tuple (xs, ys) of world coordinates
    # #   - Uses cell centers (+0.5 offset), not cell corners
    
    xs = transform.a * (np.asarray(cols) + 0.5) + transform.c
    ys = transform.e * (np.asarray(rows) + 0.5) + transform.f
    return xs, ys


def world_to_pixel(x, y, transform):
    # Convert single world (x, y) point to pixel (row, col)
    # @param x: world x coordinate
    # @param y: world y coordinate
    # @param transform: rasterio Affine transform
    # @returns: tuple (row, col) of integer pixel indices
    
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def sample_raster_at_xy(raster_path, xy_pairs):
    # Sample raster values at world coordinates (interpolation = nearest)
    # @param raster_path: Path to raster
    # @param xy_pairs: list of (x, y) tuples
    # @returns: list of sampled float values (NaN if outside raster)
    
    with rasterio.open(raster_path) as src:
        return [float(v[0]) for v in src.sample(xy_pairs)]


def polygon_to_raster_mask(polygon, raster_shape, transform, fill_value=1.0):
    # Burn a polygon into a raster mask aligned to a reference grid
    # @param polygon: shapely Polygon
    # @param raster_shape: tuple (height, width)
    # @param transform: rasterio Affine
    # @param fill_value: value to assign inside polygon (outside = 0)
    # @returns: 2D numpy array
    
    mask = geometry_mask(
        [mapping(polygon)],
        out_shape=raster_shape,
        transform=transform,
        invert=True
    )
    return np.where(mask, fill_value, 0.0).astype('float32')


def mask_to_polygons(mask, transform):
    # Polygonize a binary raster (value=1 -> polygon)
    # @param mask: 2D uint8/bool array
    # @param transform: rasterio Affine
    # @returns: list of shapely Polygons
    
    polys = []
    for geom, val in shapes(mask.astype('uint8'),
                            mask=mask.astype(bool),
                            transform=transform):
        if val == 1:
            polys.append(shp_shape(geom))
    return polys


def calculate_statistics(values):
    # Calculate basic stats from an array of values (NaN-safe)
    # @param values: numpy array
    # @returns: dict with n, mean, min, max, std, or None if no valid data
    
    clean = values[~np.isnan(values)]
    
    if len(clean) == 0:
        return None
    
    return {
        'n': int(len(clean)),
        'mean': float(np.mean(clean)),
        'min': float(np.min(clean)),
        'max': float(np.max(clean)),
        'std': float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    }


def wilson_ci(k, n, alpha=0.05):
    # Wilson score confidence interval for a binomial proportion
    # @param k: number of successes
    # @param n: number of trials
    # @param alpha: significance level (0.05 for 95% CI)
    # @returns: tuple (low, high) of CI bounds in [0, 1]
    # #   - More robust than normal approximation at small n (our n=20 regime)
    #   - Returns (0, 0) if n=0 to avoid div-by-zero
    
    if n == 0:
        return (0.0, 0.0)
    
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    
    return (max(0.0, centre - half), min(1.0, centre + half))


def steepest_descent_flowline(release_polygon, dem_path, max_steps=5000):
    # Walk downhill from the release centroid using D8-style steepest descent
    # @param release_polygon: shapely Polygon of the release area
    # @param dem_path: Path to DEM raster
    # @param max_steps: Maximum number of descent steps before giving up
    # @returns: shapely LineString of the descent path
    # #   - Step size = DEM cell size
    #   - Terminates when slope < 3 deg over a 3-step rolling window
    #   - References: Lied & Bakkehoi (1980), AvaFrame com2AB module
    
    # Load DEM
    dem, meta = read_raster(dem_path)
    transform = meta['transform']
    nodata = meta['nodata']
    step_m = float(min(abs(transform.a), abs(transform.e)))
    
    # Start at release centroid
    start = release_polygon.centroid
    x, y = start.x, start.y
    path = [(x, y)]
    
    # Helper: sample DEM at (x, y), returns None if outside or nodata
    def sample(px, py):
        row, col = world_to_pixel(px, py, transform)
        if row < 0 or row >= dem.shape[0] or col < 0 or col >= dem.shape[1]:
            return None
        v = dem[row, col]
        if nodata is not None and v == nodata:
            return None
        return float(v)
    
    # 8 compass directions
    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    recent_drops = []
    
    for _ in range(max_steps):
        z_here = sample(x, y)
        if z_here is None:
            break
        
        # Find direction of steepest descent
        best = None
        best_slope = 0.0
        for dx, dy in dirs:
            nx, ny = x + dx * step_m, y + dy * step_m
            z_next = sample(nx, ny)
            if z_next is None:
                continue
            length = math.hypot(dx * step_m, dy * step_m)
            slope = (z_here - z_next) / length if length > 0 else 0
            if slope > best_slope:
                best_slope = slope
                best = (nx, ny)
        
        # Stop if no downhill direction
        if best is None or best_slope <= 0:
            break
        
        x, y = best
        path.append((x, y))
        
        # Stop when 3-step average slope < 3 degrees (flow has flattened out)
        recent_drops.append(best_slope)
        if len(recent_drops) > 3:
            recent_drops.pop(0)
        if len(recent_drops) == 3 and np.mean(recent_drops) < math.tan(math.radians(3)):
            break
    
    # Degenerate fallback - shouldn't happen with real terrain
    if len(path) < 2:
        path.append((x + step_m, y - step_m))
    
    return LineString(path)


def compute_alpha_angle(flowline, dem_path):
    # Compute alpha-angle runout via the alpha-beta statistical model
    # (Lied & Bakkehoi 1980)
    # @param flowline: shapely LineString from release downhill
    # @param dem_path: Path to DEM raster
    # @returns: tuple (alpha_deg, alpha_point) where:
    #   - alpha_deg: alpha angle in degrees
    #   - alpha_point: shapely Point on flowline at the stopping location
    # @raises ValueError: if the flowline start is outside the DEM or the
    #                     beta point coincides with the start
    # beta point = first point where slope from start drops below
    #       10 deg; alpha = beta - 5 deg (generic offset, not regional)
    
    dem, meta = read_raster(dem_path)
    transform = meta['transform']
    
    # Helper: sample DEM at a Point
    def z_at(pt):
        row, col = world_to_pixel(pt.x, pt.y, transform)
        if row < 0 or row >= dem.shape[0] or col < 0 or col >= dem.shape[1]:
            return None
        return float(dem[row, col])
    
    coords = list(flowline.coords)
    start = Point(coords[0])
    z_start = z_at(start)
    if z_start is None:
        # throw error if start is outside DEM
        raise ValueError("Flowline start is outside DEM coverage")
    
    # Walk along flowline; beta = first point where slope-from-start < 10 deg
    beta_idx = None
    for i in range(5, len(coords)):
        pt = Point(coords[i])
        z = z_at(pt)
        if z is None:
            break
        dh = start.distance(pt)
        if dh < 1e-6:
            continue
        slope_deg = math.degrees(math.atan2(z_start - z, dh))
        if slope_deg < 10.0:
            beta_idx = i
            break
    
    # If slope never flattens, use end of flowline
    if beta_idx is None:
        beta_idx = len(coords) - 1
    
    beta_pt = Point(coords[beta_idx])
    z_beta = z_at(beta_pt) or z_start
    
    dh_total = start.distance(beta_pt)
    if dh_total < 1e-6:
        # throw error if beta point coincides with start
        raise ValueError("Degenerate flowline: beta point coincides with start")
    
    beta_deg = math.degrees(math.atan2(z_start - z_beta, dh_total))
    
    # Statistical offset: alpha = beta - 5 deg (generic; calibrate regionally)
    alpha_deg = max(beta_deg - 5.0, 10.0)
    
    # Project alpha-point on flowline by walking until profile rises above
    # the alpha-angle line from start
    tan_alpha = math.tan(math.radians(alpha_deg))
    last_valid = Point(coords[-1])
    for i in range(1, len(coords)):
        pt = Point(coords[i])
        z = z_at(pt)
        if z is None:
            break
        required_drop = start.distance(pt) * tan_alpha
        if (z_start - z) >= required_drop:
            last_valid = pt
            continue
        return alpha_deg, last_valid
    
    return alpha_deg, last_valid


def build_entrainment_polygon(release_polygon, flowline, alpha_point=None,
                              buffer_m=80.0):
    # Build the entrainment polygon as a corridor along the flowline
    # @param release_polygon: shapely Polygon of release area
    # @param flowline: shapely LineString from release downhill
    # @param alpha_point: shapely Point where flow stops (alpha-beta mode);
    #                     if None, use full flowline (flow-accumulation mode)
    # @param buffer_m: corridor half-width in meters
    # @returns: shapely Polygon (release polygon subtracted)
    # #   - Subtracting the release means entrainment starts downslope of the crown
    
    coords = list(flowline.coords)
    
    # Truncate to alpha-point if provided
    if alpha_point is not None:
        d_alpha = Point(coords[0]).distance(alpha_point)
        truncated = []
        cum = 0.0
        prev = None
        for c in coords:
            p = Point(c)
            if prev is not None:
                cum += prev.distance(p)
            truncated.append(c)
            if cum >= d_alpha:
                break
            prev = p
        if len(truncated) < 2:
            truncated = coords[:2]
        line = LineString(truncated)
    else:
        line = flowline
    
    # Buffer to corridor, subtract release area
    corridor = line.buffer(buffer_m, cap_style=2, join_style=2)
    entrainment = corridor.difference(release_polygon)
    
    # Pick largest piece if multipolygon
    if hasattr(entrainment, 'geoms'):
        entrainment = max(entrainment.geoms, key=lambda g: g.area)
    
    return entrainment


def auto_detect_entrainment(release_polygon, dem_path, polygon_out, thickness_out,
                            crs, depth_m, alpha_min=15.0, alpha_max=45.0,
                            alpha_buffer_m=15.0, fallback_buffer_m=50.0,
                            flowline_out=None, snow_height_raster=None,
                            min_slope_deg=0.0, raster_cap=True):
    # Auto-detect entrainment area for one scenario, write polygon + thickness raster
    # @param release_polygon: shapely Polygon of release area
    # @param dem_path: Path to DEM raster (also defines output grid)
    # @param polygon_out: Path to write entrainment polygon shapefile
    # @param thickness_out: Path to write entrainment thickness .asc
    # @param crs: CRS to assign to outputs
    # @param depth_m: scour cap (m) - max depth that can be entrained per cell.
    #                 acts as the upper bound when a DSM is supplied
    # @param alpha_min: lower bound on plausible alpha angle (deg)
    # @param alpha_max: upper bound on plausible alpha angle (deg)
    # @param alpha_buffer_m: corridor half-width when alpha-beta succeeds.
    #                        15 m matches a typical narrow path; widen for
    #                        broad slopes or fanned-out runouts.
    # @param fallback_buffer_m: corridor half-width if alpha-beta fails
    # @param flowline_out: Optional Path to also write the steepest-descent
    #                      flowline as GeoJSON (used by viewer for per-scenario paths)
    # @param snow_height_raster: Optional Path to a snow-height (HS) layer.
    #                            Accepts GeoTIFF / .asc (with CRS metadata) or
    #                            .npy (raw NumPy array, must be on DEM grid).
    #                            If supplied, per-cell entrainment depth is
    #                            min(snow_height_at_cell, depth_m) inside the
    #                            corridor and 0 outside. If None, falls back
    #                            to constant depth_m inside the corridor.
    # @param min_slope_deg: Minimum slope angle (deg) for entrainment.
    #                       Cells inside the corridor with slope below this
    #                       are zeroed in the thickness raster. 0.0 disables
    #                       the gate. Physically motivated by the observation
    #                       that entrainment happens on the track, not on
    #                       flat terrain at the toe of the runout.
    # @returns: dict with method_used, alpha_deg, area_m2, depth_m,
    #          depth_mode ('constant' or 'dsm_capped'), depth stats
    # @raises ValueError: if the entrainment polygon falls entirely outside
    #                     the DEM after clipping
    
    # Step 1: trace flowline by steepest descent
    flowline = steepest_descent_flowline(release_polygon, dem_path)
    
    # Optionally save the flowline for the viewer
    if flowline_out is not None:
        flowline_out = Path(flowline_out)
        flowline_out.parent.mkdir(parents=True, exist_ok=True)
        flow_gdf = gpd.GeoDataFrame(
            {'name': ['flowpath']}, geometry=[flowline], crs=crs
        )
        flow_gdf.to_file(flowline_out, driver='GeoJSON')
    
    # Step 2: try alpha-beta first, fall back to flow-accumulation buffer
    method_used = 'alpha_beta'
    alpha_deg = None
    polygon = None
    
    try:
        alpha_deg, alpha_pt = compute_alpha_angle(flowline, dem_path)
        
        # Sanity check the computed alpha
        if alpha_deg < alpha_min or alpha_deg > alpha_max:
            # throw error to trigger fallback
            raise ValueError(
                f"alpha={alpha_deg:.1f} deg outside plausible "
                f"range [{alpha_min}, {alpha_max}]"
            )
        
        polygon = build_entrainment_polygon(
            release_polygon, flowline, alpha_point=alpha_pt,
            buffer_m=alpha_buffer_m
        )
        logging.info(
            f"  Entrainment: alpha-beta corridor, alpha={alpha_deg:.1f} deg, "
            f"corridor half-width={alpha_buffer_m} m"
        )
    
    except Exception as e:
        # Fallback: flow-accumulation buffer (no alpha truncation)
        logging.warning(f"  alpha-beta failed ({e}); using flow-accumulation buffer")
        method_used = 'flow_accumulation'
        alpha_deg = None
        polygon = build_entrainment_polygon(
            release_polygon, flowline, alpha_point=None,
            buffer_m=fallback_buffer_m
        )
    
    # Clip entrainment polygon to DEM extent (with a 1-cell inset).
    # com1DFA rejects shapefiles whose vertices fall outside the DEM, even
    # by a fraction of a cell. The buffer step in build_entrainment_polygon
    # has no bounds awareness, so corridors near the DEM edge can poke out.
    # The inset gives a small safety margin around the strict bbox.
    from shapely.geometry import box
    with rasterio.open(dem_path) as dem_src:
        cell_size = abs(dem_src.transform.a)
        b = dem_src.bounds
    dem_bbox = box(
        b.left + cell_size,
        b.bottom + cell_size,
        b.right - cell_size,
        b.top - cell_size
    )
    
    if not dem_bbox.contains(polygon):
        original_area = polygon.area
        polygon = polygon.intersection(dem_bbox)
        # Pick largest piece if the clip created a multipolygon
        if hasattr(polygon, 'geoms'):
            polygon = max(polygon.geoms, key=lambda g: g.area)
        clipped_area = polygon.area
        pct_lost = 100.0 * (original_area - clipped_area) / original_area
        logging.warning(
            f"  Entrainment polygon clipped to DEM extent "
            f"({pct_lost:.1f}% area removed). DEM may not fully cover "
            f"the avalanche path - consider a larger DEM if entrainment "
            f"reaches the boundary."
        )
        if polygon.is_empty or polygon.area < 1.0:
            # throw error if clipping killed the polygon
            raise ValueError(
                "Entrainment polygon entirely outside DEM after clipping. "
                "Release area is too close to the DEM edge for any entrainment."
            )
    
    # Step 3: write polygon as shapefile
    # AvaFrame validates that entrainment polygons have a 'thickness'
    # attribute, even though we provide the actual values via raster
    # (entThFromFile=True). Field must exist to pass validation.
    # When DSM is in use, the polygon's thickness attribute is set to the
    # mean per-cell value so reports/QGIS see a representative number.
    polygon_out = Path(polygon_out)
    polygon_out.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 4: build the thickness raster
    # Two modes:
    #   constant   -> every cell in the corridor gets depth_m
    #   dsm_capped -> every cell gets min(HS_at_cell, depth_m) inside corridor,
    #                 0 outside. captures spatial variability in snow availability
    #                 while respecting the scour cap from the snowpack model.
    _, dem_meta = read_raster(dem_path)
    
    if snow_height_raster is not None and Path(snow_height_raster).exists():
        # DSM mode: build spatially-varying entrainment thickness
        result_meta = _build_dsm_capped_entrainment_raster(
            snow_height_path=snow_height_raster,
            corridor_polygon=polygon,
            # raster_cap=False: the raster already is the entrainable depth
            # (the release side's scour_depth.tif), so it is not capped again
            scour_cap_m=depth_m if raster_cap else float('inf'),
            dem_path=dem_path,
            out_path=thickness_out,
            min_slope_deg=min_slope_deg
        )
        depth_mode = 'dsm_capped'
        # Use mean per-cell value as the representative thickness for the
        # polygon attribute (AvaFrame's required 'thickness' field).
        repr_thickness = result_meta.get('depth_mean_m', depth_m)
        # If the corridor had no DSM coverage at all, fall back to the cap.
        if repr_thickness <= 0:
            repr_thickness = depth_m
    else:
        # Constant mode: depth_m everywhere inside corridor, 0 outside.
        # Slope gate (if enabled): also zero out cells with slope < min_slope_deg.
        mask = polygon_to_raster_mask(
            polygon, dem_meta['shape'], dem_meta['transform'],
            fill_value=depth_m
        ).astype('float32')
        if min_slope_deg > 0:
            slope_deg, _ = compute_slope_deg(dem_path)
            slope_ok = (slope_deg >= float(min_slope_deg))
            n_before = int((mask > 0).sum())
            mask = np.where(slope_ok, mask, 0.0).astype('float32')
            n_after = int((mask > 0).sum())
            pct_kept = 100.0 * n_after / n_before if n_before > 0 else 0.0
            logging.info(
                f'  Constant-mode entrainment: slope gate >= {min_slope_deg} deg '
                f'kept {n_after}/{n_before} cells ({pct_kept:.1f}%).'
            )
        write_raster(thickness_out, mask, dem_meta['profile'], driver='AAIGrid')
        depth_mode = 'constant'
        repr_thickness = depth_m
        n_corridor_total = int(polygon_to_raster_mask(
            polygon, dem_meta['shape'], dem_meta['transform'], fill_value=1.0
        ).sum())
        n_active = int((mask > 0).sum())
        result_meta = {
            'depth_min_m': depth_m,
            'depth_max_m': depth_m,
            'depth_mean_m': depth_m,
            'n_active_cells': n_active,
            'pct_corridor_with_dsm_data': 100.0,
            'pct_corridor_slope_ok': (
                100.0 * n_active / n_corridor_total
                if n_corridor_total > 0 else 0.0
            ),
        }
    
    # Write the polygon shapefile with the chosen representative thickness
    gdf = gpd.GeoDataFrame(
        {'name': ['entrainment'], 'thickness': [float(repr_thickness)]},
        geometry=[polygon],
        crs=crs
    )
    gdf.to_file(polygon_out, driver='ESRI Shapefile')
    
    # Build the return dict, carrying through DSM stats when present
    out = {
        'method_used': method_used,
        'alpha_deg': alpha_deg,
        'area_m2': float(polygon.area),
        'depth_m': depth_m,
        'depth_mode': depth_mode,
    }
    out.update(result_meta)
    return out


def _build_dsm_capped_entrainment_raster(snow_height_path, corridor_polygon,
                                          scour_cap_m, dem_path, out_path,
                                          min_slope_deg=0.0):
    # Build a per-cell entrainment-depth raster from a snow-height DSM
    # @param snow_height_path: Path to snow height (HS) raster (.tif/.asc) or
    #                          a .npy NumPy array already on the DEM grid
    # @param corridor_polygon: shapely Polygon defining the avalanche corridor
    # @param scour_cap_m: float - scour cap (m) limiting per-cell depth
    # @param dem_path: Path to DEM (used for grid alignment)
    # @param out_path: Path to write the resulting .asc raster
    # @returns: dict with depth_min_m, depth_max_m, depth_mean_m,
    #          n_active_cells, coverage percentages
    # @raises ValueError: if a .npy input does not match the DEM grid shape
    # per cell -> min(HS, scour_cap) inside the corridor, 0 outside;
    #       .tif/.asc inputs are resampled onto the DEM grid, .npy inputs
    #       must already be on it
    
    snow_height_path = Path(snow_height_path)
    _, dem_meta = read_raster(dem_path)
    tmp_resampled = None
    
    # Step 1: load HS onto the DEM grid
    if snow_height_path.suffix.lower() == '.npy':
        # .npy has no spatial metadata - assume it was generated on the DEM grid
        hs_data = np.load(snow_height_path)
        if hs_data.shape != dem_meta['shape']:
            # throw error if .npy shape doesn't match DEM
            raise ValueError(
                f"Snow height .npy shape {hs_data.shape} does not match "
                f"DEM shape {dem_meta['shape']}. .npy files carry no CRS or "
                f"transform - they must be pre-aligned to the DEM grid. "
                f"Either regenerate the .npy on the DEM grid, or convert it "
                f"to a GeoTIFF with explicit CRS/transform."
            )
        hs_data = hs_data.astype('float32')
    else:
        # GeoTIFF / .asc route: resample onto DEM grid via rasterio
        # AvaFrame requires entrainment thickness raster to share the DEM
        # grid (cell size, origin, extent). resample_to_reference_grid
        # handles that.
        tmp_resampled = out_path.parent / '_tmp_hs_on_dem_grid.tif'
        resample_to_reference_grid(
            src_path=snow_height_path,
            ref_path=dem_path,
            dst_path=tmp_resampled,
            resampling_method='bilinear',
            fill_value=0.0
        )
        hs_data, _ = read_raster(tmp_resampled)
    
    # Step 2: build a corridor mask aligned to the DEM grid
    # 1.0 inside corridor, 0.0 outside.
    corridor_mask = polygon_to_raster_mask(
        corridor_polygon,
        dem_meta['shape'],
        dem_meta['transform'],
        fill_value=1.0
    ).astype('float32')
    n_corridor_total = int(corridor_mask.sum())
    
    # Step 2b: optional slope gate. Zero out corridor cells below the
    # minimum slope angle. min_slope_deg=0 disables the gate.
    slope_ok_mask = None
    if min_slope_deg > 0:
        slope_deg, _ = compute_slope_deg(dem_path)
        slope_ok_mask = (slope_deg >= float(min_slope_deg)).astype('float32')
        corridor_mask = corridor_mask * slope_ok_mask
    n_corridor_after_slope = int(corridor_mask.sum())
    
    # Step 3: compute per-cell entrainment depth = min(HS, scour_cap)
    # Negative values in HS (nodata sentinels, e.g. -9999) get clamped to 0
    hs_clean = np.where(hs_data > 0, hs_data, 0.0).astype('float32')
    capped = np.minimum(hs_clean, float(scour_cap_m))
    
    # Step 3b: track DSM coverage over the corridor (before the multiply).
    # A cell has DSM coverage if HS data is finite and > 0.
    hs_has_data = (hs_clean > 0).astype('float32')
    n_corridor_with_dsm = int(((corridor_mask > 0) & (hs_has_data > 0)).sum())
    pct_with_dsm = (
        100.0 * n_corridor_with_dsm / n_corridor_after_slope
        if n_corridor_after_slope > 0 else 0.0
    )
    
    # Step 4: zero out anywhere outside the (slope-gated) corridor
    entrainment_depth = capped * corridor_mask
    
    # Step 5: write as .asc matching DEM grid
    write_raster(
        out_path,
        entrainment_depth,
        dem_meta['profile'],
        driver='AAIGrid'
    )
    
    # Clean up temp resampled file (only created in the .tif/.asc route)
    if tmp_resampled is not None:
        tmp_resampled.unlink(missing_ok=True)
    
    # Stats over the corridor cells only (where mask > 0 and depth > 0)
    pct_slope_kept = (
        100.0 * n_corridor_after_slope / n_corridor_total
        if n_corridor_total > 0 else 0.0
    )
    active = entrainment_depth[entrainment_depth > 0]
    if active.size > 0:
        stats = {
            'depth_min_m': float(active.min()),
            'depth_max_m': float(active.max()),
            'depth_mean_m': float(active.mean()),
            'n_active_cells': int(active.size),
            'pct_corridor_with_dsm_data': float(pct_with_dsm),
            'pct_corridor_slope_ok': float(pct_slope_kept),
        }
    else:
        # Degenerate case: no active cells (no DSM coverage, or slope
        # gate killed the whole corridor, or both)
        stats = {
            'depth_min_m': 0.0,
            'depth_max_m': 0.0,
            'depth_mean_m': 0.0,
            'n_active_cells': 0,
            'pct_corridor_with_dsm_data': float(pct_with_dsm),
            'pct_corridor_slope_ok': float(pct_slope_kept),
        }
        logging.warning(
            f"  DSM entrainment: zero active cells in corridor. "
            f"DSM coverage = {pct_with_dsm:.1f}%, "
            f"slope-gate kept = {pct_slope_kept:.1f}% of corridor."
        )
    
    logging.info(
        f"  DSM entrainment: {stats['n_active_cells']} cells, "
        f"depth min/mean/max = {stats['depth_min_m']:.2f}/"
        f"{stats['depth_mean_m']:.2f}/{stats['depth_max_m']:.2f} m "
        f"(scour cap = {scour_cap_m:.2f} m, "
        f"DSM coverage = {pct_with_dsm:.1f}%, "
        f"slope-gate kept = {pct_slope_kept:.1f}%)"
    )

    return stats


def build_polygon_entrainment(polygon_path, release_polygon, dem_path,
                              polygon_out, thickness_out, crs, depth_m,
                              min_slope_deg=0.0, flowline_out=None):
    # Entrainment from a hand-drawn polygon at a constant depth
    # @param polygon_path: Path to the entrainment polygon (any geopandas
    #                      readable file); features are dissolved into one
    # @param release_polygon: shapely Polygon of the release area, removed
    #                         from the entrainment area so entrainment
    #                         starts below the crown
    # @param dem_path: Path to DEM raster (defines the output grid)
    # @param polygon_out: Path to write the entrainment shapefile
    # @param thickness_out: Path to write the thickness .asc
    # @param crs: CRS to assign to outputs
    # @param depth_m: constant entrainment depth (m) inside the polygon
    # @param min_slope_deg: cells below this slope are zeroed; 0 disables
    # @param flowline_out: optional Path to also write the steepest-descent
    #                      flowline for the viewer
    # @returns: dict shaped like auto_detect_entrainment's return
    # @raises ValueError: if the polygon is empty after removing the release
    #                     area and clipping to the DEM
    # the polygon is used as drawn apart from the release subtraction
    #       and a one-cell inset at the DEM edge (com1DFA rejects features
    #       that touch the DEM boundary)

    from shapely.geometry import box
    from shapely.ops import unary_union

    gdf = gpd.read_file(polygon_path)
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    polygon = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
    polygon = polygon.buffer(0).difference(release_polygon)

    with rasterio.open(dem_path) as dem_src:
        cell_size = abs(dem_src.transform.a)
        b = dem_src.bounds
    dem_bbox = box(b.left + cell_size, b.bottom + cell_size,
                   b.right - cell_size, b.top - cell_size)
    polygon = polygon.intersection(dem_bbox)
    if hasattr(polygon, 'geoms'):
        polygon = max(polygon.geoms, key=lambda g: g.area)
    if polygon.is_empty or polygon.area < 1.0:
        # throw error if nothing is left to entrain
        raise ValueError(
            f"Entrainment polygon {Path(polygon_path).name} is empty after "
            f"removing the release area and clipping to the DEM"
        )

    # Flowline for the viewer, same as the other modes
    if flowline_out is not None:
        flowline = steepest_descent_flowline(release_polygon, dem_path)
        flowline_out = Path(flowline_out)
        flowline_out.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({'name': ['flowpath']}, geometry=[flowline],
                         crs=crs).to_file(flowline_out, driver='GeoJSON')

    # Constant thickness inside the polygon, slope gate optional
    _, dem_meta = read_raster(dem_path)
    mask = polygon_to_raster_mask(
        polygon, dem_meta['shape'], dem_meta['transform'], fill_value=depth_m
    ).astype('float32')
    n_total = int((mask > 0).sum())
    if min_slope_deg > 0:
        slope_deg, _ = compute_slope_deg(dem_path)
        mask = np.where(slope_deg >= float(min_slope_deg), mask, 0.0).astype('float32')
    n_active = int((mask > 0).sum())
    write_raster(thickness_out, mask, dem_meta['profile'], driver='AAIGrid')

    polygon_out = Path(polygon_out)
    polygon_out.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {'name': ['entrainment'], 'thickness': [float(depth_m)]},
        geometry=[polygon], crs=crs
    ).to_file(polygon_out, driver='ESRI Shapefile')

    logging.info(
        f"  Entrainment: polygon {Path(polygon_path).name}, "
        f"area={polygon.area:.0f} m2, depth={depth_m:.2f} m, "
        f"slope gate kept {n_active}/{n_total} cells"
    )
    return {
        'method_used': 'polygon',
        'alpha_deg': None,
        'area_m2': float(polygon.area),
        'depth_m': depth_m,
        'depth_mode': 'constant',
        'depth_min_m': depth_m,
        'depth_max_m': depth_m,
        'depth_mean_m': depth_m,
        'n_active_cells': n_active,
        'pct_corridor_with_dsm_data': 100.0,
        'pct_corridor_slope_ok': 100.0 * n_active / n_total if n_total else 0.0,
    }


def analyze_reference_points(reach_mask_path, pft_path, pfv_path,
                             refpoints_path, release_centroid, radius_m=10.0):
    # Test each reference point (tower, bench, road, ...) against a reach mask
    # @param reach_mask_path: Path to the scenario's binary reach mask
    # @param pft_path: Path to peak flow depth (sampled at the point)
    # @param pfv_path: Path to peak velocity (sampled at the point)
    # @param refpoints_path: Path to the point layer with a 'ref_name' field
    # @param release_centroid: shapely Point, for distance ordering
    # @param radius_m: a point is reached when any reach cell lies within
    #                  this distance of it
    # @returns: list of dicts, one per point, ordered by distance from the
    #          release: ref_name, reached, distance_short_m (0 when
    #          reached), distance_from_release_m, depth_m, velocity_ms
    # the file may hold other fields; only ref_name is read

    pts = gpd.read_file(refpoints_path)
    if 'ref_name' not in pts.columns:
        # throw error if the required field is missing
        raise ValueError(f"{refpoints_path} has no 'ref_name' field")

    reach, meta = read_raster(reach_mask_path)
    transform = meta['transform']
    if pts.crs is not None and meta['crs'] is not None and pts.crs != meta['crs']:
        pts = pts.to_crs(meta['crs'])

    if reach.any():
        ys, xs = np.where(reach.astype(bool))
        wx, wy = pixels_to_world(ys, xs, transform)
        if len(wx) > 20000:
            idx = np.linspace(0, len(wx) - 1, 20000, dtype=int)
            wx, wy = wx[idx], wy[idx]
    else:
        wx = wy = np.array([])

    out = []
    for _, row in pts.iterrows():
        pt = row.geometry
        if pt is None or pt.is_empty:
            continue
        if pt.geom_type != 'Point':
            pt = pt.centroid
        name = str(row['ref_name'])
        if len(wx):
            d = np.hypot(wx - pt.x, wy - pt.y)
            min_dist = float(d.min())
        else:
            min_dist = float(release_centroid.distance(pt))
        reached = bool(min_dist <= radius_m)
        depth = vel = None
        if reached:
            depth = sample_raster_at_xy(pft_path, [(pt.x, pt.y)])[0]
            vel = sample_raster_at_xy(pfv_path, [(pt.x, pt.y)])[0]
        out.append({
            'ref_name': name,
            'reached': reached,
            'distance_short_m': 0.0 if reached else min_dist,
            'distance_from_release_m': float(release_centroid.distance(pt)),
            'depth_m': depth,
            'velocity_ms': vel,
        })
    out.sort(key=lambda r: r['distance_from_release_m'])
    return out