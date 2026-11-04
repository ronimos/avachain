"""
build_viewer.py

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Renders an interactive Leaflet viewer (viewer.html) from the ensemble
    outputs. Light-mode operational instrument layout with collapsible
    panels (title block, layers, run summary, legend) over a hillshade
    basemap rendered from the project DEM. Self-contained, offline-safe.

    Layers (toggleable):
        - Hillshade basemap (from project DEM, always on)
        - Probability raster (image overlay, semi-transparent)
        - Each contour level as its own outline layer (viridis-colored to
          match the heatmap, not monochrome)
        - Envelope outline
        - Road centerline (always on, orange)
        - Impact points (clickable markers with per-scenario popups)
        - Per-scenario release / flowpath / reach polygons, grouped under
          a collapsible accordion with filter input and show/hide-all
        - Forest resistance polygon (off by default)

    All raster overlays are reprojected to EPSG:4326 and embedded as
    base64-encoded PNG data URIs so viewer.html is fully self-contained.

    No command line. Imported by avalanche_runout.py.

Requirements:
    - folium, branca, geopandas, rasterio, pillow, pyproj, matplotlib
"""

# --------- Load Libraries --------
import base64
import json
import logging
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import LineString
import folium
from folium import Element
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LightSource
from PIL import Image
from pyproj import Transformer

import avalanche_helpers as h


# Pipeline version stamp (shown in info-drawer header)
PIPELINE_VERSION = '1.0'


# Brand palette - used for chrome accents
BRAND = {
    'orange': '#E87722',
    'crimson': '#A6192E',
    'deep_teal': '#005F73',
    'sage': '#7A9F3A',
    'goldenrod': '#E9A52C',
    'slate_blue': '#5B7A9E',
    'forest_green': '#2E7D5B',
    'sky_blue': '#5DADE2',
    'dusty_rose': '#C97A7E',
    'olive': '#4A5A24',
    'steel_gray': '#9CA3A8',
}

# Per-scenario palette - perceptually distinct, picked from tab20 + extensions
# to scale to ~70 scenarios without collisions. Cycled when N > len(palette).
SCENARIO_COLORS = [
    '#A6192E', '#005F73', '#2E7D5B', '#E9A52C', '#5B7A9E', '#7A9F3A',
    '#C97A7E', '#4A5A24', '#E87722', '#5DADE2', '#8E44AD', '#16A085',
    '#D35400', '#2C3E50', '#C0392B', '#27AE60', '#9B59B6', '#F39C12',
    '#1ABC9C', '#34495E', '#E67E22', '#3498DB', '#E74C3C', '#2ECC71',
    '#9B7B47', '#6C5B7B', '#355C7D', '#F67280', '#C06C84', '#6A5ACD',
    '#FF8C42', '#BB3E5D', '#4D7C0F', '#0E7490', '#A16207', '#7C2D12',
    '#0F766E', '#86198F', '#B91C1C', '#15803D',
]




def reproject_raster_to_wgs84(src_path, dst_path, resampling=Resampling.nearest):
    # Reproject a raster to EPSG:4326 (WGS84) for Leaflet image overlay
    # @param src_path: Path to source raster (any CRS)
    # @param dst_path: Path to write WGS84 raster
    # @param resampling: rasterio Resampling enum (default: nearest)
    # @returns: Path to written raster

    with rasterio.open(src_path) as src:
        if src.crs.to_string() == 'EPSG:4326':
            import shutil
            shutil.copy2(src_path, dst_path)
            return dst_path

        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs='EPSG:4326', transform=transform,
            width=width, height=height
        )

        with rasterio.open(dst_path, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs='EPSG:4326',
                resampling=resampling
            )

    return dst_path


def array_to_data_uri(rgba_array):
    # Encode an RGBA numpy array as a base64 PNG data URI
    # @param rgba_array: HxWx4 uint8 array
    # @returns: str - data URI
    img = Image.fromarray(rgba_array, mode='RGBA')
    buf = BytesIO()
    img.save(buf, format='PNG')
    png_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{png_b64}'


def raster_bounds_wgs84(raster_path):
    # Compute Leaflet imageOverlay bounds
    # @param raster_path: Path to a WGS84 raster
    # @returns: list [[south, west], [north, east]]
    with rasterio.open(raster_path) as src:
        t = src.transform
        left, top = t.c, t.f
        right = left + src.width * t.a
        bottom = top + src.height * t.e
    return [[bottom, left], [top, right]]


def hillshade_array(elevation, cell_size_m, azdeg=315, altdeg=45):
    # Compute a hillshade from an elevation array
    # @param elevation: 2D array of elevations
    # @param cell_size_m: cell size in meters
    # @param azdeg: light azimuth (degrees)
    # @param altdeg: light altitude (degrees)
    # @returns: 2D array of shade values in [0, 1]
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    return ls.hillshade(elevation, dx=cell_size_m, dy=cell_size_m)


def render_hillshade_data_uri(dem_path, outputs_dir, vert_exag=1.3):
    # Render a DEM as a grayscale hillshade PNG data URI for the basemap
    # @param dem_path: Path to the DEM
    # @param outputs_dir: dir for the temp reprojected raster
    # @param vert_exag: vertical exaggeration for the shading
    # @returns: tuple (data_uri, bounds)

    dem_wgs = outputs_dir / '_tmp_dem_wgs84.tif'
    reproject_raster_to_wgs84(dem_path, dem_wgs, resampling=Resampling.bilinear)

    elevation, meta = h.read_raster(dem_wgs)
    nodata = meta['nodata']

    if nodata is not None:
        bad = (elevation == nodata)
        if bad.any():
            valid_mean = elevation[~bad].mean() if (~bad).any() else 0.0
            elevation = np.where(bad, valid_mean, elevation)
    else:
        bad = np.zeros_like(elevation, dtype=bool)

    t = meta['transform']
    mid_lat_rad = np.radians((t.f + (elevation.shape[0] / 2) * t.e))
    deg_to_m_x = 111_320 * np.cos(mid_lat_rad)
    deg_to_m_y = 111_320
    cell_m = (abs(t.a) * deg_to_m_x + abs(t.e) * deg_to_m_y) / 2

    shade = hillshade_array(elevation * vert_exag, cell_size_m=cell_m)

    # Neutral gray: shade [0..1] maps directly to gray value [0..255]
    rgba = np.zeros((*shade.shape, 4), dtype=np.uint8)
    gray = (shade * 255).astype(np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(bad, 0, 255)

    data_uri = array_to_data_uri(rgba)
    bounds = raster_bounds_wgs84(dem_wgs)
    dem_wgs.unlink(missing_ok=True)

    return data_uri, bounds


def render_probability_data_uri(prob_path, outputs_dir, colormap='viridis'):
    # Render a probability raster as a colored semi-transparent PNG data URI
    # @param prob_path: Path to the probability raster
    # @param outputs_dir: dir for the temp reprojected raster
    # @param colormap: matplotlib colormap name
    # @returns: tuple (data_uri, bounds)

    prob_wgs = outputs_dir / '_tmp_probability_wgs84.tif'
    reproject_raster_to_wgs84(prob_path, prob_wgs, resampling=Resampling.nearest)

    data, meta = h.read_raster(prob_wgs)
    nodata = meta['nodata']

    if nodata is not None:
        mask = (data == nodata) | (data < 0)
    else:
        mask = data < 0

    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)
    rgba = cmap(norm(data))

    # Alpha curve: sqrt makes high-P pop, low-P fade.
    rgba[..., 3] = np.where(mask, 0, np.clip(data, 0, 1) ** 0.5)
    rgba[mask] = 0

    rgba_8bit = (rgba * 255).astype(np.uint8)

    data_uri = array_to_data_uri(rgba_8bit)
    bounds = raster_bounds_wgs84(prob_wgs)
    prob_wgs.unlink(missing_ok=True)

    return data_uri, bounds


def _viridis_hex(t):
    # Map t in [0,1] to a viridis hex color string
    # @param t: float in [0, 1]
    # @returns: str - hex color
    cmap = plt.get_cmap('viridis')
    r, g, b, _ = cmap(float(np.clip(t, 0.0, 1.0)))
    return '#{:02x}{:02x}{:02x}'.format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def add_hillshade_basemap(m, dem_path, outputs_dir):
    # Add a hillshade DEM as the basemap (no external tiles)
    # @param m: folium Map
    # @param dem_path: Path to the DEM
    # @param outputs_dir: dir for the temp reprojected raster
    # @returns: tuple (layer, bounds) for chrome toggle registration
    data_uri, bounds = render_hillshade_data_uri(dem_path, outputs_dir)
    layer = folium.raster_layers.ImageOverlay(
        name='Hillshade',
        image=data_uri,
        bounds=bounds,
        opacity=1.0,
        z_index=1
    )
    layer.add_to(m)
    return layer, bounds


def add_esri_imagery_basemap(m):
    # Add Esri World Imagery as an optional satellite basemap
    # @param m: folium Map
    # @returns: the folium TileLayer instance
    layer = folium.raster_layers.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
              'World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr=(
            'Tiles &copy; Esri, Maxar, Earthstar Geographics, '
            'and the GIS User Community'
        ),
        name='Esri imagery',
        overlay=True,
        control=True,
        show=False,
        max_zoom=19,
        z_index=2,
    )
    layer.add_to(m)
    return layer


def add_probability_overlay(m, prob_path, outputs_dir, colormap):
    # Add the probability raster as a base64-embedded image overlay
    # @param m: folium Map
    # @param prob_path: Path to the probability raster
    # @param outputs_dir: dir for the temp reprojected raster
    # @param colormap: matplotlib colormap name
    # @returns: the folium ImageOverlay instance
    data_uri, bounds = render_probability_data_uri(
        prob_path, outputs_dir, colormap=colormap
    )
    layer = folium.raster_layers.ImageOverlay(
        name='Probability heatmap',
        image=data_uri,
        bounds=bounds,
        opacity=0.7,
        z_index=2
    )
    layer.add_to(m)
    return layer


def add_contour_layers(m, contours_path):
    # Add contour outlines, one Leaflet layer per probability level
    # @param m: folium Map
    # @param contours_path: Path to contours geojson
    # @returns: list of (display_name, folium_layer) tuples

    out = []
    try:
        contours_gdf = gpd.read_file(contours_path).to_crs('EPSG:4326')
    except Exception as e:
        logging.debug(f"No contours in viewer: {e}")
        return out

    for level in sorted(contours_gdf['level'].unique()):
        subset = contours_gdf[contours_gdf['level'] == level]
        line_color = _viridis_hex(float(level))
        layer_name = f"Contour P >= {level:.2f}"

        layer = folium.GeoJson(
            subset.to_json(),
            name=layer_name,
            style_function=lambda feat, col=line_color, lvl=level: {
                'color': col,
                'weight': 1.2 + lvl * 1.4,
                'fill': False,
                'opacity': 0.7 + lvl * 0.25
            },
            show=(level >= 0.5)
        )
        layer.add_to(m)
        out.append((layer_name, layer))
    return out


def add_envelope_layer(m, envelope_path, threshold):
    # Envelope outline - crimson dashed
    # @param m: folium Map
    # @param envelope_path: Path to envelope geojson
    # @param threshold: envelope probability threshold (for the layer name)
    # @returns: tuple (layer, env_wgs_gdf)
    env_wgs = gpd.read_file(envelope_path).to_crs('EPSG:4326')
    layer = folium.GeoJson(
        env_wgs.to_json(),
        name=f"Envelope (P >= {threshold})",
        style_function=lambda _: {
            'color': BRAND['crimson'],
            'weight': 2.2,
            'fill': False,
            'opacity': 0.95,
            'dashArray': '6, 3'
        }
    )
    layer.add_to(m)
    return layer, env_wgs


def add_road_layer(m, road_path):
    # Road centerline - signal orange
    # @param m: folium Map
    # @param road_path: Path to road shapefile
    # @returns: tuple (layer, road_wgs_gdf)
    road_wgs = gpd.read_file(road_path).to_crs('EPSG:4326')
    layer = folium.GeoJson(
        road_wgs.to_json(),
        name='Road',
        style_function=lambda _: {
            'color': BRAND['orange'],
            'weight': 3.5,
            'opacity': 1.0
        }
    )
    layer.add_to(m)
    return layer, road_wgs


def add_forest_layer(m, forest_path):
    # Forest resistance area (off by default)
    # @param m: folium Map
    # @param forest_path: Path to forest shapefile
    # @returns: folium layer if added, else None
    if not Path(forest_path).exists():
        return None
    forest_wgs = gpd.read_file(forest_path).to_crs('EPSG:4326')
    layer = folium.GeoJson(
        forest_wgs.to_json(),
        name='Forest resistance',
        style_function=lambda _: {
            'color': BRAND['forest_green'],
            'weight': 1,
            'fillOpacity': 0.18
        },
        show=False
    )
    layer.add_to(m)
    return layer


def add_impact_points(m, impact_points, source_crs):
    # Add impact points as solid crimson dots with popups
    # @param m: folium Map
    # @param impact_points: list of impact point dicts
    # @param source_crs: CRS of the impact point coordinates
    # @returns: the folium FeatureGroup, or None if no points were added

    if not impact_points:
        return None

    transformer = Transformer.from_crs(source_crs, 'EPSG:4326', always_xy=True)
    fg = folium.FeatureGroup(name='Impact points', show=True)

    for pt in impact_points:
        if not pt.get('xy'):
            continue

        lon, lat = transformer.transform(pt['xy'][0], pt['xy'][1])

        popup_html = (
            f'<div class="popup-title">{pt["scenario"]}</div>'
            f'<div class="popup-row"><span class="k">Flow depth</span>'
            f'<span class="v">{pt["depth_m"]:.2f} m</span></div>'
            f'<div class="popup-row"><span class="k">Velocity</span>'
            f'<span class="v">{pt["velocity_ms"]:.2f} m/s</span></div>'
            f'<div class="popup-row"><span class="k">Chainage</span>'
            f'<span class="v">{pt["chainage_m"]:.0f} m</span></div>'
        )

        # Solid red dot
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=BRAND['crimson'],
            weight=0,
            fill=True,
            fill_color=BRAND['crimson'],
            fill_opacity=1.0,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=pt['scenario']
        ).add_to(fg)

    fg.add_to(m)
    return fg


def add_per_scenario_layers(m, scenarios, show_default=False):
    # Add per-scenario release + reach as one FeatureGroup per scenario.
    # The custom chrome panel toggles these, not folium's LayerControl.
    # @param m: folium Map
    # @param scenarios: list of scenario dicts
    # @param show_default: show scenario layers on load
    # @returns: list of dicts [{sid, color, layer_name}] for chrome wiring

    out = []
    for i, sc in enumerate(scenarios):
        sid = sc['scenario_id']
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]

        layer_name = f"_scenario_{sid}"
        fg = folium.FeatureGroup(name=layer_name, show=show_default, overlay=True, control=False)
        added_anything = False

        filled_path = Path(sc.get('work_dir') or sc['scenario_dir']) / 'outputs' / 'release_filled.geojson'
        rel_filled_src = filled_path if filled_path.exists() else sc['release_geojson']
        try:
            filled_gdf = gpd.read_file(rel_filled_src).to_crs('EPSG:4326')
            folium.GeoJson(
                filled_gdf.to_json(),
                style_function=lambda _, c=color: {
                    'color': c,
                    'weight': 2,
                    'fillColor': c,
                    'fillOpacity': 0.35
                },
                tooltip=f"{sid} release (filled)"
            ).add_to(fg)
            added_anything = True
        except Exception as e:
            logging.debug(f"Skip filled release for {sid}: {e}")

        try:
            orig_gdf = gpd.read_file(sc['release_geojson']).to_crs('EPSG:4326')
            folium.GeoJson(
                orig_gdf.to_json(),
                style_function=lambda _, c=color: {
                    'color': c,
                    'weight': 2,
                    'dashArray': '6, 4',
                    'fillOpacity': 0.0
                },
                tooltip=f"{sid} release (original)"
            ).add_to(fg)
            added_anything = True
        except Exception as e:
            logging.debug(f"Skip original release for {sid}: {e}")

        reach_path = Path(sc.get('work_dir') or sc['scenario_dir']) / 'outputs' / 'reach_polygon.geojson'
        if reach_path.exists():
            try:
                reach_gdf = gpd.read_file(reach_path).to_crs('EPSG:4326')
                folium.GeoJson(
                    reach_gdf.to_json(),
                    style_function=lambda _, c=color: {
                        'color': c,
                        'weight': 1,
                        'fillColor': c,
                        'fillOpacity': 0.15
                    },
                    tooltip=f"{sid} runout"
                ).add_to(fg)
                added_anything = True
            except Exception as e:
                logging.debug(f"Skip reach for {sid}: {e}")

        if added_anything:
            fg.add_to(m)
            out.append({
                'sid': sid,
                'color': color,
                'layer_name': layer_name,
                'shown_by_default': show_default,
                'fg': fg,
            })

    return out


def _build_run_summary(scenarios, impact_points, outputs_dir, config):
    # Aggregate run-level numbers for the info drawer.
    # @param scenarios: list of scenario dicts
    # @param impact_points: list of impact point dicts (from road analysis)
    # @param outputs_dir: ensemble outputs dir
    # @param config: top-level config dict
    # @returns: dict with all values needed by the chrome panels

    max_v = 0.0
    max_d = 0.0
    max_p = 0.0
    max_runout = 0.0

    # Find the highest-probability scenario by scenario_probability weight
    most_likely = None
    most_likely_weight = -1.0

    per_scenario_summaries = {}

    for sc in scenarios:
        sj = Path(sc.get('work_dir') or sc['scenario_dir']) / 'outputs' / 'summary.json'
        if not sj.exists():
            continue
        try:
            with open(sj) as f:
                s = json.load(f)
        except Exception:
            continue

        per_scenario_summaries[sc['scenario_id']] = s

        max_v = max(max_v, float(s.get('peak_velocity_max_ms', 0) or 0))
        max_d = max(max_d, float(s.get('peak_flow_depth_max_m', 0) or 0))
        max_p = max(max_p, float(s.get('peak_pressure_max_kpa', 0) or 0))
        max_runout = max(max_runout, float(s.get('runout_max_distance_m', 0) or 0))

        weight = float(s.get('params', {}).get('scenario_probability', 0) or 0)
        if weight > most_likely_weight:
            most_likely_weight = weight
            most_likely = {
                'sid': sc['scenario_id'],
                'weight': weight,
                'reached_road': bool(s.get('road', {}).get('reached', False)),
                'runout_m': float(s.get('runout_max_distance_m', 0) or 0),
                'velocity_at_road_ms': s.get('road', {}).get('velocity_at_road_ms'),
            }

    # Reach-to-road counts
    n_total = len(scenarios)
    n_reached = sum(1 for sid, s in per_scenario_summaries.items()
                    if s.get('road', {}).get('reached', False))

    # Weighted P(reach road) from road_impact_summary.json if present
    p_reach_road = None
    p_reach_road_weighted = None
    rs_path = Path(outputs_dir) / 'road_impact_summary.json'
    if rs_path.exists():
        try:
            with open(rs_path) as f:
                rs = json.load(f)
            p_reach_road = rs.get('fraction_reached')
            p_reach_road_weighted = rs.get('fraction_reached_weighted')
        except Exception:
            pass

    # Envelope area (project CRS, m^2)
    envelope_area_m2 = 0.0
    env_path = Path(outputs_dir) / 'envelope.geojson'
    if env_path.exists():
        try:
            env_gdf = gpd.read_file(env_path)
            envelope_area_m2 = float(env_gdf.geometry.area.sum())
        except Exception:
            pass

    # Road length affected: length of road centerline inside envelope
    road_length_affected_m = 0.0
    road_path = Path(config['paths']['road'])
    if env_path.exists() and road_path.exists():
        try:
            env_gdf = gpd.read_file(env_path)
            road_gdf = gpd.read_file(road_path)
            if env_gdf.crs != road_gdf.crs:
                road_gdf = road_gdf.to_crs(env_gdf.crs)
            env_union = env_gdf.unary_union
            for geom in road_gdf.geometry:
                if geom is None:
                    continue
                inter = geom.intersection(env_union)
                if not inter.is_empty:
                    road_length_affected_m += float(inter.length)
        except Exception as e:
            logging.debug(f"Road-length-affected calc skipped: {e}")

    # Model date comes from config['display']['model_date'] when set.
    model_date = config.get('display', {}).get('model_date') or 'YYYY\u00b7MM\u00b7DD'
    run_date = date.today().strftime('%Y\u00b7%m\u00b7%d')
    project_name = config.get('display', {}).get('project_name') or 'Avalanche Runout'
    project_subtitle = config.get('display', {}).get('subtitle') or ''

    return {
        'project_name': project_name,
        'project_subtitle': project_subtitle,
        'model_date': model_date,
        'run_date': run_date,
        'n_scenarios': n_total,
        'n_reached': n_reached,
        'p_reach_road': p_reach_road,
        'p_reach_road_weighted': p_reach_road_weighted,
        'envelope_area_m2': envelope_area_m2,
        'road_length_affected_m': road_length_affected_m,
        'max_runout_m': max_runout,
        'max_velocity_ms': max_v,
        'max_flow_depth_m': max_d,
        'max_pressure_kpa': max_p,
        'envelope_threshold': config['probability']['envelope_min_probability'],
        'friction_model': config['com1dfa']['friction_model'],
        'mesh_cell_size_m': config['com1dfa']['mesh_cell_size_m'],
        'entrainment_enabled': config['com1dfa']['sim_type'] in ('ent', 'entres'),
        'most_likely': most_likely,
    }


def _fmt_int(x):
    # Format a number as a comma-grouped integer string
    # @param x: numeric value
    # @returns: str
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return '\u2014'


def _fmt_float(x, decimals=2, unit=''):
    # Format a number with fixed decimals and optional unit
    # @param x: numeric value or None
    # @param decimals: decimal places
    # @param unit: unit string appended after a space
    # @returns: str
    if x is None:
        return '\u2014'
    try:
        v = float(x)
    except Exception:
        return '\u2014'
    if unit:
        return f"{v:.{decimals}f} {unit}"
    return f"{v:.{decimals}f}"


def _build_chrome_html(summary, scenario_layers, contour_levels):
    # Build the chrome HTML/CSS/JS block injected into the Leaflet page:
    # four collapsible panels plus the layer-toggle wiring.
    # @param summary: dict from _build_run_summary
    # @param scenario_layers: list from add_per_scenario_layers
    # @param contour_levels: sorted list of contour probability levels
    # @returns: str - HTML block

    ml = summary.get('most_likely') or {}
    ml_sid = ml.get('sid') or '\u2014'
    ml_weight_pct = (ml.get('weight') or 0) * 100.0
    ml_reached = ml.get('reached_road')
    ml_runout = ml.get('runout_m', 0)
    ml_vel = ml.get('velocity_at_road_ms')

    if ml_reached is True:
        reached_badge = (
            '<span class="badge badge-yes">YES</span>'
        )
    elif ml_reached is False:
        reached_badge = (
            '<span class="badge badge-no">NO</span>'
        )
    else:
        reached_badge = '<span class="value">\u2014</span>'

    if ml_vel is not None:
        ml_vel_str = _fmt_float(ml_vel, 1, 'm/s')
    else:
        ml_vel_str = '\u2014'

    # P(reach road) - prefer weighted; show both if both available
    p_w = summary.get('p_reach_road_weighted')
    p_u = summary.get('p_reach_road')
    if p_w is not None:
        p_str = _fmt_float(p_w, 2)
    elif p_u is not None:
        p_str = _fmt_float(p_u, 2)
    else:
        p_str = '\u2014'

    # Scenario list rows
    scenario_rows = []
    for sl in scenario_layers:
        sid = sl['sid']
        color = sl['color']
        checked = 'checked' if sl['shown_by_default'] else ''
        scenario_rows.append(
            f'<label class="scenario-row" data-sid="{sid.lower()}">'
            f'<input type="checkbox" class="scenario-toggle" '
            f'data-layer="{sl["layer_name"]}" {checked}>'
            f'<span class="scenario-swatch" style="background:{color}"></span>'
            f'<span class="scenario-id">{sid}</span>'
            f'</label>'
        )
    scenario_list_html = '\n'.join(scenario_rows)

    summary_json = json.dumps({
        'envelope_threshold': summary['envelope_threshold'],
    })

    # Note on string formatting: chrome HTML is built via .format() rather
    # than f-strings because the CSS contains many curly braces that would
    # collide. Single-brace replacements are escaped as {{ }} and only the
    # explicit {NAME} placeholders below are substituted.
    template = """
<style>
  /* Fonts: title in Trebuchet MS, all other UI text in Calibri.
     Both are system fonts (no web import); fallbacks cover non-Windows. */

  :root {{
    /* ---- Blues theme (sampled from reference palette) ----
       light gray  #E7E6E6   pale blue   #E7EEF2
       steel blue  #35647E   mid blue    #36647F
       navy slate  #2B3A44   dusty blue  #6E93AC
       gray-blue   #7A8893
       Data symbology (road = orange, envelope / impact = crimson) is kept
       high-contrast on purpose so it stays readable over blue terrain; only
       the chrome/UI accents are themed to the blues palette. */
    --bg: #E7EEF2;
    --surface: #FFFFFF;
    --surface-2: #F4F8FB;
    --border: #CBD8E0;
    --border-strong: #7A8893;
    --text: #1E2A33;
    --text-muted: #50697A;
    --text-dim: #8CA2B0;
    --accent: #35647E;        /* chrome accent (hover, checks, carets) */
    --accent-deep: #2B3A44;   /* deep navy for headers / corner marks */
    --road: #E87722;          /* data: road centerline (kept orange) */
    --accent-2: #A6192E;      /* data: envelope + impact markers (crimson) */
    --section: #35647E;       /* section titles */
    --font-title: 'Trebuchet MS', 'Segoe UI', Tahoma, sans-serif;
    --font-body: Calibri, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  }}

  html, body, #map {{
    background: var(--bg) !important;
  }}

  body {{
    font-family: var(--font-body) !important;
    color: var(--text) !important;
  }}

  /* Soften the Leaflet attribution box */
  .leaflet-control-attribution {{
    background: rgba(255,255,255,0.75) !important;
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
    font-size: 10px !important;
    border: 1px solid var(--border) !important;
    padding: 2px 6px !important;
  }}
  .leaflet-control-attribution a {{ color: var(--text-muted) !important; }}

  /* Zoom buttons */
  .leaflet-control-zoom {{
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
  }}
  .leaflet-control-zoom a {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border-bottom: 1px solid var(--border) !important;
    font-family: var(--font-body) !important;
    font-weight: 400 !important;
  }}
  .leaflet-control-zoom a:hover {{
    background: var(--surface-2) !important;
    color: var(--accent) !important;
  }}

  /* Scale bar */
  .leaflet-control-scale-line {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    font-family: var(--font-body) !important;
    font-size: 10px !important;
    letter-spacing: 0.04em !important;
    padding: 2px 6px !important;
  }}

  /* Hide folium's default LayerControl - chrome panel replaces it */
  .leaflet-control-layers {{ display: none !important; }}

  /* Popup */
  .leaflet-popup-content-wrapper {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 2px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18) !important;
  }}
  .leaflet-popup-tip {{
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
  }}
  .leaflet-popup-content {{
    margin: 12px 14px !important;
    font-family: var(--font-body) !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
  }}
  .popup-title {{
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--section);
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }}
  .popup-row {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 3px;
  }}
  .popup-row .k {{ color: var(--text-muted); }}
  .popup-row .v {{ color: var(--text); }}

  /* Tooltips */
  .leaflet-tooltip {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 2px !important;
    font-family: var(--font-body) !important;
    font-size: 10px !important;
    letter-spacing: 0.04em !important;
    padding: 4px 8px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
  }}

  /* ============================================================
     Panel system
     ============================================================ */
  /* ============================================================
     Dock columns - panels stack vertically inside a left and a
     right dock. Absolute-positioned docks keep panels off the map
     surface; flex stacking makes collapse/expand reflow safely.
     ============================================================ */
  .dock {{
    position: absolute;
    z-index: 1000;
    top: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    max-height: calc(100vh - 32px);
    pointer-events: none;   /* let map drag through the gaps */
  }}
  .dock-left {{ left: 16px; }}
  .dock-right {{ right: 16px; align-items: flex-end; }}
  .dock > .panel {{ pointer-events: auto; }}

  .panel {{
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 12px;
    line-height: 1.5;
    box-shadow: 0 6px 18px rgba(43,58,68,0.14);
  }}

  .panel-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
  }}

  .panel-header:hover .panel-caret {{ color: var(--accent); }}

  .panel.collapsed .panel-header {{ border-bottom: none; }}
  .panel.collapsed .panel-body {{ display: none; }}
  .panel.collapsed .panel-caret::before {{ content: '\u25b8'; }}
  .panel-caret::before {{ content: '\u25be'; }}
  .panel-caret {{
    color: var(--text-muted);
    font-size: 11px;
    margin-left: 8px;
  }}

  .panel-title {{
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
  }}

  .panel-meta {{
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.04em;
    margin-left: auto;
    margin-right: 8px;
  }}

  .panel-body {{ padding: 12px 14px 14px; }}

  /* Corner registration marks */
  .corner-mark {{
    position: fixed;
    width: 16px; height: 16px;
    border-color: var(--accent-deep);
    z-index: 1500;
    opacity: 0.45;
    pointer-events: none;
  }}
  .corner-mark.tl {{ top: 6px; left: 6px; border-top: 1px solid; border-left: 1px solid; }}
  .corner-mark.tr {{ top: 6px; right: 6px; border-top: 1px solid; border-right: 1px solid; }}
  .corner-mark.bl {{ bottom: 6px; left: 6px; border-bottom: 1px solid; border-left: 1px solid; }}
  .corner-mark.br {{ bottom: 6px; right: 6px; border-bottom: 1px solid; border-right: 1px solid; }}

  /* ============================================================
     Title block
     ============================================================ */
  .title-block {{
    width: 340px;
  }}
  .title-block .eyebrow {{
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
  }}
  .title-block .project-name {{
    font-family: var(--font-title);
    font-size: 25px;
    font-weight: 700;
    line-height: 1.1;
    margin: 0 0 6px;
    color: var(--accent-deep);
  }}
  .title-block .subtitle {{
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.02em;
  }}
  .title-block .stats {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    margin-top: 14px;
    background: var(--border);
    border: 1px solid var(--border);
  }}
  .title-block .stat {{
    background: var(--surface-2);
    padding: 10px 12px;
  }}
  .title-block .stat-label {{
    font-weight: 700;
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
  }}
  .title-block .stat-value {{
    font-size: 18px;
    color: var(--text);
  }}
  .title-block .stat-value .unit {{
    font-size: 10px;
    color: var(--text-muted);
    margin-left: 3px;
    letter-spacing: 0.04em;
  }}

  /* ============================================================
     Run summary drawer
     ============================================================ */
  .info-drawer {{
    width: 340px;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }}
  .info-drawer .panel-body {{
    overflow-y: auto;
    max-height: 52vh;
    flex: 1;
  }}
  .info-drawer .panel-body::-webkit-scrollbar {{ width: 6px; }}
  .info-drawer .panel-body::-webkit-scrollbar-thumb {{
    background: var(--border-strong);
    border-radius: 3px;
  }}

  .info-section {{ margin-bottom: 14px; }}
  .info-section:last-child {{ margin-bottom: 0; }}
  .info-section-title {{
    font-weight: 700;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--section);
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }}
  .info-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    font-size: 12px;
    border-bottom: 1px solid var(--border);
  }}
  .info-row:last-child {{ border-bottom: none; }}
  .info-row .label {{ color: var(--text-muted); }}
  .info-row .value {{ color: var(--text); }}

  .badge {{
    display: inline-block;
    padding: 1px 8px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    border-radius: 2px;
  }}
  .badge-yes {{
    background: rgba(166,25,46,0.12);
    color: var(--accent-2);
    border: 1px solid var(--accent-2);
  }}
  .badge-no {{
    background: rgba(46,125,91,0.12);
    color: #2E7D5B;
    border: 1px solid #2E7D5B;
  }}

  /* ============================================================
     Layer control panel
     ============================================================ */
  .layer-control {{
    width: 230px;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }}
  .layer-control .panel-body {{
    overflow-y: auto;
    max-height: 52vh;
    flex: 1;
  }}

  .layer-toggle {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    cursor: pointer;
    font-size: 12px;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    transition: color 120ms ease;
  }}
  .layer-toggle:last-of-type {{ border-bottom: none; }}
  .layer-toggle:hover {{ color: var(--accent); }}

  .layer-toggle input[type="checkbox"],
  .scenario-row input[type="checkbox"] {{
    appearance: none;
    -webkit-appearance: none;
    width: 12px; height: 12px;
    border: 1px solid var(--text-dim);
    background: var(--surface);
    cursor: pointer;
    position: relative;
    flex-shrink: 0;
    margin: 0;
  }}
  .layer-toggle input[type="checkbox"]:checked,
  .scenario-row input[type="checkbox"]:checked {{
    background: var(--accent);
    border-color: var(--accent);
  }}
  .layer-toggle input[type="checkbox"]:checked::after,
  .scenario-row input[type="checkbox"]:checked::after {{
    content: '';
    position: absolute;
    inset: 2px;
    background: var(--surface);
  }}

  /* Per-scenario subgroup */
  .scenario-group {{
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }}
  .scenario-group-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    user-select: none;
    margin-bottom: 8px;
  }}
  .scenario-group-title {{
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text);
  }}
  .scenario-group-count {{
    font-size: 10px;
    color: var(--text-muted);
  }}
  .scenario-group.collapsed .scenario-group-body {{ display: none; }}
  .scenario-group.collapsed .scenario-group-caret::before {{ content: '\u25b8'; }}
  .scenario-group-caret::before {{ content: '\u25be'; color: var(--text-muted); font-size: 11px; margin-right: 4px; }}

  .scenario-actions {{
    display: flex;
    gap: 6px;
    margin-bottom: 6px;
  }}
  .scenario-actions button {{
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 10px;
    padding: 4px 0;
    cursor: pointer;
  }}
  .scenario-actions button:hover {{
    background: var(--bg);
    border-color: var(--accent);
    color: var(--accent);
  }}

  .scenario-filter {{
    width: 100%;
    box-sizing: border-box;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 11px;
    padding: 5px 8px;
    margin-bottom: 8px;
  }}
  .scenario-filter::placeholder {{ color: var(--text-dim); }}

  .scenario-list {{
    max-height: 240px;
    overflow-y: auto;
    padding-right: 4px;
  }}
  .scenario-list::-webkit-scrollbar {{ width: 5px; }}
  .scenario-list::-webkit-scrollbar-thumb {{
    background: var(--border-strong);
    border-radius: 2px;
  }}

  .scenario-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
    cursor: pointer;
    font-size: 11px;
  }}
  .scenario-row.hidden {{ display: none; }}

  .scenario-swatch {{
    width: 12px; height: 12px;
    border: 1px solid rgba(0,0,0,0.15);
    flex-shrink: 0;
  }}

  /* ============================================================
     Legend panel
     ============================================================ */
  .legend {{
    width: 320px;
  }}
  .legend-section {{ margin-bottom: 14px; }}
  .legend-section:last-child {{ margin-bottom: 0; }}
  .legend-section-title {{
    font-weight: 700;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--section);
    margin-bottom: 8px;
  }}

  .prob-ramp {{
    height: 12px;
    background: linear-gradient(to right,
      #440154 0%, #3b528b 25%, #21918c 50%, #5ec962 75%, #fde725 100%);
    margin-bottom: 4px;
  }}
  .prob-ticks {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.02em;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 5px;
    font-size: 12px;
    color: var(--text);
  }}
  .legend-item:last-child {{ margin-bottom: 0; }}
  .legend-swatch {{
    flex-shrink: 0;
    width: 24px; height: 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }}

  .legend-pattern-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 16px;
  }}
</style>

<div class="corner-mark tl"></div>
<div class="corner-mark tr"></div>
<div class="corner-mark bl"></div>
<div class="corner-mark br"></div>

<!-- ============= LEFT DOCK: title + run summary ============= -->
<div class="dock dock-left">

<!-- ============= TITLE BLOCK ============= -->
<div class="panel title-block" id="panel-title">
  <div class="panel-header" data-target="panel-title">
    <span class="panel-title">Runout Envelope</span>
    <span class="panel-caret"></span>
  </div>
  <div class="panel-body">
    <div class="project-name">{PROJECT_NAME}</div>
    {SUBTITLE_BLOCK}
    <div class="stats">
      <div class="stat">
        <div class="stat-label">Model Date</div>
        <div class="stat-value">{MODEL_DATE}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Scenarios</div>
        <div class="stat-value">{N_SCENARIOS}</div>
      </div>
    </div>
  </div>
</div>

<!-- ============= RUN SUMMARY ============= -->
<div class="panel info-drawer" id="panel-summary">
  <div class="panel-header" data-target="panel-summary">
    <span class="panel-title">Run Summary</span>
    <span class="panel-meta">{RUN_DATE} \u00b7 v{VERSION}</span>
    <span class="panel-caret"></span>
  </div>
  <div class="panel-body">

    <div class="info-section">
      <div class="info-section-title">Ensemble Metrics</div>
      <div class="info-row"><span class="label">Reach to road</span><span class="value">{N_REACHED} / {N_SCENARIOS}</span></div>
      <div class="info-row"><span class="label">P(reach road)</span><span class="value">{P_REACH}</span></div>
      <div class="info-row"><span class="label">Road length affected</span><span class="value">{ROAD_LEN} m</span></div>
      <div class="info-row"><span class="label">Envelope (P \u2265 {ENV_THRESH})</span><span class="value">{ENV_AREA} m\u00b2</span></div>
      <div class="info-row"><span class="label">Max runout</span><span class="value">{MAX_RUNOUT} m</span></div>
    </div>

    <div class="info-section">
      <div class="info-section-title">Configuration</div>
      <div class="info-row"><span class="label">Friction model</span><span class="value">{FRICTION_MODEL}</span></div>
      <div class="info-row"><span class="label">Mesh resolution</span><span class="value">{MESH_M} m</span></div>
      <div class="info-row"><span class="label">Entrainment</span><span class="value">{ENT_STATE}</span></div>
    </div>

    <div class="info-section">
      <div class="info-section-title">Most Likely Scenario</div>
      <div class="info-row"><span class="label">Scenario ID</span><span class="value">{ML_SID}</span></div>
      <div class="info-row"><span class="label">Probability weight</span><span class="value">{ML_WEIGHT} %</span></div>
      <div class="info-row"><span class="label">Reached road</span>{REACHED_BADGE}</div>
      <div class="info-row"><span class="label">Runout distance</span><span class="value">{ML_RUNOUT} m</span></div>
      <div class="info-row"><span class="label">Velocity at road</span><span class="value">{ML_VEL}</span></div>
    </div>

    <div class="info-section">
      <div class="info-section-title">Ensemble Peaks</div>
      <div class="info-row"><span class="label">Max impact velocity</span><span class="value">{MAX_V} m/s</span></div>
      <div class="info-row"><span class="label">Max flow depth</span><span class="value">{MAX_D} m</span></div>
      <div class="info-row"><span class="label">Max pressure</span><span class="value">{MAX_P} kPa</span></div>
    </div>

  </div>
</div>

</div><!-- /dock-left -->

<!-- ============= RIGHT DOCK: layers + legend ============= -->
<div class="dock dock-right">

<!-- ============= LAYER CONTROL ============= -->
<div class="panel layer-control collapsed" id="panel-layers">
  <div class="panel-header" data-target="panel-layers">
    <span class="panel-title">Layers</span>
    <span class="panel-caret"></span>
  </div>
  <div class="panel-body">
    <label class="layer-toggle"><input type="checkbox" class="ensemble-toggle" data-name="Hillshade" checked> Hillshade</label>
    <label class="layer-toggle"><input type="checkbox" class="ensemble-toggle" data-name="Esri imagery"> Imagery</label>
    <label class="layer-toggle"><input type="checkbox" class="ensemble-toggle" data-name="Probability heatmap" checked> Probability heatmap</label>
    <label class="layer-toggle"><input type="checkbox" class="ensemble-toggle" data-name="Envelope (P >= {ENV_THRESH})" checked> Envelope</label>
    <label class="layer-toggle"><input type="checkbox" class="ensemble-toggle" data-name="Road" checked> Road</label>
    <label class="layer-toggle"><input type="checkbox" class="ensemble-toggle" data-name="Impact points" checked> Impact points</label>
    <label class="layer-toggle"><input type="checkbox" class="ensemble-toggle" data-name="Forest resistance"> Forest resistance</label>

    <div class="scenario-group collapsed" id="scenario-group" style="{SCENARIO_GROUP_STYLE}">
      <div class="scenario-group-header">
        <span><span class="scenario-group-caret"></span><span class="scenario-group-title">Per-scenario</span></span>
        <span class="scenario-group-count">{N_SCENARIOS}</span>
      </div>
      <div class="scenario-group-body">
        <div class="scenario-actions">
          <button id="scenarios-show-all">Show all</button>
          <button id="scenarios-hide-all">Hide all</button>
        </div>
        <input class="scenario-filter" id="scenario-filter" type="text" placeholder="Filter by ID...">
        <div class="scenario-list" id="scenario-list">
          {SCENARIO_LIST}
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ============= LEGEND ============= -->
<div class="panel legend" id="panel-legend">
  <div class="panel-header" data-target="panel-legend">
    <span class="panel-title">Legend</span>
    <span class="panel-meta">P(reach)</span>
    <span class="panel-caret"></span>
  </div>
  <div class="panel-body">

    <div class="legend-section">
      <div class="legend-section-title">Probability</div>
      <div class="prob-ramp"></div>
      <div class="prob-ticks">
        <span>0.05</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span>
      </div>
    </div>

    <div class="legend-section">
      <div class="legend-section-title">Reference</div>
      <div class="legend-item">
        <span class="legend-swatch"><svg width="24" height="4"><line x1="0" y1="2" x2="24" y2="2" stroke="#E87722" stroke-width="3.5"/></svg></span>
        Road centerline
      </div>
      <div class="legend-item">
        <span class="legend-swatch"><svg width="24" height="14"><rect x="1" y="1" width="22" height="12" fill="none" stroke="#A6192E" stroke-width="2" stroke-dasharray="4,2"/></svg></span>
        Envelope (P \u2265 {ENV_THRESH})
      </div>
      <div class="legend-item">
        <span class="legend-swatch"><svg width="24" height="14"><circle cx="12" cy="7" r="4.5" fill="#FFFFFF" stroke="#A6192E" stroke-width="1.8"/><circle cx="12" cy="7" r="1.5" fill="#A6192E"/></svg></span>
        Impact point
      </div>
    </div>

    <div class="legend-section">
      <div class="legend-section-title">Per-scenario pattern</div>
      <div class="legend-pattern-grid">
        <div class="legend-item">
          <span class="legend-swatch"><svg width="24" height="14"><rect x="1" y="1" width="22" height="12" fill="#9CA3A8" fill-opacity="0.35" stroke="#9CA3A8" stroke-width="1.5"/></svg></span>
          Release (filled)
        </div>
        <div class="legend-item">
          <span class="legend-swatch"><svg width="24" height="14"><rect x="1" y="1" width="22" height="12" fill="none" stroke="#9CA3A8" stroke-width="1.5" stroke-dasharray="4,2"/></svg></span>
          Release (original)
        </div>
        <div class="legend-item">
          <span class="legend-swatch"><svg width="24" height="14"><line x1="0" y1="7" x2="24" y2="7" stroke="#9CA3A8" stroke-width="1.5" stroke-dasharray="1,3"/></svg></span>
          Flowpath
        </div>
        <div class="legend-item">
          <span class="legend-swatch"><svg width="24" height="14"><rect x="1" y="1" width="22" height="12" fill="#9CA3A8" fill-opacity="0.15" stroke="#9CA3A8" stroke-width="1"/></svg></span>
          Runout polygon
        </div>
      </div>
    </div>

  </div>
</div>

</div><!-- /dock-right -->

<script>
(function() {{
  var meta = {SUMMARY_JSON};

  // Panel collapse/expand
  document.querySelectorAll('.panel-header').forEach(function(h) {{
    h.addEventListener('click', function(e) {{
      // Don't collapse when clicking interactive children
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
      var id = h.getAttribute('data-target');
      var panel = document.getElementById(id);
      if (panel) panel.classList.toggle('collapsed');
    }});
  }});

  // Scenario sub-group collapse
  var sg = document.getElementById('scenario-group');
  if (sg) {{
    var sgh = sg.querySelector('.scenario-group-header');
    sgh.addEventListener('click', function(e) {{
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
      sg.classList.toggle('collapsed');
    }});
  }}

  // Helper: look up a Leaflet layer by its display name in the registry
  // injected by Python build_viewer at write-time. This is far more
  // reliable than walking map._layers / inspecting options.name (Folium
  // doesn't always populate options.name on the JS layer instance).
  function findLayerByName(name) {{
    if (!window.__layersByName) return null;
    return window.__layersByName[name] || null;
  }}

  // Run cb once the map AND the layer registry are available. Both are
  // set by the init script Python injected near the end of the document.
  function withMap(cb) {{
    if (window.__leafletMap && window.__layersByName) {{
      cb(window.__leafletMap); return;
    }}
    window.addEventListener('__leafletReady', function once() {{
      window.removeEventListener('__leafletReady', once);
      cb(window.__leafletMap);
    }});
    // Also poll, in case the event fired before this listener attached
    var tries = 0;
    var poll = setInterval(function() {{
      tries++;
      if (window.__leafletMap && window.__layersByName) {{
        clearInterval(poll);
        cb(window.__leafletMap);
      }} else if (tries > 200) {{
        clearInterval(poll);
        console.error('Map / layer registry never appeared');
      }}
    }}, 50);
  }}

  withMap(function(map) {{
    // Wire ensemble layer toggles
    document.querySelectorAll('.ensemble-toggle').forEach(function(cb) {{
      var name = cb.getAttribute('data-name');
      cb.addEventListener('change', function() {{
        var layer = findLayerByName(name);
        if (!layer) return;
        if (cb.checked) {{
          if (!map.hasLayer(layer)) map.addLayer(layer);
        }} else {{
          if (map.hasLayer(layer)) map.removeLayer(layer);
        }}
      }});
    }});

    // Contour toggle: all 'Contour P >= *' layers together
    var contourCb = document.querySelector('.contour-toggle');
    if (contourCb) {{
      contourCb.addEventListener('change', function() {{
        var add = contourCb.checked;
        // Use the static array of contour-level names rendered in the page
        var levels = {CONTOUR_NAMES_JSON};
        levels.forEach(function(name) {{
          var layer = findLayerByName(name);
          if (!layer) return;
          if (add) {{
            if (!map.hasLayer(layer)) map.addLayer(layer);
          }} else {{
            if (map.hasLayer(layer)) map.removeLayer(layer);
          }}
        }});
      }});
    }}

    // Per-scenario toggles
    document.querySelectorAll('.scenario-toggle').forEach(function(cb) {{
      var name = cb.getAttribute('data-layer');
      cb.addEventListener('change', function() {{
        var layer = findLayerByName(name);
        if (!layer) return;
        if (cb.checked) {{
          if (!map.hasLayer(layer)) map.addLayer(layer);
        }} else {{
          if (map.hasLayer(layer)) map.removeLayer(layer);
        }}
      }});
    }});

    // Show all / hide all
    var sa = document.getElementById('scenarios-show-all');
    var ha = document.getElementById('scenarios-hide-all');
    function setAll(on) {{
      document.querySelectorAll('.scenario-toggle').forEach(function(cb) {{
        if (cb.checked !== on) {{
          cb.checked = on;
          cb.dispatchEvent(new Event('change'));
        }}
      }});
    }}
    if (sa) sa.addEventListener('click', function() {{ setAll(true); }});
    if (ha) ha.addEventListener('click', function() {{ setAll(false); }});

    // Filter input
    var filt = document.getElementById('scenario-filter');
    if (filt) {{
      filt.addEventListener('input', function() {{
        var q = filt.value.trim().toLowerCase();
        document.querySelectorAll('.scenario-row').forEach(function(r) {{
          var sid = r.getAttribute('data-sid') || '';
          if (!q || sid.indexOf(q) !== -1) {{
            r.classList.remove('hidden');
          }} else {{
            r.classList.add('hidden');
          }}
        }});
      }});
    }}

    // Fit map to envelope bounds if present
    var envName = 'Envelope (P >= ' + meta.envelope_threshold + ')';
    var env = findLayerByName(envName);
    if (env && env.getBounds) {{
      try {{
        var b = env.getBounds();
        if (b.isValid()) map.fitBounds(b, {{ padding: [60, 60] }});
      }} catch (e) {{ /* ignore */ }}
    }}
  }});
}})();
</script>
"""

    # Subtitle conditional block
    if summary.get('project_subtitle'):
        subtitle_block = (
            f'<div class="subtitle">{summary["project_subtitle"]}</div>'
        )
    else:
        subtitle_block = ''

    # Pre-format the numeric values
    p_str_pct = p_str  # already a 2-decimal string or em dash
    contour_names = [
        f"Contour P >= {lvl:.2f}" for lvl in contour_levels
    ]

    return template.format(
        PROJECT_NAME=summary['project_name'],
        SUBTITLE_BLOCK=subtitle_block,
        MODEL_DATE=summary['model_date'],
        N_SCENARIOS=summary['n_scenarios'],
        RUN_DATE=summary['run_date'],
        VERSION=PIPELINE_VERSION,
        ENV_THRESH=f"{summary['envelope_threshold']:.2f}",
        N_REACHED=summary['n_reached'],
        P_REACH=p_str_pct,
        ROAD_LEN=_fmt_int(summary['road_length_affected_m']),
        ENV_AREA=_fmt_int(summary['envelope_area_m2']),
        MAX_RUNOUT=_fmt_int(summary['max_runout_m']),
        FRICTION_MODEL=summary['friction_model'],
        MESH_M=summary['mesh_cell_size_m'],
        ENT_STATE='Enabled' if summary['entrainment_enabled'] else 'Disabled',
        ML_SID=ml_sid,
        ML_WEIGHT=f"{ml_weight_pct:.2f}",
        REACHED_BADGE=reached_badge,
        ML_RUNOUT=_fmt_int(ml_runout),
        ML_VEL=ml_vel_str,
        MAX_V=_fmt_float(summary['max_velocity_ms'], 1),
        MAX_D=_fmt_float(summary['max_flow_depth_m'], 1),
        MAX_P=_fmt_float(summary['max_pressure_kpa'], 0),
        SCENARIO_LIST=scenario_list_html,
        SCENARIO_GROUP_STYLE='' if scenario_layers else 'display:none',
        SUMMARY_JSON=summary_json,
        CONTOUR_NAMES_JSON=json.dumps(contour_names),
    )


def build_viewer(outputs_dir, scenarios, shared_paths, impact_points, config):
    # Build viewer.html with all toggleable layers + custom chrome
    # @param outputs_dir: Path to ensemble outputs directory
    # @param scenarios: list of scenario dicts (each with scenario_id,
    #                    release_geojson, scenario_dir)
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param impact_points: list of impact point dicts
    # @param config: top-level config dict
    # @returns: Path to written viewer.html

    outputs_dir = Path(outputs_dir)
    prob_path = outputs_dir / 'probability.tif'
    envelope_path = outputs_dir / 'envelope.geojson'
    contours_path = outputs_dir / 'contours.geojson'

    # Determine map center from envelope bounds (fall back to road)
    env_wgs = gpd.read_file(envelope_path).to_crs('EPSG:4326')
    if len(env_wgs) > 0:
        bounds = env_wgs.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    else:
        road_wgs = gpd.read_file(shared_paths['road']).to_crs('EPSG:4326')
        b = road_wgs.total_bounds
        center = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]

    # Initialize Leaflet map - no external tiles
    m = folium.Map(
        location=center,
        zoom_start=15,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # ------------------------------------------------------------------
    # Collect every layer we added, keyed by the display name used in the
    # chrome's checkbox data-name attributes. The chrome JS will look up
    # layers in window.__layersByName[displayName] - far more reliable
    # than walking map._layers and inspecting options.name (which Folium
    # doesn't always populate on the JS layer instance).
    # ------------------------------------------------------------------
    layer_registry = {}   # {display_name: folium_layer}

    hillshade_layer, _ = add_hillshade_basemap(m, shared_paths['dem'], outputs_dir)
    layer_registry['Hillshade'] = hillshade_layer

    esri_layer = add_esri_imagery_basemap(m)
    layer_registry['Esri imagery'] = esri_layer

    prob_layer = add_probability_overlay(
        m, prob_path, outputs_dir,
        colormap=config['output']['probability_colormap']
    )
    layer_registry['Probability heatmap'] = prob_layer

    # Contour outlines are not drawn; the envelope carries the outer boundary.
    env_threshold = config['probability']['envelope_min_probability']
    env_layer, _ = add_envelope_layer(m, envelope_path, threshold=env_threshold)
    layer_registry[f"Envelope (P >= {env_threshold})"] = env_layer

    road_layer, _ = add_road_layer(m, shared_paths['road'])
    layer_registry['Road'] = road_layer

    forest_layer = add_forest_layer(m, shared_paths['forest'])
    if forest_layer is not None:
        layer_registry['Forest resistance'] = forest_layer

    source_crs = gpd.read_file(shared_paths['road']).crs
    impact_layer = add_impact_points(m, impact_points, source_crs)
    if impact_layer is not None:
        layer_registry['Impact points'] = impact_layer

    # Per-scenario layers (release, runout polygon) under the accordion.
    # output.viewer_scenario_layers: false drops them for a lighter file.
    scenario_layers = []
    if (config.get('output') or {}).get('viewer_scenario_layers', True):
        scenario_layers = add_per_scenario_layers(m, scenarios, show_default=False)
    for sl in scenario_layers:
        # per-scenario uses the synthetic name '_scenario_<sid>'
        layer_registry[sl['layer_name']] = sl['fg']

    # Identify the highest-probability scenario and turn its layer on by default
    summary = _build_run_summary(scenarios, impact_points, outputs_dir, config)
    ml = summary.get('most_likely') or {}
    ml_sid = (ml.get('sid') or '').lower()
    for sl in scenario_layers:
        if sl['sid'].lower() == ml_sid:
            sl['shown_by_default'] = True

    contour_levels = []

    # Inject chrome (CSS + HTML + JS)
    chrome_html = _build_chrome_html(summary, scenario_layers, contour_levels)
    m.get_root().html.add_child(Element(chrome_html))

    out_path = outputs_dir / 'viewer.html'
    m.save(str(out_path))

    # ------------------------------------------------------------------
    # Inject the layer registry + map handle as window globals.
    #
    # Folium emits each layer as `var <unique_id> = L.<type>(...)` at the
    # top level of the page script. We harvest those JS variable names
    # from the Python layer objects via .get_name(), then emit a tiny
    # script that maps each display name to its JS variable. The chrome
    # JS (findLayerByName) reads from window.__layersByName directly.
    # ------------------------------------------------------------------
    map_var = m.get_name()
    # Build the {display_name: js_var_name} mapping as a JS object literal
    pairs = []
    for display_name, layer in layer_registry.items():
        js_var = layer.get_name()
        # Escape any double-quotes in the display name (envelope name has
        # parentheses but no quotes; safe). Just to be defensive:
        esc = display_name.replace('\\', '\\\\').replace('"', '\\"')
        pairs.append(f'    "{esc}": {js_var}')
    registry_body = ',\n'.join(pairs)

    init_js = (
        "\n<script>\n"
        "  (function() {\n"
        "    function waitForMap() {\n"
        f"      if (typeof {map_var} !== 'undefined') {{\n"
        f"        window.__leafletMap = {map_var};\n"
        "        try {\n"
        "          window.__layersByName = {\n"
        f"{registry_body}\n"
        "          };\n"
        "        } catch (e) {\n"
        "          console.error('Layer registry init failed:', e);\n"
        "          window.__layersByName = {};\n"
        "        }\n"
        "        // Notify any waiting code that the registry is ready\n"
        "        window.dispatchEvent(new Event('__leafletReady'));\n"
        "      } else {\n"
        "        setTimeout(waitForMap, 50);\n"
        "      }\n"
        "    }\n"
        "    waitForMap();\n"
        "  })();\n"
        "</script>\n"
    )
    # Insert AFTER folium's main script block. Folium places its main
    # `var map_xxx = L.map(...); var image_overlay_xxx = ...` block *after*
    # </body> as a separate <script> after the closing body tag. The chrome
    # script (inside <body>) runs first; if we put the registry init in the
    # chrome script or right before </body>, the layer variables don't exist
    # yet and the registry comes back empty.
    #
    # Strategy: find the LAST </script> in the document and insert init_js
    # right after it. That puts the registry init after every layer var
    # declaration, so all the var references resolve correctly.
    html = out_path.read_text()
    last_script_close = html.rfind('</script>')
    if last_script_close >= 0:
        insert_at = last_script_close + len('</script>')
        html = html[:insert_at] + init_js + html[insert_at:]
    elif '</body>' in html:
        # Fallback: insert before </body>
        idx = html.rfind('</body>')
        html = html[:idx] + init_js + html[idx:]
    else:
        html = html + init_js
    out_path.write_text(html)

    logging.info(f"Wrote viewer: {out_path}")
    return out_path