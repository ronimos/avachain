"""
build_report.py

Author: Valerie Foley
Last Updated: 7/2026

Description:
    Renders a static HTML report from the ensemble outputs:
        - Headline: N of M scenarios reached the road, with Wilson 95% CI
        - Run configuration table (model, thresholds, weighting)
        - Probability map (hillshade basemap, road, contours, impact points)
        - Chainage histogram of road impact points
        - Runout distance histogram (reached vs missed)
        - Parameter sensitivity scatter (density, release volume)
        - Per-scenario table sorted by runout distance

    Plots are embedded as base64 PNGs so the report is one self-contained
    file. Colors and fonts match the Leaflet viewer.

    No command line. Imported by avalanche_runout.py.

Requirements:
    - matplotlib, jinja2, geopandas, rasterio, pandas
"""

# --------- Load Libraries --------
import base64
import logging
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
import geopandas as gpd
from jinja2 import Template

import avalanche_helpers as h


# Palette shared with build_viewer so the report and viewer look alike
COLORS = {
    'road': '#E87722',        # road centerline (orange)
    'accent2': '#A6192E',     # envelope, impact points, reached (crimson)
    'slate': '#5B7A9E',       # did-not-reach
    'accent': '#35647E',      # chrome accent
    'text': '#1E2A33',
}


def fig_to_base64(fig, dpi=120):
    # Encode a matplotlib figure as base64 PNG for inline HTML embedding
    # @param fig: matplotlib Figure
    # @param dpi: int - rendering DPI
    # @returns: str - base64-encoded PNG

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _raster_extent(meta):
    # Compute the [left, right, bottom, top] extent for imshow
    # @param meta: meta dict from avalanche_helpers.read_raster
    # @returns: list [left, right, bottom, top]

    t = meta['transform']
    left, top = t.c, t.f
    right = left + meta['shape'][1] * t.a
    bottom = top + meta['shape'][0] * t.e
    return [left, right, bottom, top]


def _viridis_hex(t):
    # Map t in [0, 1] to a viridis hex color string
    # @param t: float in [0, 1]
    # @returns: str - hex color

    cmap = plt.get_cmap('viridis')
    r, g, b, _ = cmap(float(np.clip(t, 0.0, 1.0)))
    return '#{:02x}{:02x}{:02x}'.format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def plot_probability_map(probability_path, dem_path, road_path,
                         envelope_path, contours_path, impact_points,
                         dpi=120):
    # Probability raster over a hillshade basemap, matching the viewer.
    # Cells with P = 0 or nodata are left transparent.
    # @param probability_path: Path to probability raster
    # @param dem_path: Path to the DEM used for the run (hillshade source)
    # @param road_path: Path to road shapefile/geojson
    # @param envelope_path: Path to envelope geojson
    # @param contours_path: Path to contours geojson
    # @param impact_points: list of dicts with 'xy' field
    # @param dpi: int - rendering DPI
    # @returns: str - base64 PNG

    prob, prob_meta = h.read_raster(probability_path)
    prob_extent = _raster_extent(prob_meta)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Hillshade basemap from the DEM (same grayscale style as the viewer)
    try:
        dem, dem_meta = h.read_raster(dem_path)
        nodata = dem_meta['nodata']
        bad = np.zeros(dem.shape, dtype=bool)
        if nodata is not None:
            bad = (dem == nodata)
        bad |= ~np.isfinite(dem)
        if bad.any():
            fill = dem[~bad].mean() if (~bad).any() else 0.0
            dem = np.where(bad, fill, dem)
        cell_m = abs(dem_meta['transform'].a)
        ls = LightSource(azdeg=315, altdeg=45)
        shade = ls.hillshade(dem * 1.3, dx=cell_m, dy=cell_m)
        shade = np.ma.masked_where(bad, shade)
        ax.imshow(shade, extent=_raster_extent(dem_meta), origin='upper',
                  cmap='gray', vmin=0, vmax=1, zorder=1)
    except Exception as e:
        logging.warning(f"Report map hillshade skipped: {e}")

    # Probability overlay. Mask nodata AND zero so untouched terrain shows
    # the hillshade only.
    masked = np.ma.masked_where(~(prob > 0), prob)
    im = ax.imshow(masked, extent=prob_extent, origin='upper',
                   cmap='viridis', vmin=0, vmax=1, alpha=0.75, zorder=2)
    plt.colorbar(im, ax=ax, label='P(reach)', shrink=0.8)

    # Envelope outline - crimson dashed, like the viewer
    try:
        env_gdf = gpd.read_file(envelope_path)
        env_gdf.boundary.plot(ax=ax, color=COLORS['accent2'], linewidth=1.6,
                              linestyle=(0, (6, 3)), zorder=3,
                              label='Envelope')
    except Exception as e:
        logging.debug(f"No envelope drawn: {e}")

    # Contour outlines - viridis-colored by level, like the viewer
    try:
        contours = gpd.read_file(contours_path)
        for level in sorted(contours['level'].unique()):
            subset = contours[contours['level'] == level]
            subset.boundary.plot(
                ax=ax, linewidth=0.8 + level, zorder=4,
                edgecolor=_viridis_hex(float(level)),
                alpha=0.7 + level * 0.25
            )
    except Exception as e:
        logging.debug(f"No contours drawn: {e}")

    # Road centerline - orange, like the viewer
    road_gdf = gpd.read_file(road_path)
    road_gdf.plot(ax=ax, color=COLORS['road'], linewidth=2.5, zorder=5,
                  label='Road')

    # Impact points - solid crimson dots, like the viewer
    if impact_points:
        xs = [p['xy'][0] for p in impact_points if p['xy']]
        ys = [p['xy'][1] for p in impact_points if p['xy']]
        ax.scatter(xs, ys, c=COLORS['accent2'], s=45, zorder=6,
                   label=f"Impact points (n={len(xs)})")

    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=8)

    return fig_to_base64(fig, dpi)


def plot_chainage_histogram(impact_points, dpi=120):
    # Histogram of road chainage at impact points
    # @param impact_points: list of dicts with 'chainage_m' field
    # @param dpi: int - rendering DPI
    # @returns: str (base64 PNG) or None if no impact points

    chainages = [p['chainage_m'] for p in impact_points
                 if p['chainage_m'] is not None]

    if not chainages:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(chainages, bins=20, color=COLORS['accent'], edgecolor='white')
    ax.set_xlabel('Chainage along road (m)')
    ax.set_ylabel('Number of scenarios reaching')
    return fig_to_base64(fig, dpi)


def plot_runout_histogram(df, dpi=120):
    # Stacked histogram of max runout, split by reached/missed
    # @param df: DataFrame from scenarios_summary.csv
    # @param dpi: int - rendering DPI
    # @returns: str - base64 PNG

    fig, ax = plt.subplots(figsize=(8, 4))

    reached = df[df['reached_road'] == True]
    missed = df[df['reached_road'] == False]

    ax.hist(
        [reached['runout_max_m'], missed['runout_max_m']],
        bins=15, stacked=True,
        label=['Reached road', 'Did not reach'],
        color=[COLORS['accent2'], COLORS['slate']],
        edgecolor='white'
    )
    ax.set_xlabel('Max runout distance (m)')
    ax.set_ylabel('Number of scenarios')
    ax.legend()

    return fig_to_base64(fig, dpi)


def plot_parameter_sensitivity(df, dpi=120):
    # Scatter of runout vs the inputs that vary across the ensemble
    # (release density and release volume). Friction parameters are fixed
    # per run so they are not plotted.
    # @param df: DataFrame from scenarios_summary.csv
    # @param dpi: int - rendering DPI
    # @returns: str (base64 PNG) or None if no plottable parameters

    params = [
        ('rho_kgm3', 'Release density (kg/m3)'),
        ('release_volume_m3', 'Release volume (m3)'),
    ]
    usable = [(k, lbl) for k, lbl in params
              if k in df.columns and not df[k].isna().all()]
    if not usable:
        return None

    fig, axes = plt.subplots(1, len(usable), figsize=(6 * len(usable), 4),
                             squeeze=False)

    for ax, (key, label) in zip(axes[0], usable):
        colors = [COLORS['accent2'] if r else COLORS['slate']
                  for r in df['reached_road']]
        ax.scatter(df[key], df['runout_max_m'], c=colors, s=55,
                   edgecolor='white', linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel('Runout (m)')

    legend_handles = [
        Patch(facecolor=COLORS['accent2'], label='Reached road'),
        Patch(facecolor=COLORS['slate'], label='Did not reach')
    ]
    fig.legend(handles=legend_handles, loc='upper right')

    return fig_to_base64(fig, dpi)


# Single Jinja2 template for the whole report. Styling matches the viewer
# theme (light blue-gray chrome, Calibri body, Trebuchet titles).
REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ project_name }} - Runout Report</title>
<style>
  :root {
    --bg: #E7EEF2;
    --surface: #FFFFFF;
    --surface-2: #F4F8FB;
    --border: #CBD8E0;
    --border-strong: #7A8893;
    --text: #1E2A33;
    --text-muted: #50697A;
    --accent: #35647E;
    --accent-deep: #2B3A44;
    --road: #E87722;
    --accent-2: #A6192E;
    --font-title: 'Trebuchet MS', 'Segoe UI', Tahoma, sans-serif;
    --font-body: Calibri, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  }
  body {
    font-family: var(--font-body);
    max-width: 1100px;
    margin: 2em auto;
    padding: 0 1.5em 3em;
    color: var(--text);
    line-height: 1.55;
    background: var(--bg);
  }
  h1 {
    font-family: var(--font-title);
    color: var(--accent-deep);
    border-bottom: 3px solid var(--accent);
    padding-bottom: 0.3em;
    margin-bottom: 0.1em;
    font-weight: 600;
  }
  h2 {
    font-family: var(--font-title);
    margin-top: 2em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.25em;
    font-weight: 600;
    color: var(--accent);
  }
  .subtitle {
    color: var(--text-muted);
    font-size: 1.05em;
    margin: 0 0 0.2em;
  }
  .meta {
    color: var(--text-muted);
    font-size: 0.85em;
    margin-bottom: 2em;
  }
  .headline {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-2);
    padding: 1.2em 1.5em;
    font-size: 1.05em;
    margin: 1.5em 0;
    border-radius: 0 4px 4px 0;
  }
  .headline .big {
    font-size: 2.2em;
    font-weight: 700;
    color: var(--accent-2);
    line-height: 1;
  }
  .headline p { margin: 0.5em 0; }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.88em;
    background: var(--surface);
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  th, td {
    border: 1px solid var(--border);
    padding: 0.5em 0.7em;
    text-align: right;
  }
  th {
    background: var(--accent-deep);
    color: white;
    font-weight: 500;
    border-color: var(--border-strong);
  }
  td.id, th.id {
    text-align: left;
    font-family: 'SF Mono', Monaco, Menlo, monospace;
    font-size: 0.92em;
  }
  table.config { width: auto; min-width: 50%; }
  table.config td { text-align: left; }
  table.config td.k { color: var(--text-muted); }
  .reached-yes { background: #FBF0F1; }
  .reached-yes td.id { color: var(--accent-2); font-weight: 600; }
  tr:hover { background: var(--surface-2); }
  img {
    max-width: 100%;
    height: auto;
    margin: 1em 0;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--surface);
  }
  code {
    background: var(--surface-2);
    border: 1px solid var(--border);
    padding: 0.05em 0.4em;
    border-radius: 3px;
    font-size: 0.92em;
    font-family: 'SF Mono', Monaco, Menlo, monospace;
  }
</style>
</head>
<body>
<h1>{{ project_name }}</h1>
{% if project_subtitle %}<p class="subtitle">{{ project_subtitle }}</p>{% endif %}
<p class="meta">Runout report &middot; generated {{ timestamp }} &middot;
   pipeline v{{ version }} &middot;
   {{ n_scenarios }} scenarios &middot;
   {{ friction_model }}</p>

<div class="headline">
  <p><span class="big">{{ n_reached }} / {{ n_scenarios }}</span>
     scenarios reach the road</p>
  <p>Fraction reached (equal-weight): <b>{{ "%.1f"|format(fraction * 100) }}%</b><br>
     Wilson 95% CI: [<b>{{ "%.1f"|format(ci_low * 100) }}%</b>,
                     <b>{{ "%.1f"|format(ci_high * 100) }}%</b>]</p>
  {% if fraction_weighted is not none %}
  <p>Fraction reached (likelihood-weighted):
     <b>{{ "%.1f"|format(fraction_weighted * 100) }}%</b></p>
  {% endif %}
  {% if nearest_miss %}
  <p>Nearest miss: <code>{{ nearest_miss.scenario }}</code> stopped
     <b>{{ "%.1f"|format(nearest_miss.distance_short_m) }} m</b>
     short of the road.</p>
  {% endif %}
</div>

<p>This report summarizes an ensemble of {{ n_scenarios }} release scenarios
run through AvaFrame com1DFA on the project DEM. The map shows, for every
cell, the {% if weighting_on %}likelihood-weighted{% endif %} fraction of
scenarios whose flow reached that cell. This is a probability conditional on
the scenario set, not an annual or return-period probability.</p>

<h2>Run configuration</h2>
<table class="config">
  <tr><td class="k">Friction model</td><td>{{ friction_model }}</td></tr>
  {% if tau0 is not none %}
  <tr><td class="k">Minimum shear stress (tau0)</td><td>{{ tau0 }} Pa</td></tr>
  {% endif %}
  <tr><td class="k">Simulation type</td><td>{{ sim_type }}</td></tr>
  <tr><td class="k">Mesh cell size</td><td>{{ cell_size }} m</td></tr>
  <tr><td class="k">Reach threshold (flow depth)</td><td>{{ reach_threshold }} m</td></tr>
  {% if pressure_threshold %}
  <tr><td class="k">Reach threshold (pressure)</td><td>{{ pressure_threshold }} kPa</td></tr>
  {% endif %}
  <tr><td class="k">Scenario weighting</td>
      <td>{{ "by scenario probability" if weighting_on else "equal" }}</td></tr>
  <tr><td class="k">Road buffer</td><td>{{ road_buffer }} m</td></tr>
</table>

<h2>Probability map</h2>
<img src="data:image/png;base64,{{ map_b64 }}" alt="Probability map">

<h2>Road impact distribution</h2>
{% if chainage_b64 %}
<img src="data:image/png;base64,{{ chainage_b64 }}" alt="Chainage histogram">
<p class="meta">Chainage origin: far end of road polyline from release centroid.</p>
{% else %}
<p><em>No scenarios reached the road.</em></p>
{% endif %}

<h2>Runout distance distribution</h2>
<img src="data:image/png;base64,{{ runout_b64 }}" alt="Runout histogram">

{% if sens_b64 %}
<h2>Parameter sensitivity</h2>
<p class="meta">Runout against the inputs that vary across the ensemble.
Friction parameters are fixed per run and are not shown.</p>
<img src="data:image/png;base64,{{ sens_b64 }}" alt="Parameter sensitivity">
{% endif %}

<h2>Per-scenario results</h2>
<table>
<thead>
<tr>
  <th class="id">Scenario</th>
  <th>P(scenario)</th>
  <th>Density (kg/m3)</th><th>Vol (m3)</th>
  <th>Runout (m)</th>
  <th>Peak v (m/s)</th>
  <th>Peak depth (m)</th>
  <th>Reached</th>
  <th>Dist to road (m)</th>
  <th>Depth at road (m)</th>
  <th>v at road (m/s)</th>
</tr>
</thead>
<tbody>
{% for row in scenarios %}
<tr class="{{ 'reached-yes' if row.reached_road else 'reached-no' }}">
  <td class="id">{{ row.scenario_id }}</td>
  <td>{{ row.scenario_probability }}</td>
  <td>{{ row.rho_kgm3 }}</td>
  <td>{{ row.release_volume_m3 or "-" }}</td>
  <td>{{ row.runout_max_m }}</td>
  <td>{{ row.peak_velocity_ms }}</td>
  <td>{{ row.peak_flow_depth_m }}</td>
  <td>{{ "yes" if row.reached_road else "-" }}</td>
  <td>{{ row.distance_to_road_m }}</td>
  <td>{{ row.depth_at_road_m or "-" }}</td>
  <td>{{ row.velocity_at_road_ms or "-" }}</td>
</tr>
{% endfor %}
</tbody>
</table>

</body>
</html>
"""


def build_report(outputs_dir, scenario_summaries, road_impact, shared_paths,
                 config, wrapper_version):
    # Render the full HTML report and save to outputs_dir/report.html
    # @param outputs_dir: Path to ensemble outputs directory
    # @param scenario_summaries: list of summary dicts (one per scenario)
    # @param road_impact: dict from build_road_impact_summary
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param config: top-level config dict
    # @param wrapper_version: str - version string for the report header
    # @returns: Path to written HTML file

    outputs_dir = Path(outputs_dir)

    # Read the scenarios CSV written by the stack module
    csv_path = outputs_dir / 'scenarios_summary.csv'
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    dpi = config['output']['report_plots_dpi']

    # Build all plots
    map_b64 = plot_probability_map(
        probability_path=outputs_dir / 'probability.tif',
        dem_path=shared_paths['dem'],
        road_path=shared_paths['road'],
        envelope_path=outputs_dir / 'envelope.geojson',
        contours_path=outputs_dir / 'contours.geojson',
        impact_points=road_impact['impact_points'],
        dpi=dpi
    )
    chainage_b64 = plot_chainage_histogram(
        road_impact['impact_points'], dpi=dpi
    )
    runout_b64 = plot_runout_histogram(df, dpi=dpi)
    sens_b64 = plot_parameter_sensitivity(df, dpi=dpi)

    display_cfg = config.get('display', {}) or {}
    friction_cfg = config.get('friction', {}) or {}
    reach_cfg = config.get('reach', {}) or {}

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        project_name=display_cfg.get('project_name', 'Avalanche Runout'),
        project_subtitle=display_cfg.get('subtitle', ''),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=wrapper_version,
        n_scenarios=road_impact['n_scenarios'],
        n_reached=road_impact['n_reached'],
        fraction=road_impact['fraction_reached'],
        fraction_weighted=road_impact.get('fraction_reached_weighted'),
        ci_low=road_impact['wilson_95_ci'][0],
        ci_high=road_impact['wilson_95_ci'][1],
        nearest_miss=road_impact['nearest_miss'],
        scenarios=df.to_dict(orient='records'),
        map_b64=map_b64,
        chainage_b64=chainage_b64,
        runout_b64=runout_b64,
        sens_b64=sens_b64,
        friction_model=config['com1dfa']['friction_model'],
        tau0=friction_cfg.get('tau0'),
        sim_type=config['com1dfa']['sim_type'],
        cell_size=config['com1dfa']['mesh_cell_size_m'],
        reach_threshold=reach_cfg.get('flow_depth_threshold_m'),
        pressure_threshold=reach_cfg.get('pressure_threshold_kpa') or None,
        weighting_on=bool(config.get('probability', {}).get(
            'weight_by_scenario_probability', False)),
        road_buffer=config['road']['buffer_m'],
    )

    out_path = outputs_dir / 'report.html'
    out_path.write_text(html)
    logging.info(f"Wrote report: {out_path}")

    return out_path
