"""
compare_reinit_runs.py — Compare no-reinit vs avalanche-aware SNOWPACK runs.

Reads two Zarr stores (built from separate SNOWPACK runs), spatially interpolates
per-cluster values onto the DEM grid, and generates:
  - Side-by-side + difference maps for each field at each post-event survey date
  - Time series of release-zone mean values (HS, slab thickness, min Sk38)
  - Summary CSV with daily release-zone stats for both runs

Fields compared:
  HS            — total snow depth (m), direct from Zarr
  slab_thickness — depth from surface to WL interface (m), derived per cluster
  min_sk38      — minimum Sk38 near the WL interface, derived per cluster

Usage:
  python compare_reinit_runs.py \
      --zarr-no-reinit   path/no_reinit/slope_snowpack.zarr \
      --zarr-with-reinit path/with_reinit/slope_snowpack.zarr \
      --smet-dir         outputs/smet \
      --dem-tif          outputs/resampled_1m/dem_1m.tif \
      --release-geojson  data/boundaries/avalanche_release_area.geojson \
      --event-date       2026-01-18 \
      --out-dir          outputs/plots/comparison_v2
"""

import argparse
import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import zarr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
from scipy.interpolate import griddata

matplotlib.use("Agg")

# ── Presentation style ────────────────────────────────────────────────────────
_PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,   # embed fonts for Illustrator / PowerPoint
    "ps.fonttype":  42,
}

# Two-run color identity — matches dataviz categorical slots 1 & 2
_COLOR_NO_REINIT   = "#2a78d6"   # blue
_COLOR_WITH_REINIT = "#c0392b"   # deep red

# Zone boundary styles
_RELEASE_LW, _RELEASE_COLOR = 1.8, "#f39c12"   # amber
_CROWN_LW,   _CROWN_COLOR   = 1.3, "#ecf0f1"   # off-white dashed

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FIELDS = {
    "HS": {
        "label": "Snow depth (m)",
        "cmap": "Blues",
        "vmin": 0.0,
        "vmax": 2.5,
        "diff_cmap": "RdBu",
        "diff_vmax": 0.5,
    },
    "slab_thickness": {
        "label": "Slab thickness (m)",
        "cmap": "YlOrBr",
        "vmin": 0.0,
        "vmax": 1.2,
        "diff_cmap": "RdBu",
        "diff_vmax": 0.4,
    },
    "min_sk38": {
        "label": "Min Sk38",
        "cmap": "RdYlGn",
        "vmin": 0.0,
        "vmax": 3.0,
        "diff_cmap": "RdBu",
        "diff_vmax": 0.5,
    },
    "tau_g": {
        "label": "τ_g driving shear (Pa)",
        "cmap": "YlOrRd",
        "vmin": 0.0,
        "vmax": 1500.0,
        "diff_cmap": "RdBu",
        "diff_vmax": 200.0,
    },
    "Lambda": {
        "label": "Λ elastic length (m)",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 3.0,
        "diff_cmap": "RdBu",
        "diff_vmax": 0.5,
    },
    "sigma_t": {
        "label": "σ_t tensile strength (kPa)",
        "cmap": "plasma",
        "vmin": 0.0,
        "vmax": 10.0,
        "diff_cmap": "RdBu",
        "diff_vmax": 2.0,
    },
}

# Meloche-framework physical constants (Gaume et al. 2018; Reiweger et al. 2010)
_G_WL  = 0.2e6   # WL shear modulus (Pa)
_NU    = 0.3      # Poisson's ratio
_PHI   = 27.0     # friction angle (deg)
_G_GRAV = 9.81    # m/s²

EPOCH_STR = "hours since "


# ──────────────────────────────────────────────────────────────────────────────
# Zarr helpers
# ──────────────────────────────────────────────────────────────────────────────

def open_zarr(path: Path) -> zarr.Group:
    return zarr.open(str(path), mode="r")


def zarr_times_as_datetimes(z: zarr.Group) -> list[datetime]:
    t_arr = z["time"]
    units = t_arr.attrs.get("units", "")
    if units.startswith(EPOCH_STR):
        epoch = datetime.fromisoformat(units[len(EPOCH_STR):])
    else:
        raise ValueError(f"Unrecognised time units: {units}")
    return [epoch + timedelta(hours=int(h)) for h in t_arr[:]]


def find_time_index(times: list[datetime], target: datetime, tol_h: int = 12) -> int | None:
    for i, t in enumerate(times):
        if abs((t - target).total_seconds()) <= tol_h * 3600:
            return i
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-cluster feature extraction at a single time index
# ──────────────────────────────────────────────────────────────────────────────

def extract_cluster_features(
    z: zarr.Group,
    t_idx: int,
    slope_angles: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by cluster name with columns:
      HS, slab_thickness, min_sk38, tau_g, Lambda, sigma_t

    WL detection: layer with minimum sk38 restricted to lower 60% of pack.
    Meloche-framework fields (tau_g, Lambda, sigma_t) require slab density
    and slope angle; set to NaN when inputs are missing.
    """
    locations = z["location"][:]

    hs_arr      = z["HS"][:, t_idx]
    sk38_arr    = z["sk38"][:, t_idx, :]
    height_arr  = z["height"][:, t_idx, :]
    density_arr = z["density"][:, t_idx, :]

    phi_rad = np.radians(_PHI)
    tan_phi = np.tan(phi_rad)

    nan_row = {k: np.nan for k in ("HS", "slab_thickness", "min_sk38",
                                    "tau_g", "Lambda", "sigma_t")}
    rows = []
    for i, loc in enumerate(locations):
        hs = float(hs_arr[i])
        sk38 = sk38_arr[i]
        hgt  = height_arr[i]
        dens = density_arr[i]

        valid = np.isfinite(sk38) & np.isfinite(hgt) & (hgt != 0.0)
        if valid.sum() < 2 or not np.isfinite(hs) or hs <= 0:
            rows.append(dict(nan_row))
            continue

        sk_v   = sk38[valid]
        hgt_v  = hgt[valid]
        dens_v = dens[valid]

        # WL interface: min sk38 restricted to lower 60% of pack
        slab_zone = (hgt_v > 0) & (hgt_v < hs * 0.6) & (sk_v < 2.0)
        if slab_zone.sum() < 1:
            wl_idx = int(np.argmin(sk_v))
        else:
            slab_sk = np.where(slab_zone, sk_v, np.inf)
            wl_idx = int(np.argmin(slab_sk))

        interface_ht = float(hgt_v[wl_idx])   # m from snowpack base
        slab_thick   = hs - interface_ht if interface_ht > 0 else np.nan

        lo = max(0, wl_idx - 2)
        hi = min(len(sk_v), wl_idx + 3)
        min_sk38 = float(np.nanmin(sk_v[lo:hi]))

        row: dict = {
            "HS": hs,
            "slab_thickness": max(slab_thick, 0.0) if np.isfinite(slab_thick) else np.nan,
            "min_sk38": min_sk38,
            "tau_g": np.nan,
            "Lambda": np.nan,
            "sigma_t": np.nan,
        }

        if not (np.isfinite(slab_thick) and slab_thick > 0.01):
            rows.append(row)
            continue

        # Slab mean density (layers above WL interface)
        slab_mask = np.isfinite(dens_v) & (hgt_v > interface_ht)
        rho = float(np.nanmean(dens_v[slab_mask])) if slab_mask.sum() >= 1 else np.nan

        if not np.isfinite(rho) or rho <= 0:
            rows.append(row)
            continue

        # WL thickness: element spacing around the WL layer
        n_v = len(hgt_v)
        if wl_idx > 0 and wl_idx < n_v - 1:
            D_wl = abs(float(hgt_v[wl_idx + 1] - hgt_v[wl_idx - 1])) / 2.0
        elif wl_idx > 0:
            D_wl = abs(float(hgt_v[wl_idx] - hgt_v[wl_idx - 1]))
        elif n_v > 1:
            D_wl = abs(float(hgt_v[1] - hgt_v[0]))
        else:
            D_wl = 0.02
        D_wl = max(D_wl, 0.005)   # floor at 5 mm

        h_m     = float(slab_thick)
        E_slab  = (rho / 300.0) ** 2.5 * 4.0e6      # Pa — van Herwijnen 2016
        sigma_t = (rho / 300.0) ** 1.4 * 5.0e3      # Pa — Meloche 2-10 kPa range
        E_prime = E_slab / (1.0 - _NU ** 2)
        K_wl    = _G_WL / D_wl
        Lambda  = float(np.sqrt(E_prime * h_m / K_wl))

        row["sigma_t"] = sigma_t / 1000.0   # Pa → kPa for display
        row["Lambda"]  = Lambda

        # tau_g needs slope angle per cluster
        loc_str = str(loc)
        psi_deg = (slope_angles or {}).get(loc_str, np.nan)
        if np.isfinite(psi_deg) and psi_deg > 1.0:
            psi_rad = np.radians(psi_deg)
            sin_psi = np.sin(psi_rad)
            tan_psi = np.tan(psi_rad)
            factor  = max(0.0, 1.0 - tan_phi / tan_psi)
            row["tau_g"] = rho * _G_GRAV * h_m * sin_psi * factor

        rows.append(row)

    return pd.DataFrame(rows, index=locations)


# ──────────────────────────────────────────────────────────────────────────────
# Spatial interpolation
# ──────────────────────────────────────────────────────────────────────────────

def load_cluster_coords_utm(
    smet_dir: Path, dem_crs_wkt: str
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Read lat/lon and slope_angle from each cluster SMET header.

    Returns (coords, slopes):
      coords  — cluster_id → (easting, northing) in DEM CRS
      slopes  — cluster_id → slope_angle in degrees
    """
    tr = Transformer.from_crs("EPSG:4326", dem_crs_wkt, always_xy=True)
    coords: dict[str, tuple[float, float]] = {}
    slopes: dict[str, float] = {}
    for smet in sorted(smet_dir.glob("cluster_*.smet")):
        cid = smet.stem
        lat = lon = slope = None
        with open(smet) as fh:
            for line in fh:
                if line.startswith("latitude"):
                    lat = float(line.split("=")[1])
                elif line.startswith("longitude"):
                    lon = float(line.split("=")[1])
                elif line.startswith("slope_angle"):
                    slope = float(line.split("=")[1])
                elif line.startswith("[DATA]"):
                    break
        if lat is not None and lon is not None:
            x, y = tr.transform(lon, lat)
            coords[cid] = (x, y)
        if slope is not None:
            slopes[cid] = slope
    return coords, slopes


def interpolate_to_grid(
    values: np.ndarray,
    xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate scattered (n_pts, 2) xy and values onto grid_x/grid_y meshgrid."""
    pts = np.column_stack([xy[:, 0], xy[:, 1]])
    grid = griddata(pts, values, (grid_x, grid_y), method=method)
    return grid.astype(np.float32)


def build_release_mask(geojson_path: Path, dem_transform, dem_shape) -> np.ndarray:
    """Rasterize a GeoJSON polygon onto the DEM grid. Returns bool array."""
    if not geojson_path.exists():
        return np.ones(dem_shape, dtype=bool)
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs("EPSG:6342")
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    mask = rasterio.features.rasterize(
        shapes, out_shape=dem_shape, transform=dem_transform,
        fill=0, dtype="uint8",
    )
    return mask.astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison_date(
    no_reinit_grids: dict[str, np.ndarray],
    with_reinit_grids: dict[str, np.ndarray],
    hillshade: np.ndarray,
    release_mask: np.ndarray,
    crown_mask: np.ndarray | None,
    date_str: str,
    out_path: Path,
):
    fields = list(FIELDS.keys())
    n_fields = len(fields)

    with matplotlib.rc_context(_PLOT_RC):
        fig, axes = plt.subplots(
            n_fields, 3,
            figsize=(17, 5.2 * n_fields),
            gridspec_kw={"hspace": 0.38, "wspace": 0.05},
        )
        if n_fields == 1:
            axes = axes[np.newaxis, :]

        fig.text(0.5, 1.002, date_str,
                 ha="center", va="bottom",
                 fontsize=15, fontweight="bold", color="#1a1a1a")

        for col, (header, color) in enumerate(zip(
            ["No Reinit", "With Reinit", "Δ (Reinit − No Reinit)"],
            [_COLOR_NO_REINIT, _COLOR_WITH_REINIT, "#555555"],
        )):
            axes[0, col].set_title(header, fontsize=11, fontweight="semibold",
                                   color=color, pad=8)

        def _show(ax, data, cmap_name, vmin, vmax, label):
            hs = (hillshade if hillshade is not None
                  else np.zeros(data.shape[:2], dtype=np.uint8))
            ax.imshow(hs, cmap="gray", vmin=0, vmax=255, alpha=0.45,
                      interpolation="bilinear")
            cm = copy.copy(plt.get_cmap(cmap_name))
            cm.set_bad("#d0d0d0")   # no-data areas → light gray
            im = ax.imshow(np.ma.masked_invalid(data), cmap=cm,
                           vmin=vmin, vmax=vmax, alpha=0.85,
                           interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_color("#b8b8b8")
            # Horizontal colorbar below the panel via inset_axes
            cax = inset_axes(ax, width="88%", height="5%",
                             loc="lower center",
                             bbox_to_anchor=(0, -0.14, 1, 1),
                             bbox_transform=ax.transAxes,
                             borderpad=0)
            cb = plt.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_label(label, fontsize=8.5, labelpad=3, color="#444444")
            cb.ax.tick_params(labelsize=8, colors="#555555", length=2, width=0.5)
            cb.outline.set_linewidth(0.4)
            cb.locator = mticker.MaxNLocator(nbins=5, prune="both")
            cb.update_ticks()

        def _contours(ax, mask, color, lw, ls="solid"):
            if mask is not None and mask.any():
                ax.contour(mask.astype(float), levels=[0.5],
                           colors=[color], linewidths=lw, linestyles=ls)

        for row, fname in enumerate(fields):
            cfg = FIELDS[fname]
            nr = no_reinit_grids.get(fname)
            wr = with_reinit_grids.get(fname)

            axes[row, 0].set_ylabel(cfg["label"], fontsize=9.5,
                                    labelpad=5, color="#333333")

            if nr is None or wr is None:
                for col in range(3):
                    axes[row, col].set_visible(False)
                continue

            _show(axes[row, 0], nr,     cfg["cmap"],      cfg["vmin"],         cfg["vmax"],        cfg["label"])
            _show(axes[row, 1], wr,     cfg["cmap"],      cfg["vmin"],         cfg["vmax"],        cfg["label"])
            _show(axes[row, 2], wr - nr, cfg["diff_cmap"], -cfg["diff_vmax"],   cfg["diff_vmax"],  f'Δ {cfg["label"]}')

            for col in range(3):
                _contours(axes[row, col], release_mask, _RELEASE_COLOR, _RELEASE_LW)
                if crown_mask is not None:
                    _contours(axes[row, col], crown_mask, _CROWN_COLOR, _CROWN_LW, "dashed")

        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  → {out_path.name}")


def plot_time_series(
    stats_no_reinit: pd.DataFrame,
    stats_with_reinit: pd.DataFrame,
    event_date: datetime,
    out_path: Path,
):
    fields = list(FIELDS.keys())
    n = len(fields)

    with matplotlib.rc_context(_PLOT_RC):
        fig, axes = plt.subplots(n, 1, figsize=(13, 3.4 * n), sharex=True,
                                 gridspec_kw={"hspace": 0.35})
        fig.suptitle("Release-zone mean — no-reinit vs with-reinit",
                     fontsize=14, fontweight="bold", y=1.01, color="#1a1a1a")

        for ax, fname in zip(axes, fields):
            cfg = FIELDS[fname]

            # Subtle event shading (±12 h)
            ax.axvspan(event_date - timedelta(hours=12),
                       event_date + timedelta(hours=12),
                       color="#888888", alpha=0.09, zorder=0, lw=0)

            if fname in stats_no_reinit.columns:
                ax.plot(stats_no_reinit.index, stats_no_reinit[fname],
                        color=_COLOR_NO_REINIT, lw=2.0,
                        solid_capstyle="round", label="No reinit", zorder=3)
            if fname in stats_with_reinit.columns:
                ax.plot(stats_with_reinit.index, stats_with_reinit[fname],
                        color=_COLOR_WITH_REINIT, lw=2.0, linestyle="--",
                        solid_capstyle="round", label="With reinit", zorder=3)

            ax.axvline(event_date, color="#333333", linestyle=":",
                       lw=1.4, zorder=4, label="Jan 18 event")

            ax.set_ylabel(cfg["label"], fontsize=10, color="#333333", labelpad=4)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))

            ax.grid(True, axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#d0d0d0")
            ax.spines["bottom"].set_color("#d0d0d0")
            ax.tick_params(colors="#555555", length=3, width=0.7)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=8.5, loc="upper right",
                          frameon=False, labelcolor="#333333")

        # Date formatting on the shared x-axis
        loc = mdates.AutoDateLocator(minticks=5, maxticks=9)
        axes[-1].xaxis.set_major_locator(loc)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        axes[-1].tick_params(axis="x", labelsize=9, colors="#555555")

        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  → {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Index HTML
# ──────────────────────────────────────────────────────────────────────────────

def write_index_html(out_dir: Path) -> None:
    """Generate index.html in out_dir for browsing spatial comparison images."""
    spatial_dir = out_dir / "spatial"
    images = sorted(spatial_dir.glob("compare_*.png"))
    if not images:
        return

    rel_paths = [f"spatial/{img.name}" for img in images]

    def _fmt_label(stem: str) -> str:
        tag = stem.replace("compare_", "")          # e.g. "20260119_0000"
        parts = tag.split("_")
        d = parts[0]                                # "20260119"
        t = parts[1] if len(parts) > 1 else "0000"  # "0000"
        try:
            return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:]}"
        except Exception:
            return tag.replace("_", " ")

    labels = [_fmt_label(img.stem) for img in images]

    paths_js  = json.dumps(rel_paths)
    labels_js = json.dumps(labels)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reinit Comparison</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; color: #e0e0e0; font-family: sans-serif;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  #header {{ display: flex; align-items: center; gap: 16px; padding: 8px 16px;
             background: #2a2a2a; border-bottom: 1px solid #444; flex-shrink: 0; }}
  #header h1 {{ font-size: 1rem; font-weight: 600; }}
  #date-label {{ font-size: 1.1rem; font-weight: 700; color: #7ec8e3; min-width: 160px; }}
  #counter {{ font-size: 0.85rem; color: #888; }}
  .btn {{ background: #3a3a3a; border: 1px solid #555; color: #e0e0e0;
          padding: 5px 14px; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }}
  .btn:hover {{ background: #4a4a4a; }}
  #main {{ flex: 1; display: flex; overflow: hidden; }}
  #img-wrap {{ flex: 1; display: flex; align-items: center; justify-content: center;
               overflow: hidden; position: relative; }}
  #main-img {{ max-width: 100%; max-height: 100%; object-fit: contain;
               transition: opacity 0.1s; }}
  #strip-wrap {{ width: 120px; overflow-y: auto; background: #222;
                 border-left: 1px solid #444; flex-shrink: 0; padding: 4px; }}
  .thumb {{ cursor: pointer; margin-bottom: 4px; border: 2px solid transparent;
             border-radius: 3px; overflow: hidden; }}
  .thumb img {{ width: 100%; display: block; }}
  .thumb.active {{ border-color: #7ec8e3; }}
  .thumb .tlabel {{ font-size: 0.6rem; text-align: center; padding: 2px 0;
                    color: #aaa; background: #2a2a2a; }}
  #footer {{ padding: 4px 16px; background: #2a2a2a; border-top: 1px solid #444;
             font-size: 0.75rem; color: #666; flex-shrink: 0; }}
</style>
</head>
<body>
<div id="header">
  <h1>No-reinit vs With-reinit</h1>
  <span id="date-label"></span>
  <button class="btn" id="btn-prev">&#8592; Prev</button>
  <button class="btn" id="btn-next">Next &#8594;</button>
  <span id="counter"></span>
</div>
<div id="main">
  <div id="img-wrap">
    <img id="main-img" src="" alt="comparison plot">
  </div>
  <div id="strip-wrap" id="strip"></div>
</div>
<div id="footer">&#8592;&#8594; or A/D to navigate &nbsp;|&nbsp; Home/End for first/last</div>

<script>
const PATHS  = {paths_js};
const LABELS = {labels_js};
let idx = 0;

const mainImg   = document.getElementById('main-img');
const dateLabel = document.getElementById('date-label');
const counter   = document.getElementById('counter');
const stripWrap = document.getElementById('strip-wrap');

function buildStrip() {{
  PATHS.forEach((p, i) => {{
    const d = document.createElement('div');
    d.className = 'thumb' + (i === 0 ? ' active' : '');
    d.dataset.i = i;
    d.innerHTML = `<img src="${{p}}" loading="lazy"><div class="tlabel">${{LABELS[i]}}</div>`;
    d.addEventListener('click', () => show(i));
    stripWrap.appendChild(d);
  }});
}}

function show(i) {{
  idx = (i + PATHS.length) % PATHS.length;
  mainImg.style.opacity = 0.3;
  mainImg.src = PATHS[idx];
  mainImg.onload = () => {{ mainImg.style.opacity = 1; }};
  dateLabel.textContent = LABELS[idx];
  counter.textContent = (idx + 1) + ' / ' + PATHS.length;
  document.querySelectorAll('.thumb').forEach((el, j) => {{
    el.classList.toggle('active', j === idx);
    if (j === idx) el.scrollIntoView({{block: 'nearest'}});
  }});
}}

document.getElementById('btn-prev').addEventListener('click', () => show(idx - 1));
document.getElementById('btn-next').addEventListener('click', () => show(idx + 1));

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft'  || e.key === 'a') show(idx - 1);
  if (e.key === 'ArrowRight' || e.key === 'd') show(idx + 1);
  if (e.key === 'Home') show(0);
  if (e.key === 'End')  show(PATHS.length - 1);
}});

buildStrip();
show(0);
</script>
</body>
</html>
"""
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  → index.html  ({len(rel_paths)} images)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compare no-reinit vs with-reinit SNOWPACK runs.")
    ap.add_argument("--zarr-no-reinit",   required=True, type=Path)
    ap.add_argument("--zarr-with-reinit", required=True, type=Path)
    ap.add_argument("--smet-dir",         required=True, type=Path)
    ap.add_argument("--dem-tif",          required=True, type=Path)
    ap.add_argument("--release-geojson",  required=True, type=Path)
    ap.add_argument("--crown-geojson",    default=None,  type=Path)
    ap.add_argument("--event-date",       default="2026-01-18")
    ap.add_argument("--out-dir",          required=True, type=Path)
    ap.add_argument("--days-after",       type=int, default=60,
                    help="Number of days after event to plot (default 60)")
    ap.add_argument("--plot-interval-h",  type=int, default=24,
                    help="Time step between spatial plots in hours (default 24)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spatial").mkdir(exist_ok=True)

    event_dt = datetime.fromisoformat(args.event_date)
    print(f"Event date: {event_dt.date()}")

    # ── Load DEM ──
    with rasterio.open(args.dem_tif) as src:
        dem = src.read(1).astype(np.float32)
        dem_transform = src.transform
        dem_crs_wkt = src.crs.wkt
        dem_shape = src.shape
        dem_bounds = src.bounds

    # Hillshade for background
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem, vert_exag=2.0)
    hillshade = (hillshade * 255).astype(np.uint8)

    # Build pixel-coordinate meshgrid (for griddata output)
    rows_idx = np.arange(dem_shape[0])
    cols_idx = np.arange(dem_shape[1])
    grid_col, grid_row = np.meshgrid(cols_idx, rows_idx)
    # Convert pixel coords to UTM (for griddata target points)
    xs = dem_bounds.left + (grid_col + 0.5) * dem_transform.a
    ys = dem_bounds.top  + (grid_row + 0.5) * dem_transform.e  # e is negative

    # ── Load cluster coordinates and slope angles ──
    print("Loading cluster coordinates...")
    coords, slope_angles = load_cluster_coords_utm(args.smet_dir, dem_crs_wkt)
    print(f"  {len(coords)} clusters, {len(slope_angles)} with slope angles")

    # ── Load Zarr stores ──
    print("Opening Zarr stores...")
    z_nr = open_zarr(args.zarr_no_reinit)
    z_wr = open_zarr(args.zarr_with_reinit)

    times_nr = zarr_times_as_datetimes(z_nr)
    times_wr = zarr_times_as_datetimes(z_wr)
    print(f"  No-reinit:   {times_nr[0]} → {times_nr[-1]}  ({len(times_nr)} steps)")
    print(f"  With-reinit: {times_wr[0]} → {times_wr[-1]}  ({len(times_wr)} steps)")

    # ── Release and crown masks ──
    release_mask = build_release_mask(args.release_geojson, dem_transform, dem_shape)
    crown_mask = build_release_mask(args.crown_geojson, dem_transform, dem_shape) \
        if args.crown_geojson and args.crown_geojson.exists() else None

    # ── Build list of comparison dates (post-event, every plot_interval_h) ──
    compare_datetimes: list[datetime] = []
    t = event_dt + timedelta(hours=args.plot_interval_h)
    end_t = event_dt + timedelta(days=args.days_after)
    while t <= end_t:
        compare_datetimes.append(t)
        t += timedelta(hours=args.plot_interval_h)

    print(f"\nComputing features and grids for {len(compare_datetimes)} dates...")

    stats_nr_rows: list[dict] = []
    stats_wr_rows: list[dict] = []

    print(f"\nGenerating spatial comparison plots (every {args.plot_interval_h} h, # days: {len(compare_datetimes)})...")
    print(f" start date: {compare_datetimes[0]}  |  end date: {compare_datetimes[-1]}")
    for dt in compare_datetimes:
        ti_nr = find_time_index(times_nr, dt, tol_h=args.plot_interval_h // 2)
        ti_wr = find_time_index(times_wr, dt, tol_h=args.plot_interval_h // 2)

        if ti_nr is None or ti_wr is None:
            continue

        date_str = dt.strftime("%Y-%m-%d %H:%M")
        date_tag  = dt.strftime("%Y%m%d_%H%M")

        feats_nr = extract_cluster_features(z_nr, ti_nr, slope_angles)
        feats_wr = extract_cluster_features(z_wr, ti_wr, slope_angles)

        # Build scatter arrays — only clusters present in both and in coords dict
        common = set(feats_nr.index) & set(feats_wr.index) & set(coords.keys())
        if len(common) < 10:
            print(f"  {date_str}: only {len(common)} common clusters — skipping")
            continue

        common_sorted = sorted(common)
        xy = np.array([coords[c] for c in common_sorted])

        grids_nr: dict[str, np.ndarray] = {}
        grids_wr: dict[str, np.ndarray] = {}

        for fname in FIELDS:
            vals_nr = feats_nr.loc[common_sorted, fname].values.astype(float)
            vals_wr = feats_wr.loc[common_sorted, fname].values.astype(float)

            ok = np.isfinite(vals_nr) & np.isfinite(vals_wr)
            if ok.sum() < 10:
                continue

            g_nr = interpolate_to_grid(vals_nr[ok], xy[ok], xs, ys)
            g_wr = interpolate_to_grid(vals_wr[ok], xy[ok], xs, ys)

            grids_nr[fname] = g_nr
            grids_wr[fname] = g_wr

        # Release-zone mean stats
        row_nr = {"date": dt}
        row_wr = {"date": dt}
        for fname in FIELDS:
            for run_grids, row in [(grids_nr, row_nr), (grids_wr, row_wr)]:
                g = run_grids.get(fname)
                if g is not None:
                    vals_in = g[release_mask & np.isfinite(g)]
                    row[fname] = float(np.nanmean(vals_in)) if len(vals_in) > 0 else np.nan
                else:
                    row[fname] = np.nan
        stats_nr_rows.append(row_nr)
        stats_wr_rows.append(row_wr)

        # Spatial comparison plot
        out_spatial = args.out_dir / "spatial" / f"compare_{date_tag}.png"
        plot_comparison_date(
            grids_nr, grids_wr, hillshade, release_mask, crown_mask,
            date_str, out_spatial,
        )

    # ── Time series ──────────────────────────────────────────────────────────
    if stats_nr_rows:
        stats_nr = pd.DataFrame(stats_nr_rows).set_index("date")
        stats_wr = pd.DataFrame(stats_wr_rows).set_index("date")

        plot_time_series(stats_nr, stats_wr, event_dt,
                         args.out_dir / "release_zone_timeseries.png")

        # Save summary CSV
        merged = stats_nr.add_suffix("_no_reinit").join(stats_wr.add_suffix("_with_reinit"))
        merged.to_csv(args.out_dir / "release_zone_stats.csv")
        print(f"  → release_zone_stats.csv")

    write_index_html(args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
