"""
reinitialize_snowpack.py — Post-avalanche SNOWPACK reinitialization.

Detects the release area from UAS dHS (min-kernel method), identifies
which clusters are inside the release zone, and modifies their .sno
restart files by removing layers from the top down to the slab/WL
interface depth.

Two-pass SNOWPACK workflow:
  Pass 1:  Run SNOWPACK from season start to event date (already done)
  This:    Scour release cluster .sno files at event timestamp
  Pass 2:  Rerun SNOWPACK from event date to end of season

Called from forcing_pipeline.py as step_reinit(), or standalone:

    python src/avachain/reinitialize_snowpack.py \
        --date-before 2026-01-14 --date-after 2026-01-20 \
        --event-date 2026-01-18 \
        --snapshot-date 2026-01-18

    # Dry run
    python src/avachain/reinitialize_snowpack.py \
        --date-before 2026-01-14 --date-after 2026-01-20 \
        --event-date 2026-01-18 --dry-run

    # Use pre-drawn release GeoJSON instead of auto-detection
    python src/avachain/reinitialize_snowpack.py \
        --release-geojson data/boundaries/avalanche_release_area.geojson \
        --event-date 2026-01-18
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "src" / "avachain"))


# -----------------------------------------------------------------------
# .sno file reader/writer
# -----------------------------------------------------------------------

def read_sno(path: str) -> dict:
    """
    Parse a SNOWPACK .sno restart file.

    Returns dict with:
        header_lines : list of raw header lines (for passthrough)
        header       : dict of parsed header values
        fields       : list of column names
        layers       : list of dicts, one per layer (bottom to top)
    """
    header_lines = []
    header = {}
    fields = []
    layers = []
    in_header = True
    in_data = False

    with open(path) as f:
        for line in f:
            line_stripped = line.rstrip('\n')

            if in_header:
                header_lines.append(line_stripped)
                if line_stripped.strip() == '[DATA]':
                    in_header = False
                    in_data = True
                    continue
                if '=' in line_stripped and not line_stripped.startswith('#'):
                    key, val = line_stripped.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    header[key] = val
                    if key == 'fields':
                        fields = val.split()
                continue

            if in_data:
                parts = line_stripped.split()
                if len(parts) >= len(fields):
                    layer = {}
                    for i, fname in enumerate(fields):
                        layer[fname] = parts[i]
                    layers.append(layer)

    return {
        'header_lines': header_lines,
        'header': header,
        'fields': fields,
        'layers': layers,
        'path': path,
    }


def write_sno(sno_data: dict, path: str) -> None:
    """Write a modified .sno file."""
    with open(path, 'w') as f:
        for line in sno_data['header_lines']:
            f.write(line + '\n')
        for layer in sno_data['layers']:
            vals = [layer[fname] for fname in sno_data['fields']]
            f.write('     '.join(vals) + '\n')


def scour_sno(sno_data: dict,
              scour_depth_m: float,
              event_timestamp: str) -> dict:
    """
    Remove layers from the top of a .sno file to simulate avalanche scour.

    Parameters
    ----------
    sno_data       : parsed .sno from read_sno()
    scour_depth_m  : depth to remove from surface (m)
    event_timestamp: ISO timestamp for the new ProfileDate

    Returns modified sno_data (new copy).
    """
    import copy
    result = copy.deepcopy(sno_data)
    layers = result['layers']

    if not layers:
        return result

    # Layers are bottom to top — scour from the end
    removed_depth = 0.0
    n_original = len(layers)

    while layers and removed_depth < scour_depth_m:
        top_layer = layers[-1]
        layer_thick = float(top_layer['Layer_Thick'])

        remaining_to_scour = scour_depth_m - removed_depth

        if layer_thick <= remaining_to_scour:
            # Remove entire layer
            layers.pop()
            removed_depth += layer_thick
        else:
            # Partial removal — trim the top layer
            new_thick = layer_thick - remaining_to_scour
            if new_thick <= 0.0:
                layers.pop()
                removed_depth += layer_thick
            else:
                top_layer['Layer_Thick'] = f"{new_thick:.6f}"
                removed_depth += remaining_to_scour

    # Compute new HS
    new_hs = sum(float(l['Layer_Thick']) for l in layers)
    n_removed = n_original - len(layers)

    # Update header
    _update_header_line(result, 'nSnowLayerData', str(len(layers)))
    _update_header_line(result, 'HS_Last', f"{new_hs:.6f}")
    _update_header_line(result, 'ProfileDate', event_timestamp)
    _update_header_line(result, 'ErosionLevel', str(max(0, len(layers) - 1)))

    result['scour_stats'] = {
        'n_layers_original': n_original,
        'n_layers_remaining': len(layers),
        'n_layers_removed': n_removed,
        'scour_depth_requested_m': scour_depth_m,
        'scour_depth_actual_m': removed_depth,
        'hs_new_m': new_hs,
    }

    return result


def sanitize_sno_dir(sno_dir: Path, dry_run: bool = False) -> int:
    """Strip zero-thickness layers from all cluster .sno files in sno_dir.

    SNOWPACK writes restart files with Layer_Thick = 0 when a layer has fully
    compacted or melted.  Reading such a file in a subsequent run causes a
    divide-by-zero SIGFPE.  This pass removes those degenerate layers and
    updates nSnowLayerData / HS_Last / ErosionLevel accordingly.

    Returns the count of files that were modified.
    """
    import copy
    n_fixed = 0
    for sno_path in sorted(sno_dir.glob("cluster_*.sno")):
        sno_data = read_sno(str(sno_path))
        layers = sno_data['layers']
        clean = [l for l in layers if float(l['Layer_Thick']) > 1e-7]
        if len(clean) == len(layers):
            continue

        result = copy.deepcopy(sno_data)
        result['layers'] = clean
        new_hs = sum(float(l['Layer_Thick']) for l in clean)
        _update_header_line(result, 'nSnowLayerData', str(len(clean)))
        _update_header_line(result, 'HS_Last', f"{new_hs:.6f}")
        _update_header_line(result, 'ErosionLevel', str(max(0, len(clean) - 1)))

        n_removed = len(layers) - len(clean)
        print(f"  sanitize {sno_path.name}: {n_removed} zero-layer(s) removed, "
              f"nSnowLayerData {len(layers)}→{len(clean)}")
        if not dry_run:
            write_sno(result, str(sno_path))
        n_fixed += 1
    return n_fixed


def align_smet_hs(smet_path: Path,
                  event_ts: pd.Timestamp,
                  hs_event_m: float) -> int:
    """Shift a cluster's HS forcing so it agrees with its scoured .sno.

    SNOWPACK runs with ENFORCE_MEASURED_SNOW_HEIGHTS, so any excess of the
    forcing HS over the modelled HS is deposited as new snow. Reinit scours
    the .sno to the post-event survey HS, but the gap-fill forcing is not
    pinned to that survey (endpoint RMSE ~65 cm) and its avalanche dHS is a
    different, usually much smaller, field. Left alone, the restart deposits
    the difference straight back: on the 2026-03-03 replay event, a median
    37.9 cm per cluster, and cluster 5859 crashed on a 78 cm "snowfall".

    Every row from event_ts on is lowered by one constant offset, so the
    forcing equals hs_event_m at the event and keeps its own increments
    (snowfall, settlement) after it. A ramp that also pinned the forcing to
    the survey HS at the survey time was tried and removed: the survey HS
    already set the scoured state, so pinning it again cancels the
    settlement in between and SNOWPACK deposits that as fresh snow. On
    cluster 0727 (36 deg, 4.1 m) the forcing rose 14 cm in 30 h with no
    snowfall, and the surface node failed at 207 K under midday sun.

    The file stores HS in cm (header units_multiplier 0.01), hs_event_m is
    in m, both vertical (run_snowpack.sh projects the forcing onto the slope
    normal; see scour_cluster). Only the HS token of each row is rewritten, keeping
    every other byte.

    Returns the number of rows changed.
    """
    lines = smet_path.read_text().splitlines(keepends=True)
    i_data = next(i for i, l in enumerate(lines) if l.strip() == '[DATA]')
    fields = next(l.split('=', 1)[1].split() for l in lines[:i_data]
                  if l.split('=', 1)[0].strip() == 'fields')
    nodata = next((float(l.split('=', 1)[1]) for l in lines[:i_data]
                   if l.split('=', 1)[0].strip() == 'nodata'), -999.0)
    j = fields.index('HS')

    rows = []                                   # (line index, ts, hs_cm)
    for i in range(i_data + 1, len(lines)):
        parts = lines[i].split()
        if len(parts) <= j or parts[0].startswith('#'):
            continue
        hs = float(parts[j])
        if hs != nodata:
            rows.append((i, pd.Timestamp(parts[0]), hs))

    # Forcing HS at event_ts, linear between rows
    k = next((k for k, r in enumerate(rows) if r[1] >= event_ts), len(rows))
    if k == len(rows):
        return 0
    if rows[k][1] == event_ts:
        f_event = rows[k][2]
    elif k == 0:
        return 0
    else:
        (_, t0, h0), (_, t1, h1) = rows[k - 1], rows[k]
        f_event = h0 + (event_ts - t0) / (t1 - t0) * (h1 - h0)
    d = f_event - 100.0 * hs_event_m

    for i, t, hs in rows[k:]:
        tokens = re.split(r'(\s+)', lines[i])   # keeps the separators
        tok = tokens[2 * j]
        decimals = len(tok.split('.')[1]) if '.' in tok else 0
        tokens[2 * j] = f"{max(0.0, hs - d):.{decimals}f}"
        lines[i] = ''.join(tokens)
    smet_path.write_text(''.join(lines))
    return len(rows) - k


def align_scoured_forcing(smet_dir: Path, stats: list,
                          event_ts: pd.Timestamp) -> int:
    """align_smet_hs for every scoured cluster, to its scoured HS only.

    stats may carry survey_hs_m; it is deliberately not a forcing target
    (see align_smet_hs for why a ramp to it was removed). Returns the number
    of files changed.
    """
    n_files = n_rows = 0
    for s in stats:
        p = smet_dir / f"cluster_{s['cid']:04d}.smet"
        if not p.exists():
            continue
        k = align_smet_hs(p, event_ts, s['hs_after_m'])
        n_files += k > 0
        n_rows += k
    print(f"  HS forcing aligned to the scoured state: {n_files}/"
          f"{len(stats)} SMET file(s), {n_rows} row(s)")
    return n_files


def scour_cluster(sno_data: dict, event_ts: str,
                  survey_hs_m: float = None,
                  slab_thickness_m: float = None,
                  cid: int = 0):
    """Choose a cluster's scour depth and apply it.

    Scour sources, in order: the post-event survey HS (only passed in a clean
    window, and only when it is below the modelled HS), the slab thickness
    from the features CSV, else 80% of HS.

    Frames: .sno Layer_Thick is slope-normal. Everything it is compared with
    is vertical: the survey HS; the forcing HS, which run_snowpack.sh
    projects onto the slope normal (x cos) before SNOWPACK enforces it
    against the layer sum; and HS_mod and the .pro heights behind
    slab_thickness, which SNOWPACK reports as layer sum / cos. So this works
    in vertical and converts at the cut. All depths returned are vertical,
    scour_stats['hs_new_m'] too.

    Returns (scoured sno_data, hs_before_m, scour_m, source), or None when
    the scour is 1 cm or less.
    """
    cos_slope = vertical_to_normal(sno_data)
    hs_before = sum(float(l['Layer_Thick'])
                    for l in sno_data['layers']) / cos_slope

    if survey_hs_m is not None and hs_before - survey_hs_m > 0:
        scour_m = hs_before - survey_hs_m
        source = 'survey_HS'
    elif slab_thickness_m is not None:
        scour_m = slab_thickness_m
        source = 'features'
    else:
        scour_m = hs_before * 0.80
        source = 'HS×0.80'

    # Cap scour at current HS — can't remove more snow than exists.
    # This handles timing mismatches where slab_thickness is from
    # the snapshot date but the .sno is from a different time
    # (e.g., end of season with melt/settlement).
    if scour_m > hs_before:
        print(f"  {cid:04d}: scour {scour_m:.2f}m > HS {hs_before:.2f}m "
              f"— capping at HS")
        scour_m = hs_before

    if scour_m <= 0.01:
        return None

    scoured = scour_sno(sno_data, scour_m * cos_slope, event_ts)
    scoured['scour_stats']['hs_new_m'] /= cos_slope
    return scoured, hs_before, scour_m, source


def vertical_to_normal(sno_data: dict) -> float:
    """cos(slope): a vertical height x this = slope-normal .sno thickness.

    SNOWPACK reports HS_mod = sum(Layer_Thick) / cos(SlopeAngle), checked to
    0.01 cm on the replay restarts. The .sno SlopeAngle is the slope SNOWPACK
    runs (clamped at 55 deg), and the one run_snowpack.sh projects the
    forcing with.
    """
    angle = float(sno_data['header'].get('SlopeAngle', 0.0))
    return float(np.cos(np.radians(angle)))


# A survey cluster HS is trusted only with this much real coverage. Exact
# zeros are clipped negatives (spatial_model.resample clips at
# min_valid_hs_m = 0), i.e. survey bias/registration error, not bare ground:
# they cover 1-22% of the domain depending on the survey (02-12: 22%), and
# on 03-02 whole clusters read 0 between ~0.7 m on 02-21 and ~0.85 m on
# 03-05. Anchoring to them thinned 38 of 301 sampled clusters to nothing,
# which SNOWPACK cannot restart from. 10% of clusters are a single cell, and
# a median of one cell is that cell (3230: 0 -> 1.39 -> 0.23 m).
MIN_SURVEY_CELLS = 3
MIN_SURVEY_COVERAGE = 0.5


def survey_cluster_hs(hs_grid: np.ndarray, cluster_map: np.ndarray,
                      cids=None, min_cells: int = MIN_SURVEY_CELLS,
                      min_coverage: float = MIN_SURVEY_COVERAGE) -> dict:
    """Cluster HS from a survey grid: the median of the cluster's valid cells.

    Valid = finite and not exactly 0 (see MIN_SURVEY_CELLS). Clusters with
    fewer than min_cells valid cells, or less than min_coverage of their
    cells valid, are left out: no survey value is better than a wrong one.
    Median, not mean: single cells read 0 or 4+ m against their neighbours,
    and cluster 5859 (5 cells) swung 358 -> 0 -> 456 -> 6 cm across four
    December-January surveys on the mean.
    """
    inside = cluster_map > 0
    if cids is not None:
        inside &= np.isin(cluster_map, np.fromiter(cids, dtype=cluster_map.dtype))
    n_all = np.bincount(cluster_map[inside].ravel())
    ok = inside & np.isfinite(hs_grid) & (hs_grid != 0)
    ids = cluster_map[ok]
    vals = hs_grid[ok].astype(np.float64)
    order = np.argsort(ids, kind='stable')
    ids, vals = ids[order], vals[order]
    uniq, start = np.unique(ids, return_index=True)
    out = {}
    for c, v in zip(uniq, np.split(vals, start[1:])):
        if v.size >= min_cells and v.size >= min_coverage * n_all[c]:
            out[int(c)] = float(np.median(v))
    return out


def thin_sno(sno_data: dict, factor: float) -> dict:
    """Scale every layer's thickness by factor (0 < factor <= 1).

    The non-event HS correction: the model kept snow that settlement, melt or
    sublimation removed, so the whole column is thinned and the stratigraphy
    (each weak layer's relative depth) is kept. Volume fractions are
    unchanged, so each layer loses mass in proportion: conserving mass
    instead would raise density by 1/factor, and the corrections run to
    factor 0.2 (an ice fraction of 5x is not physical).
    """
    import copy
    result = copy.deepcopy(sno_data)
    for layer in result['layers']:
        layer['Layer_Thick'] = f"{float(layer['Layer_Thick']) * factor:.6f}"
    new_hs = sum(float(l['Layer_Thick']) for l in result['layers'])
    _update_header_line(result, 'HS_Last', f"{new_hs:.6f}")
    return result


def anchor_to_survey(cfg, survey_ts: pd.Timestamp, top_down_cids=(),
                     sno_dir: Path = None, smet_dir: Path = None,
                     tol_m: float = 0.02) -> dict:
    """Set every cluster's state at a new survey to the surveyed HS.

    The survey cycle re-runs S_prev -> S_new on the season gap-fill, but
    SNOWPACK's HS enforcement only adds snow: it deposits when the forcing is
    above the model and ignores it when below. So the model ratchets up on
    every high forcing step and survey noise: in the season run at 03-05,
    HS_mod sat a median 122 cm above its own forcing, 4010 of 6565 clusters
    more than 1 m above. Re-running from the previous survey cannot fix this,
    because the previous survey's state carries the excess.

    At S_new (the res .sno, dated survey_ts):
    - model above survey by more than tol_m: remove the excess. Clusters in
      top_down_cids (scoured by a wind-scour or avalanche event in this
      window) lose it from the top, like the event reinit; every other
      cluster is thinned (thin_sno), since the loss is settlement or melt.
    - model below survey: nothing here. The forcing is shifted below so it
      starts at the survey value, and SNOWPACK deposits the deficit at the
      first step, as in the season run where the next gap-fill interval
      starts at the survey.
    Then every cluster's forcing from survey_ts on is lowered or raised by a
    constant offset to start at the survey HS (align_smet_hs).

    HS is vertical throughout (layer sum / cos, see scour_cluster). Returns
    and saves per-cluster stats as anchor_stats_<date>.json.
    """
    sno_dir = sno_dir or cfg.pro_dir
    smet_dir = smet_dir or cfg.smet_dir
    date = survey_ts.strftime('%Y-%m-%d')
    stamp = survey_ts.strftime('%Y-%m-%dT%H:%M:%S')
    cluster_map = np.load(str(cfg.cluster_map_path))
    survey = survey_cluster_hs(
        np.load(str(cfg.resampled_dir / f"hs_{date}.npy")), cluster_map)
    # Clusters the survey is biased for (survey_validity) are not anchored;
    # gap-fill carries them forward from their last valid survey.
    from survey_validity import invalid_clusters
    for c in invalid_clusters(cfg.analysis_dir, date):
        survey.pop(c, None)
    top_down = set(int(c) for c in top_down_cids)

    stats, n = [], {'top_down': 0, 'thinned': 0, 'below': 0, 'held': 0,
                    'no_survey': 0}
    for cid, target in sorted(survey.items()):
        name = f"cluster_{cid:04d}"
        sno_path = sno_dir / f"{name}_{name}.sno"
        if not sno_path.exists():
            continue
        target = max(0.0, target)
        sno = read_sno(str(sno_path))
        cos_slope = vertical_to_normal(sno)
        hs = sum(float(l['Layer_Thick']) for l in sno['layers']) / cos_slope
        excess = hs - target
        if excess > tol_m:
            if cid in top_down:
                new = scour_sno(sno, excess * cos_slope, stamp)
                how = 'top_down'
            else:
                new = thin_sno(sno, target / hs)
                how = 'thinned'
            write_sno(new, str(sno_path))
            hs_new = sum(float(l['Layer_Thick']) for l in new['layers']) / cos_slope
        else:
            how = 'below' if excess < -tol_m else 'held'
            hs_new = hs
        n[how] += 1
        k = align_smet_hs(smet_dir / f"{name}.smet", survey_ts, target)
        stats.append({'cid': cid, 'hs_model_m': round(hs, 4),
                      'hs_survey_m': round(target, 4),
                      'hs_after_m': round(hs_new, 4), 'action': how,
                      'forcing_rows_aligned': k})

    n['no_survey'] = sum(1 for p in sno_dir.glob("cluster_*_cluster_*.sno")
                         if int(p.name.split('_')[1]) not in survey)
    exc = np.array([s['hs_model_m'] - s['hs_survey_m'] for s in stats])
    print(f"  Survey anchor {date}: {len(stats)} clusters — "
          f"{n['top_down']} top-down, {n['thinned']} thinned, "
          f"{n['below']} below survey (forcing deposits), {n['held']} within "
          f"{100 * tol_m:.0f} cm, {n['no_survey']} without a trusted survey value (left as modelled); "
          f"model - survey before: median {100 * np.median(exc):+.0f} cm, "
          f"p90 {100 * np.percentile(exc, 90):+.0f} cm", flush=True)
    out = cfg.analysis_dir / f"anchor_stats_{date}.json"
    out.write_text(json.dumps(stats, indent=1))
    return {'counts': n, 'stats': stats}


def _update_header_line(sno_data, key, new_value):
    """Update a header value in both the parsed dict and raw lines."""
    sno_data['header'][key] = new_value
    for i, line in enumerate(sno_data['header_lines']):
        if line.strip().startswith(key) and '=' in line:
            # Preserve original formatting/spacing
            parts = line.split('=', 1)
            # Right-pad key to original width
            sno_data['header_lines'][i] = f"{parts[0]}= {new_value}"
            return


# -----------------------------------------------------------------------
# Core workflow (callable from forcing_pipeline.py)
# -----------------------------------------------------------------------

def run_reinit(cfg,
               date_before: str = '2026-01-14',
               date_after: str = '2026-01-20',
               event_date: str = '2026-01-18',
               event_time: str = '12:00:00',
               snapshot_date: str = '2026-01-18',
               release_geojson: str = None,
               kernel_size: int = 7,
               threshold_sigma: float = 1.2,
               sno_dir: Path = None,
               smet_dir: Path = None,
               dry_run: bool = False,
               no_backup: bool = False,
               clean_window_max_stn_dhs_m: float = 0.05) -> dict:
    """
    Post-avalanche SNOWPACK reinitialization.

    Auto-detects (or loads) the release area, identifies release clusters,
    and scours their .sno restart files.

    Parameters
    ----------
    cfg : ProjectConfig
    date_before/after : survey dates bracketing the event
    event_date/time : avalanche event timestamp
    snapshot_date : SNOWPACK snapshot for slab thickness lookup
    release_geojson : path to pre-drawn release GeoJSON (None = auto-detect)
    kernel_size : min-kernel filter size
    threshold_sigma : min-kernel threshold in std devs
    sno_dir : directory with _res.sno files (default: cfg.pro_dir)
    smet_dir : cluster forcing SMETs to align to the scour (default: cfg.smet_dir)
    dry_run : if True, report without modifying files
    no_backup : if True, skip .sno.bak creation
    clean_window_max_stn_dhs_m : max station dHS (m) between event_date and
        date_after that qualifies as a "clean window".  When clean, cluster-mean
        HS from the post-event UAS survey is used as the scour target instead of
        the modeled slab thickness.  Default 0.05 m (5 cm).

    Returns
    -------
    dict with keys: n_scoured, n_missing, n_skipped, stats (list of dicts)
    """
    import rasterio
    from avalanche import detect_minkernel, load_slope_mask, \
        build_persistent_noise_mask

    sno_dir = sno_dir or cfg.pro_dir
    event_ts = f"{event_date}T{event_time}"

    # --- Load DEM + cluster map ---
    with rasterio.open(str(cfg.dem_1m_path)) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        transform = src.transform

    cluster_map = np.load(str(cfg.cluster_map_path))

    # --- Determine release mask ---
    if release_geojson:
        from snowpack_analysis import geojson_to_mask
        release_mask = geojson_to_mask(release_geojson, dem.shape, transform)
        print(f"Release mask from GeoJSON: {release_mask.sum()} cells")
    else:
        hs_before = np.load(str(cfg.resampled_dir /
                                f"hs_{date_before}.npy"))
        hs_after = np.load(str(cfg.resampled_dir /
                               f"hs_{date_after}.npy"))

        # Station dHS
        try:
            wx = pd.read_csv(str(cfg.weather_csv),
                             parse_dates=[0], index_col=0)
            hs_col = wx.iloc[:, 11]
            period = hs_col.loc[date_before:date_after]
            stn_dhs = float((period.iloc[-1] - period.iloc[0])
                            * 0.1 * 2.54)
        except Exception:
            stn_dhs = 0.0

        start_zone_mask = None
        if cfg.start_zone_kml.exists():
            start_zone_mask = load_slope_mask(
                cfg.start_zone_kml, dem.shape, transform)

        persistent_noise_mask = None
        all_surveys = sorted(cfg.resampled_dir.glob("hs_*.npy"))
        survey_pairs = []
        for i in range(len(all_surveys) - 1):
            d_b = all_surveys[i].name.replace("hs_", "").replace(".npy", "")
            d_a = all_surveys[i+1].name.replace("hs_", "").replace(".npy", "")
            if d_b == date_before and d_a == date_after:
                continue
            survey_pairs.append((all_surveys[i], all_surveys[i+1], 0.0))
        if survey_pairs:
            persistent_noise_mask = build_persistent_noise_mask(
                survey_pairs, dem, min_negative_fraction=0.6)

        print(f"Auto-detecting release area: {date_before} -> {date_after}")
        mk_result = detect_minkernel(
            hs_before, hs_after, dem, stn_dhs,
            kernel_size=kernel_size,
            threshold_sigma=threshold_sigma,
            start_zone_mask=start_zone_mask,
            persistent_noise_mask=persistent_noise_mask,
            transform=transform,
        )
        release_mask = mk_result['release_mask']
        print(f"  Release detected: {mk_result['release_area_m2']} m², "
              f"{mk_result['release_volume_m3']:.0f} m³")

    # --- Find release clusters ---
    release_cids = set(int(c) for c in np.unique(cluster_map[release_mask])
                       if c > 0)
    print(f"\nRelease clusters: {len(release_cids)}")

    # --- Get slab thickness per cluster ---
    feat_csv = cfg.analysis_dir / \
        f"all_start_zone_features_{snapshot_date}.csv"
    if not feat_csv.exists():
        feat_csv = cfg.analysis_dir / \
            f"release_zone_features_{snapshot_date}.csv"

    slab_thickness = {}
    if feat_csv.exists():
        snap_features = pd.read_csv(str(feat_csv), index_col=0)
        snap_features = snap_features[
            ~snap_features.index.duplicated(keep='first')]
        for cid in release_cids:
            if cid in snap_features.index and \
               'slab_thickness' in snap_features.columns:
                st = snap_features.loc[cid, 'slab_thickness']
                if isinstance(st, pd.Series):
                    st = st.iloc[0]
                try:
                    st = float(st)
                    if not np.isnan(st) and st > 0:
                        slab_thickness[cid] = st
                except (ValueError, TypeError):
                    pass
        print(f"  Slab thickness from features: "
              f"{len(slab_thickness)}/{len(release_cids)} clusters")
    else:
        print(f"  WARNING: {feat_csv.name} not found, "
              f"using 80% of HS as scour depth")

    # --- Post-event survey HS per cluster (Priority 1 scour source) ---
    # Use cluster HS from the post-event UAS grid when the window
    # between the event and the survey is free of new snow (clean window).
    hs_after_path = cfg.resampled_dir / f"hs_{date_after}.npy"
    post_survey_hs = {}   # cid -> cluster HS (cell median) on date_after
    clean_window = False

    if hs_after_path.exists():
        # Reinit's own rule since it started: zeros out, no size minimum.
        post_survey_hs = survey_cluster_hs(np.load(str(hs_after_path)),
                                           cluster_map, release_cids,
                                           min_cells=1, min_coverage=0.0)
        from survey_validity import invalid_clusters
        for c in invalid_clusters(cfg.analysis_dir, date_after):
            post_survey_hs.pop(c, None)

        # Station dHS from event_date to date_after to judge window cleanliness
        try:
            wx = pd.read_csv(str(cfg.weather_csv), parse_dates=[0], index_col=0)
            hs_col = wx.iloc[:, 11]
            window = hs_col.loc[event_date:date_after]
            if len(window) >= 2:
                stn_new = float((window.iloc[-1] - window.iloc[0]) * 0.1 * 2.54)
                clean_window = stn_new < clean_window_max_stn_dhs_m
                print(f"  Post-event stn_dHS={stn_new:.3f}m  "
                      f"window={'clean' if clean_window else 'contaminated, using slab_thickness'}")
            else:
                print("  Insufficient station data for clean-window check "
                      "— using slab_thickness")
        except Exception as exc:
            print(f"  Weather CSV error ({exc}) — using slab_thickness")

        print(f"  Post-survey HS available: "
              f"{len(post_survey_hs)}/{len(release_cids)} clusters")
    else:
        print(f"  {hs_after_path.name} not found — "
              f"scour from slab_thickness or HS×0.80")

    # --- Scour .sno files ---
    print(f"\nEvent timestamp: {event_ts}")
    print(f"SNO directory: {sno_dir}")
    if dry_run:
        print("DRY RUN — no files will be modified\n")

    n_scoured = 0
    n_missing = 0
    n_skipped = 0
    stats = []

    for cid in sorted(release_cids):
        sno_name = f"cluster_{cid:04d}_res.sno"
        sno_path = sno_dir / sno_name

        if not sno_path.exists():
            sno_name = f"cluster_{cid:04d}_cluster_{cid:04d}_res.sno"
            sno_path = sno_dir / sno_name

        if not sno_path.exists():
            sno_name = f"cluster_{cid:04d}_cluster_{cid:04d}.sno"
            sno_path = sno_dir / sno_name

        if not sno_path.exists():
            n_missing += 1
            continue

        sno_data = read_sno(str(sno_path))
        survey_hs = post_survey_hs.get(cid) if clean_window else None
        plan = scour_cluster(sno_data, event_ts, survey_hs,
                             slab_thickness.get(cid), cid=cid)
        if plan is None:
            n_skipped += 1
            continue
        scoured, hs_before_scour, scour_m, source = plan
        ss = scoured['scour_stats']

        stats.append({
            'cid': cid,
            'hs_before_m': hs_before_scour,
            'scour_depth_m': scour_m,
            'scour_source': source,
            'n_layers_removed': ss['n_layers_removed'],
            'hs_after_m': ss['hs_new_m'],
        })

        if not dry_run:
            if not no_backup:
                shutil.copy2(str(sno_path), str(sno_path) + '.bak')
            write_sno(scoured, str(sno_path))

        n_scoured += 1
        print(f"  {sno_name}: HS {hs_before_scour:.2f}m → "
              f"{ss['hs_new_m']:.2f}m  "
              f"(scour {scour_m:.2f}m, {ss['n_layers_removed']} layers, "
              f"{source})")

    # --- Summary ---
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"  Scoured:  {n_scoured}")
    print(f"  Missing:  {n_missing} (.sno file not found)")
    print(f"  Skipped:  {n_skipped} (scour depth ≤ 1cm)")

    if stats:
        scour_depths = [s['scour_depth_m'] for s in stats]
        hs_afters = [s['hs_after_m'] for s in stats]
        print(f"  Scour depth: median={np.median(scour_depths):.2f}m  "
              f"range=[{min(scour_depths):.2f}, {max(scour_depths):.2f}]m")
        print(f"  HS after:    median={np.median(hs_afters):.2f}m  "
              f"range=[{min(hs_afters):.2f}, {max(hs_afters):.2f}]m")

    # --- Validation: compare reinitialized HS against post-event survey HS ---
    val_flag_threshold_m = 0.10
    if post_survey_hs and stats:
        val_records = []
        flagged = []
        for s in stats:
            cid = s['cid']
            if cid not in post_survey_hs:
                continue
            survey_hs = post_survey_hs[cid]
            residual  = s['hs_after_m'] - survey_hs
            rec = {
                'cid':          cid,
                'hs_after_m':   s['hs_after_m'],
                'survey_hs_m':  round(survey_hs, 4),
                'residual_m':   round(residual, 4),
                'scour_source': s['scour_source'],
                'flag':         abs(residual) > val_flag_threshold_m,
            }
            val_records.append(rec)
            if rec['flag']:
                flagged.append(rec)

        if val_records:
            residuals = [r['residual_m'] for r in val_records]
            print(f"\nPost-reinit validation ({len(val_records)} clusters):")
            print(f"  Residual (model−survey):  "
                  f"median={np.median(residuals):+.3f}m  "
                  f"range=[{min(residuals):+.3f}, {max(residuals):+.3f}]m")
            if flagged:
                print(f"  Flagged (|res| > {val_flag_threshold_m:.2f}m): "
                      f"{len(flagged)} cluster(s)")
                for r in flagged:
                    print(f"    cluster {r['cid']:04d}: "
                          f"model={r['hs_after_m']:.2f}m  "
                          f"survey={r['survey_hs_m']:.2f}m  "
                          f"res={r['residual_m']:+.3f}m  "
                          f"[{r['scour_source']}]")
            else:
                print(f"  All residuals within ±{val_flag_threshold_m:.2f}m")

            for s, v in zip(
                    [s for s in stats if s['cid'] in {r['cid'] for r in val_records}],
                    val_records):
                s['survey_hs_m'] = v['survey_hs_m']
                s['residual_m']  = v['residual_m']
                s['validation_flag'] = v['flag']

    if stats and not dry_run:
        stats_path = cfg.analysis_dir / f"reinit_stats_{event_date}.json"
        with open(str(stats_path), 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved: {stats_path}")

    # Make the HS forcing agree with the scoured state, or SNOWPACK deposits
    # the scour straight back as new snow (see align_smet_hs).
    if stats and not dry_run:
        align_scoured_forcing(smet_dir or cfg.smet_dir, stats,
                              pd.Timestamp(event_ts))

    # Strip zero-thickness layers from all .sno files — prevents SIGFPE on
    # restart when SNOWPACK wrote a layer that compacted to zero thickness.
    n_sanitized = sanitize_sno_dir(sno_dir, dry_run=dry_run)
    if n_sanitized:
        print(f"  Sanitized {n_sanitized} .sno file(s) "
              f"(zero-thickness layers removed)")

    if not dry_run:
        print(f"\nNext step: rerun SNOWPACK from {event_date} "
              f"for the scoured clusters:")
        print(f"  bash snowpack/little_prof/run_snowpack.sh "
              f"{event_date}T00:00")

    return {
        'n_scoured': n_scoured,
        'n_missing': n_missing,
        'n_skipped': n_skipped,
        'stats': stats,
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Post-avalanche SNOWPACK reinitialization")
    ap.add_argument('--project-dir', type=Path, default=Path('.'))
    ap.add_argument('--date-before', default='2026-01-14')
    ap.add_argument('--date-after', default='2026-01-20')
    ap.add_argument('--event-date', default='2026-01-18')
    ap.add_argument('--event-time', default='12:00:00')
    ap.add_argument('--snapshot-date', default='2026-01-18')
    ap.add_argument('--release-geojson', default=None)
    ap.add_argument('--kernel-size', type=int, default=7)
    ap.add_argument('--threshold-sigma', type=float, default=1.2)
    ap.add_argument('--sno-dir', default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-backup', action='store_true')
    args = ap.parse_args()

    from config import ProjectConfig
    cfg = ProjectConfig(project_dir=args.project_dir)

    run_reinit(
        cfg=cfg,
        date_before=args.date_before,
        date_after=args.date_after,
        event_date=args.event_date,
        event_time=args.event_time,
        snapshot_date=args.snapshot_date,
        release_geojson=args.release_geojson,
        kernel_size=args.kernel_size,
        threshold_sigma=args.threshold_sigma,
        sno_dir=Path(args.sno_dir) if args.sno_dir else None,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )


if __name__ == '__main__':
    main()
    