"""
report_interactive.py

Description:
    The interactive parts of report.html: one probability map per forecast
    day (today, tomorrow, the day after), sharing a crosshair, zoom and pan,
    with P(reach) at the cursor on every day and its distance to the road;
    and a histogram of each scenario's closest approach to the road.

    Everything is embedded in the page, so report.html stays one file that
    can be copied or mailed. Per horizon the page carries:
      - the probability field as uint16 (P x 10000), gzipped. The browser
        decompresses it (DecompressionStream) for the cursor readout.
      - the same field as a viridis PNG for drawing.
    The hillshade PNG and the road geometry are shared, once per DEM.

    At the cursor, for every day: P(reach), how many scenarios reach the
    spot (of how many), and the weighted probability of damaging flow,
    i.e. peak pressure and flow depth at or above set thresholds. These
    come from each scenario's own rasters (reach_mask, peak_pressure,
    peak_flow_depth in <work_dir>/scenario_###/outputs/).

    Distance to the road is measured to where the avalanche path crosses
    US-6 (Ron, 2026-11-02), not to whichever bit of road is nearest the
    cursor. The path is the steepest descent on the pit-filled DEM from the
    top of the P >= 0.5 core. Within APPROACH_M of the road it stops
    following the terrain: there it would turn along the road-cut ditch, as
    water does and an avalanche does not. From there it continues in the
    track's own direction over the previous LOOKBACK_M until it meets the
    road. The distance is straight-line and horizontal, in map metres
    (EPSG:6342).
"""

# --------- Load Libraries --------
import base64
import datetime
import gzip
import json
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import geopandas as gpd
from matplotlib import colormaps
from matplotlib.colors import LightSource
from PIL import Image
from shapely.geometry import box

import avalanche_helpers as h


# Days shown even when no scenarios exist for them (Ron, 2026-11-02:
# today, tomorrow and the next day).
MIN_HORIZONS = 3
DAY_NAMES = {0: 'Today', 1: 'Tomorrow', 2: 'Day after tomorrow'}
P_SCALE = 10000          # P stored as uint16 P * P_SCALE
ROAD_MARGIN_M = 150.0    # road drawn this far beyond the raster
HIST_BIN_M = 25.0

# Damaging-flow thresholds for the cursor readout; output.impact_pressure_kpa
# and output.impact_depth_m override them. 30 kPa is the Swiss red-zone
# pressure (destroys wood-frame buildings); 5 kPa is far below that and
# still enough to damage or push a vehicle.
IMPACT_PRESSURE_KPA = (5.0, 30.0)
IMPACT_DEPTH_M = (1.0,)

# Avalanche path to the road (see the module docstring)
PATH_START_P = 0.5       # start at the highest cell of the P >= this core
APPROACH_M = 40.0        # stop following the terrain this close to the road
LOOKBACK_M = 100.0       # track direction taken over this much path above
EXTEND_M = 300.0         # how far that direction may be projected


# --------- Encoding ---------

def _png_b64(arr):
    # @param arr: uint8 array, (H, W) gray or (H, W, 4) RGBA
    # @returns: str base64 PNG
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _gz_b64(arr, dtype='<u2'):
    # @param arr: numpy array
    # @param dtype: stored type, uint16 (P x P_SCALE) or uint8 (counts)
    # @returns: str base64 of the gzipped raw bytes
    return base64.b64encode(gzip.compress(arr.astype(dtype).tobytes(), 6)).decode('ascii')


def hillshade_rgba(dem_path, shape):
    # Grey hillshade as RGBA, transparent where the DEM has no data.
    # Same light (315/45, z x1.3) as the static report map.
    # @param dem_path: Path to the DEM (must be on the probability grid)
    # @param shape: (H, W) the probability grid
    # @returns: uint8 (H, W, 4) or None when the DEM is off-grid
    dem, meta = h.read_raster(dem_path)
    if dem.shape != tuple(shape):
        logging.warning(f"hillshade skipped: DEM {dem.shape} is not on the "
                        f"probability grid {shape}")
        return None
    bad = ~np.isfinite(dem)
    if meta['nodata'] is not None:
        bad |= (dem == meta['nodata'])
    fill = dem[~bad].mean() if (~bad).any() else 0.0
    dem = np.where(bad, fill, dem)
    cell = abs(meta['transform'].a)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(dem * 1.3, dx=cell, dy=cell)
    g = (np.clip(shade, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([g, g, g, np.where(bad, 0, 255).astype(np.uint8)])
    return rgba


def probability_rgba(prob, alpha=0.8):
    # Viridis over [0, 1], transparent where P <= 0 or nodata (as the
    # static map: untouched terrain shows the hillshade only)
    # @param prob: float array
    # @returns: uint8 (H, W, 4)
    p = np.where(np.isfinite(prob), prob, 0.0)
    rgba = (colormaps['viridis'](np.clip(p, 0, 1)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(p > 0, int(alpha * 255), 0)
    return rgba


def road_lines(road_path, crs, bounds, margin=2000.0):
    # Road as lists of [x, y], clipped generously around the raster so the
    # distance readout sees every nearby segment
    # @param road_path: road shapefile / GeoJSON
    # @param crs: CRS of the probability raster
    # @param bounds: (left, bottom, right, top)
    # @returns: list of lists of [x, y]
    gdf = gpd.read_file(road_path)
    if gdf.crs is not None and crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    clip = box(bounds[0] - margin, bounds[1] - margin,
               bounds[2] + margin, bounds[3] + margin)
    lines = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        g = geom.intersection(clip)
        parts = getattr(g, 'geoms', [g])
        for part in parts:
            if part.geom_type == 'LineString' and len(part.coords) > 1:
                lines.append([[round(x, 2), round(y, 2)] for x, y, *_ in part.coords])
    return lines


# --------- Avalanche path to the road ---------

def filled_dem(dem, bad):
    # Pit-filled DEM where every cell has a strictly downhill path to the
    # grid edge or the nodata boundary (priority-flood + epsilon, Barnes et
    # al. 2014). ~2 s on the 1020 x 765 Little Professor DEM.
    # @param dem: float array
    # @param bad: bool array, True where the DEM has no data
    # @returns: float64 array (-inf on bad cells)
    import heapq
    H, W = dem.shape
    z = np.where(bad, -np.inf, dem).astype('float64')
    done = bad.copy()
    pad = np.pad(bad, 1, constant_values=True)
    near_bad = np.zeros_like(bad)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            near_bad |= pad[1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc]
    seeds = np.argwhere(near_bad & ~bad)
    pq = [(z[r, c], int(r), int(c)) for r, c in seeds]
    heapq.heapify(pq)
    done[near_bad & ~bad] = True
    while pq:
        zz, r, c = heapq.heappop(pq)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < H and 0 <= cc < W and not done[rr, cc]:
                    done[rr, cc] = True
                    if z[rr, cc] <= zz:
                        z[rr, cc] = zz + 1e-6
                    heapq.heappush(pq, (z[rr, cc], rr, cc))
    return z


def _steepest_descent(z, bad, r, c, transform, stop):
    # Cell-centre path down the filled DEM until stop(x, y) or no way down
    # @returns: list of (x, y)
    H, W = z.shape
    pts, seen = [], set()
    while True:
        x, y = transform * (c + 0.5, r + 0.5)
        pts.append((x, y))
        seen.add((r, c))
        if stop(x, y):
            break
        best = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < H and 0 <= cc < W and not bad[rr, cc]:
                    s = (z[r, c] - z[rr, cc]) / np.hypot(dr, dc)
                    if s > 0 and (best is None or s > best[0]):
                        best = (s, rr, cc)
        if best is None or (best[1], best[2]) in seen:
            break
        r, c = best[1], best[2]
    return pts


def avalanche_path(dem_path, pmax, road_lines_xy):
    # The avalanche path and where it crosses the road
    # @param dem_path: DEM on the probability grid
    # @param pmax: float array, max P(reach) over the days shown
    # @param road_lines_xy: list of lists of [x, y]
    # @returns: dict {line: [[x, y], ...], crossing: [x, y] or None,
    #          how: str} or None when there is nothing to trace
    from shapely.geometry import LineString, MultiLineString, Point
    from shapely.ops import nearest_points
    dem, meta = h.read_raster(dem_path)
    if dem.shape != pmax.shape or not road_lines_xy:
        return None
    bad = ~np.isfinite(dem)
    if meta['nodata'] is not None:
        bad |= (dem == meta['nodata'])
    core = (pmax >= PATH_START_P) & ~bad
    if not core.any():
        core = (pmax > 0) & ~bad
    if not core.any():
        return None
    rs, cs = np.nonzero(core)
    i = int(np.argmax(dem[rs, cs]))
    road = MultiLineString([LineString(ln) for ln in road_lines_xy])
    z = filled_dem(dem, bad)
    pts = _steepest_descent(z, bad, rs[i], cs[i], meta['transform'],
                            stop=lambda x, y: road.distance(Point(x, y)) <= APPROACH_M)
    end = Point(pts[-1])
    # direction of the track over the LOOKBACK_M above the end
    acc, j = 0.0, len(pts) - 1
    while j > 0 and acc < LOOKBACK_M:
        acc += np.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1])
        j -= 1
    dx, dy = pts[-1][0] - pts[j][0], pts[-1][1] - pts[j][1]
    n = np.hypot(dx, dy)
    crossing, how = None, ''
    if n > 0:
        ray = LineString([pts[-1], (pts[-1][0] + dx / n * EXTEND_M,
                                    pts[-1][1] + dy / n * EXTEND_M)])
        hit = ray.intersection(road)
        if not hit.is_empty:
            cands = [hit] if hit.geom_type == 'Point' else \
                [g for g in getattr(hit, 'geoms', []) if g.geom_type == 'Point']
            if not cands and hit.geom_type != 'Point':
                cands = [Point(c) for c in getattr(hit, 'coords', [])]
            if cands:
                first = min(cands, key=lambda q: q.distance(end))
                crossing = [first.x, first.y]
                how = 'track direction projected to the road'
    if crossing is None:
        q = nearest_points(end, road)[1]
        crossing = [q.x, q.y]
        how = 'road point nearest the end of the path (projection missed the road)'
    line = [[round(x, 2), round(y, 2)] for x, y in pts] + \
        [[round(crossing[0], 2), round(crossing[1], 2)]]
    return {'line': line, 'crossing': [round(crossing[0], 2), round(crossing[1], 2)],
            'how': how}


# --------- Per-scenario fields ---------

def scenario_fields(hz, shape, pressures_kpa, depths_m):
    # Per-cell scenario count and weighted P(damaging flow), from every
    # scenario's own rasters
    # @param hz: horizon dict (manifest work_dir, df with scenario weights)
    # @param shape: (H, W) of the probability grid
    # @param pressures_kpa / depths_m: thresholds
    # @returns: dict {count: uint8 array, n: int, impacts: [(label, P array)]}
    #          or None when the scenario folders are not on disk
    work = hz['manifest'].get('work_dir')
    if not work or not Path(work).is_dir():
        return None
    df = hz['df']
    weights = dict(zip(df['scenario_id'], df['scenario_probability'])) \
        if 'scenario_probability' in df.columns else {}
    count = np.zeros(shape, dtype=np.int32)
    acc = [np.zeros(shape) for _ in list(pressures_kpa) + list(depths_m)]
    wsum, n = 0.0, 0
    for sid in df['scenario_id']:
        out = Path(work) / sid / 'outputs'
        try:
            reach, _ = h.read_raster(out / 'reach_mask.tif')
            ppr, _ = h.read_raster(out / 'peak_pressure.tif')
            pft, _ = h.read_raster(out / 'peak_flow_depth.tif')
        except Exception as e:
            logging.warning(f"[{hz['label']}] {sid}: scenario rasters unreadable ({e}); "
                            f"counts and impact P left out")
            return None
        if reach.shape != tuple(shape):
            logging.warning(f"[{hz['label']}] {sid}: off-grid rasters; impact P left out")
            return None
        w = float(weights.get(sid, 1.0))
        count += (reach > 0)
        ppr = np.nan_to_num(ppr, nan=0.0)
        pft = np.nan_to_num(pft, nan=0.0)
        k = 0
        for t in pressures_kpa:
            acc[k] += w * (ppr >= t)
            k += 1
        for t in depths_m:
            acc[k] += w * (pft >= t)
            k += 1
        wsum += w
        n += 1
    labels = [f"P(pressure &ge; {t:g} kPa)" for t in pressures_kpa] + \
             [f"P(depth &ge; {t:g} m)" for t in depths_m]
    return {'count': np.minimum(count, 255).astype(np.uint8), 'n': n,
            'impacts': [(lab, a / wsum if wsum > 0 else a) for lab, a in zip(labels, acc)]}


def crossing_distances(hz, crossing, grid):
    # Each scenario's closest approach to the path's road crossing: the
    # smallest distance from any of its reach cells to the crossing point
    # @param hz: horizon dict (manifest work_dir, df)
    # @param crossing: [x, y]
    # @param grid: payload grid dict
    # @returns: dict scenario_id -> metres (None when it reaches nothing),
    #          or None when the scenario folders are not on disk
    work = hz['manifest'].get('work_dir')
    if not work or not Path(work).is_dir() or hz['df'].empty:
        return None
    xs = grid['left'] + (np.arange(grid['w']) + 0.5) * grid['res']
    ys = grid['top'] - (np.arange(grid['h']) + 0.5) * grid['res']
    dist = np.hypot(xs[None, :] - crossing[0], ys[:, None] - crossing[1])
    out = {}
    for sid in hz['df']['scenario_id']:
        try:
            reach, _ = h.read_raster(Path(work) / sid / 'outputs' / 'reach_mask.tif')
        except Exception as e:
            logging.warning(f"[{hz['label']}] {sid}: no reach mask ({e}); "
                            f"distances to the crossing left out")
            return None
        if reach.shape != dist.shape:
            return None
        m = reach > 0
        out[sid] = float(dist[m].min()) if m.any() else None
    return out


# --------- Payload ---------

def _horizon_number(label):
    return int(label.split('_', 1)[1])


def build_payload(horizons, cycle_date):
    # Everything the interactive map and histogram need, as one dict
    # @param horizons: completed horizon dicts from cycle_products.load_cycle
    # @param cycle_date: 'YYYYMMDD'
    # @returns: dict (JSON-serialisable) or None when there is nothing to show
    if not horizons:
        return None
    base = datetime.date(int(cycle_date[:4]), int(cycle_date[4:6]), int(cycle_date[6:]))
    by_n = {_horizon_number(hz['label']): hz for hz in horizons}
    n_max = max(MIN_HORIZONS - 1, max(by_n))

    out_cfg = (horizons[0]['config'].get('output') or {})
    pressures = tuple(out_cfg.get('impact_pressure_kpa') or IMPACT_PRESSURE_KPA)
    depths = tuple(out_cfg.get('impact_depth_m') or IMPACT_DEPTH_M)

    grid = None
    shades = {}
    roads = {}
    slots = []
    pmax = None
    first_dem = None
    for n in range(n_max + 1):
        valid = (base + datetime.timedelta(days=n)).isoformat()
        slot = {'n': n, 'label': f'T_{n}', 'name': f'T+{n}',
                'day': DAY_NAMES.get(n, f'Day +{n}'), 'valid': valid}
        hz = by_n.get(n)
        if hz is None:
            slot['missing'] = True
            slots.append(slot)
            continue
        slot['valid'] = hz['manifest'].get('valid_date', valid)

        prob, meta = h.read_raster(hz['dir'] / 'probability.tif')
        prob = np.where(np.isfinite(prob) & (prob > 0), prob, 0.0)
        t = meta['transform']
        this_grid = {'left': t.c, 'top': t.f, 'res': abs(t.a),
                     'w': int(prob.shape[1]), 'h': int(prob.shape[0])}
        if grid is None:
            grid = this_grid
            bounds = (t.c, t.f - abs(t.e) * prob.shape[0],
                      t.c + t.a * prob.shape[1], t.f)
        elif this_grid != grid:
            # one shared crosshair needs one grid; a mismatch is a pipeline bug
            raise ValueError(f"[{hz['label']}] probability grid differs from "
                             f"{slots[0]['label']}: {this_grid} vs {grid}")

        pmax = prob if pmax is None else np.maximum(pmax, prob)
        first_dem = first_dem or hz['dem']
        fields = scenario_fields(hz, prob.shape, pressures, depths)
        if fields is not None:
            slot['count_gz'] = _gz_b64(fields['count'], 'u1')
            slot['count_n'] = fields['n']
            slot['damage'] = [{'label': lab, 'gz': _gz_b64(np.round(a * P_SCALE))}
                               for lab, a in fields['impacts']]

        dem_key = str(hz['dem'])
        if dem_key not in shades:
            rgba = hillshade_rgba(hz['dem'], prob.shape)
            shades[dem_key] = _png_b64(rgba) if rgba is not None else None
        road_key = str(hz['config']['paths']['road'])
        if road_key not in roads:
            roads[road_key] = road_lines(road_key, meta['crs'], bounds)

        df = hz['df']
        scen = []
        if not df.empty:
            w = df['scenario_probability'] if 'scenario_probability' in df.columns \
                else np.ones(len(df)) / len(df)
            for (_, r), wi in zip(df.iterrows(), w):
                reached = bool(r.get('reached_road'))
                d = r.get('distance_to_road_m')
                d = 0.0 if reached else (float(d) if d is not None and np.isfinite(d) else None)
                scen.append({'id': r['scenario_id'], 'd': d, 'w': float(wi),
                             'reached': reached,
                             'runout': float(r['runout_max_m'])
                             if np.isfinite(r.get('runout_max_m', np.nan)) else None})
        road = hz['road']
        pw = road.get('fraction_reached_weighted')
        slot.update({
            'p_road': pw if pw is not None else road.get('fraction_reached', 0.0),
            'n_scen': int(road.get('n_scenarios') or len(scen)),
            'n_reached': int(road.get('n_reached') or 0),
            'prob_gz': _gz_b64(np.round(prob * P_SCALE)),
            'prob_png': _png_b64(probability_rgba(prob)),
            'shade': dem_key, 'road': road_key,
            'impacts': [p['xy'] for p in road.get('impact_points', []) if p.get('xy')],
            'scenarios': scen,
        })
        slots.append(slot)

    if grid is None:
        return None
    # View extent: the raster plus the road near it, as the static map shows
    xs = [x for lines in roads.values() for ln in lines for x, _ in ln]
    ys = [y for lines in roads.values() for ln in lines for _, y in ln]
    ext = list(bounds)
    if xs:
        m = ROAD_MARGIN_M
        ext = [max(min(ext[0], min(xs)), bounds[0] - m), max(min(ext[1], min(ys)), bounds[1] - m),
               min(max(ext[2], max(xs)), bounds[2] + m), min(max(ext[3], max(ys)), bounds[3] + m)]
    try:
        path = avalanche_path(first_dem, pmax,
                              [ln for lines in roads.values() for ln in lines])
    except Exception as e:
        logging.warning(f"avalanche path not traced: {e}", exc_info=True)
        path = None

    # Scenario distances (histogram, nearest miss, approach curve) are to
    # the path's road crossing, like the cursor readout (Ron, 2026-11-02).
    # A scenario that reaches the road anywhere stays at 0. Without a
    # crossing they fall back to the closest approach to any road point.
    dist_ref = 'road'
    if path and path.get('crossing'):
        maps = {}
        for slot in slots:
            if slot.get('missing'):
                continue
            hz = by_n[slot['n']]
            dm = crossing_distances(hz, path['crossing'], grid)
            if dm is None:
                maps = None
                break
            maps[slot['n']] = dm
        if maps is not None:
            dist_ref = 'crossing'
            for slot in slots:
                for q in slot.get('scenarios', []):
                    q['d_road'] = q['d']
                    dc = maps[slot['n']].get(q['id'])
                    q['d'] = 0.0 if q['reached'] else (round(dc, 1) if dc is not None else None)
    return {'dist_ref': dist_ref,'grid': grid, 'extent': [round(v, 2) for v in ext], 'p_scale': P_SCALE,
            'hist_bin_m': HIST_BIN_M, 'shades': shades, 'roads': roads,
            'slots': slots, 'path': path,
            'impact_labels': [f"P(pressure &ge; {t:g} kPa)" for t in pressures] +
                             [f"P(depth &ge; {t:g} m)" for t in depths]}


def payload_json(payload):
    # JSON safe to drop inside <script type="application/json">
    return json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')


# --------- Page parts ---------

CSS = """
  .imap-read { border: 1px solid #000; padding: 0.35em 0.7em; margin: 0.4em 0 0.6em;
               font-size: 13px; min-height: 1.4em; font-variant-numeric: tabular-nums; }
  .imap-read b { font-weight: bold; }
  .imap-tab { width: auto; font-size: 12px; }
  .imap-tab th, .imap-tab td { padding: 0.15em 0.6em; }
  .imap-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .imap-cell { flex: 1 1 300px; min-width: 260px; }
  .imap-title { font-size: 12px; text-align: center; margin-bottom: 3px; }
  .imap-title b { font-size: 13px; }
  .imap-box { position: relative; border: 1px solid #000; background: #fff; box-sizing: border-box;
              overflow: hidden;
              cursor: crosshair; touch-action: none; }
  .imap-box canvas { position: absolute; left: 0; top: 0; display: block; width: 100%; height: 100%; }
  .imap-tag { position: absolute; left: 6px; top: 6px; background: rgba(255,255,255,0.88);
              border: 1px solid #000; padding: 1px 6px; font-size: 12px; pointer-events: none;
              font-variant-numeric: tabular-nums; }
  .imap-none { display: flex; align-items: center; justify-content: center; text-align: center;
               color: #555; font-size: 12px; padding: 1em; cursor: default; }
  .imap-legend { display: flex; align-items: center; gap: 8px; font-size: 11px; margin-top: 6px; flex-wrap: wrap; }
  .imap-ramp { width: 220px; height: 10px; border: 1px solid #000;
               background: linear-gradient(to right, #440154, #482878, #3e4989, #31688e, #26828e,
               #1f9e89, #35b779, #6ece58, #b5de2b, #fde725); }
  .imap-swatch { display: inline-block; width: 22px; height: 0; border-top: 3px solid #E87722;
                 vertical-align: middle; }
  .imap-help { font-size: 11px; color: #333; }
  .ihist { position: relative; }
  .ihist svg { width: 100%; height: auto; display: block; }
  .ihist-tip { position: absolute; pointer-events: none; background: #fff; border: 1px solid #000;
               padding: 3px 7px; font-size: 12px; display: none; white-space: nowrap;
               font-variant-numeric: tabular-nums; }
  @media print { .imap-screen { display: none; } .imap-print { display: block !important; } }
"""

HTML = """
<div class="imap-screen">
<div class="imap-read" id="imap-read">Move over a map: for each day, the probability that flow
reaches that spot, how many scenarios reach it, and the probability of damaging pressure or depth
there, with its distance from where the avalanche path crosses the road.</div>
<div class="imap-row" id="imap-row"></div>
<div class="imap-legend"><span>P(reach)</span><span>0</span><span class="imap-ramp"></span><span>1</span>
<span style="margin-left:14px"><span class="imap-swatch"></span> road</span>
<span style="margin-left:14px">&#9679; road impact</span>
<span style="margin-left:14px">- - - avalanche path, &#9711; where it crosses the road</span>
<span class="imap-help" style="margin-left:14px">Scroll to zoom, drag to pan, double-click to reset,
click to pin the readout (the address bar then links to that spot).</span></div>
</div>
"""

HIST_HTML = """
<div class="ihist" id="ihist"><svg id="ihist-svg"></svg><div class="ihist-tip" id="ihist-tip"></div></div>
<p class="imap-help">Each scenario is counted once, at its flow's closest approach to where the
avalanche path crosses the road ({{BIN}} m bins). The first bar is scenarios that reach the road
anywhere. The tooltip gives the weighted share as well.</p>
"""

JS = r"""
(function () {
  const D = JSON.parse(document.getElementById('imap-data').textContent);
  const G = D.grid, E = D.extent;
  const row = document.getElementById('imap-row');
  const read = document.getElementById('imap-read');
  const ROAD = '#E87722', REACH = '#A6192E';

  // ---- data ----
  async function gunzip(b64, T) {
    const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const ds = new DecompressionStream('gzip');
    const buf = await new Response(new Blob([bin]).stream().pipeThrough(ds)).arrayBuffer();
    return new T(buf);
  }
  function img(b64) { const i = new Image(); i.src = 'data:image/png;base64,' + b64; return i; }
  const shades = {}; for (const k in D.shades) if (D.shades[k]) shades[k] = img(D.shades[k]);

  // nearest point on the road and its distance (map metres)
  function nearestRoad(lines, x, y) {
    let best = {d: Infinity, x: 0, y: 0};
    for (const ln of lines) for (let i = 1; i < ln.length; i++) {
      const ax = ln[i-1][0], ay = ln[i-1][1], bx = ln[i][0], by = ln[i][1];
      const dx = bx - ax, dy = by - ay, L = dx*dx + dy*dy;
      let t = L > 0 ? ((x-ax)*dx + (y-ay)*dy) / L : 0; t = Math.max(0, Math.min(1, t));
      const px = ax + t*dx, py = ay + t*dy, d = Math.hypot(x-px, y-py);
      if (d < best.d) best = {d, x: px, y: py};
    }
    return best;
  }
  // the road reference: where the avalanche path crosses the road
  function refPoint(lines, x, y) {
    if (D.path && D.path.crossing) { const c = D.path.crossing;
      return {d: Math.hypot(x - c[0], y - c[1]), x: c[0], y: c[1]}; }
    return nearestRoad(lines, x, y);
  }

  // ---- shared view ----
  const view = {cx: (E[0]+E[2])/2, cy: (E[1]+E[3])/2, k: null};
  const maps = [];
  let cursor = null, pinned = false;

  function fit(m) {
    const kx = m.cv.clientWidth / (E[2]-E[0]), ky = m.cv.clientHeight / (E[3]-E[1]);
    return Math.min(kx, ky);
  }
  function resetView() { view.cx = (E[0]+E[2])/2; view.cy = (E[1]+E[3])/2; view.k = null; }
  function toScreen(m, x, y) {
    const k = view.k || fit(m);
    return [(x - view.cx) * k + m.cv.clientWidth/2, (view.cy - y) * k + m.cv.clientHeight/2];
  }
  function toWorld(m, sx, sy) {
    const k = view.k || fit(m);
    return [(sx - m.cv.clientWidth/2) / k + view.cx, view.cy - (sy - m.cv.clientHeight/2) / k];
  }
  function cellAt(x, y) {
    const c = Math.floor((x - G.left) / G.res), r = Math.floor((G.top - y) / G.res);
    return (c < 0 || r < 0 || c >= G.w || r >= G.h) ? -1 : r * G.w + c;
  }
  function pAt(m, x, y) {
    if (!m.p) return null;
    const i = cellAt(x, y); return i < 0 ? 0 : m.p[i] / D.p_scale;
  }

  function draw(m) {
    const cv = m.cv, dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    // resize the backing store whenever EITHER side changes: checking the
    // width alone left a later height change to stretch the drawing
    if (cv.width !== Math.round(W*dpr) || cv.height !== Math.round(H*dpr)) {
      cv.width = Math.round(W*dpr); cv.height = Math.round(H*dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const [x0, y0] = toScreen(m, G.left, G.top);
    const k = view.k || fit(m), w = G.w * G.res * k, h = G.h * G.res * k;
    g.imageSmoothingEnabled = true;
    if (m.shade && m.shade.complete) g.drawImage(m.shade, x0, y0, w, h);
    g.imageSmoothingEnabled = k < 2;          // crisp cells once zoomed in
    if (m.pimg && m.pimg.complete) g.drawImage(m.pimg, x0, y0, w, h);
    // road
    g.strokeStyle = ROAD; g.lineWidth = 2.5; g.lineJoin = 'round';
    for (const ln of m.road) { g.beginPath();
      ln.forEach((p, i) => { const [sx, sy] = toScreen(m, p[0], p[1]); i ? g.lineTo(sx, sy) : g.moveTo(sx, sy); });
      g.stroke(); }
    // avalanche path and its road crossing
    if (D.path) {
      for (const [col, lw, dash] of [['#fff', 3, []], ['#000', 1.2, [5, 4]]]) {
        g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.beginPath();
        D.path.line.forEach((p, i) => { const [sx, sy] = toScreen(m, p[0], p[1]); i ? g.lineTo(sx, sy) : g.moveTo(sx, sy); });
        g.stroke(); }
      g.setLineDash([]);
      const [cx, cy] = toScreen(m, D.path.crossing[0], D.path.crossing[1]);
      g.fillStyle = '#fff'; g.strokeStyle = '#000'; g.lineWidth = 2;
      g.beginPath(); g.arc(cx, cy, 6, 0, 2*Math.PI); g.fill(); g.stroke();
      g.strokeStyle = ROAD; g.lineWidth = 2.5; g.beginPath();
      g.moveTo(cx-3.5, cy-3.5); g.lineTo(cx+3.5, cy+3.5); g.moveTo(cx+3.5, cy-3.5); g.lineTo(cx-3.5, cy+3.5); g.stroke();
    }
    // impacts
    g.fillStyle = REACH;
    for (const p of m.slot.impacts || []) { const [sx, sy] = toScreen(m, p[0], p[1]);
      g.beginPath(); g.arc(sx, sy, 3.5, 0, 2*Math.PI); g.fill(); }
    // cursor: crosshair and the line to the nearest road point
    if (cursor) {
      const [sx, sy] = toScreen(m, cursor.x, cursor.y);
      const nr = refPoint(m.road, cursor.x, cursor.y);
      const [rx, ry] = toScreen(m, nr.x, nr.y);
      // white under black so both read over hillshade and viridis
      for (const [col, lw] of [['#fff', 4], ['#000', 1.6]]) {
        g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(col === '#000' ? [6, 4] : []);
        g.beginPath(); g.moveTo(sx, sy); g.lineTo(rx, ry); g.stroke(); }
      g.setLineDash([]);
      g.fillStyle = '#000'; g.strokeStyle = '#fff'; g.lineWidth = 1.5;
      g.beginPath(); g.arc(rx, ry, 4, 0, 2*Math.PI); g.fill(); g.stroke();
      const mx = (sx + rx) / 2, my = (sy + ry) / 2, lab = Math.round(nr.d) + ' m';
      g.font = 'bold 12px Arial'; const tw = g.measureText(lab).width;
      g.fillStyle = 'rgba(255,255,255,0.9)'; g.fillRect(mx - tw/2 - 3, my - 9, tw + 6, 16);
      g.fillStyle = '#000'; g.textAlign = 'center'; g.fillText(lab, mx, my + 3); g.textAlign = 'left';
      const cc = pinned ? REACH : '#000';
      for (const [col, lw] of [['#fff', 4.5], [cc, 2]]) {
        g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
        g.moveTo(sx-11, sy); g.lineTo(sx-3, sy); g.moveTo(sx+3, sy); g.lineTo(sx+11, sy);
        g.moveTo(sx, sy-11); g.lineTo(sx, sy-3); g.moveTo(sx, sy+3); g.lineTo(sx, sy+11); g.stroke(); }
      const p = pAt(m, cursor.x, cursor.y);
      m.tag.textContent = 'P = ' + (p === null ? '-' : p.toFixed(2));
      m.tag.style.display = '';
    } else { m.tag.style.display = 'none'; }
  }
  function drawAll() { maps.forEach(draw); readout(); }

  function readout() {
    if (!cursor || !maps.length) { return; }
    const nr = refPoint(maps[0].road, cursor.x, cursor.y);
    const i = cellAt(cursor.x, cursor.y);
    const IL = D.impact_labels || [];
    let h = '<table class="imap-tab"><thead><tr><th class="l">' + (pinned ? 'Pinned spot' : 'At the cursor') +
      '</th><th>P(reach)</th><th>Scenarios reaching</th>' + IL.map(l => '<th>' + l + '</th>').join('') +
      '</tr></thead><tbody>';
    for (const s of D.slots) {
      const m = maps.find(q => q.slot === s);
      h += '<tr><td class="l"><b>' + s.day + '</b> (' + s.name + ')</td>';
      if (!m) { h += '<td colspan="' + (2 + IL.length) + '" class="c">no forecast</td></tr>'; continue; }
      const p = pAt(m, cursor.x, cursor.y);
      const cnt = (m.count && i >= 0) ? m.count[i] : (m.count ? 0 : null);
      h += '<td>' + (p === null ? '-' : p.toFixed(2)) + '</td>';
      h += '<td>' + (cnt === null ? '-' : cnt + ' of ' + s.count_n) + '</td>';
      IL.forEach((_, k) => { const a = m.imp && m.imp[k];
        h += '<td>' + (a ? (i >= 0 ? (a[i] / D.p_scale).toFixed(2) : '0.00') : '-') + '</td>'; });
      h += '</tr>';
    }
    h += '</tbody></table>';
    const what = (D.path && D.path.crossing) ? 'from where the avalanche path crosses the road' : 'from the nearest road point';
    h += '<div style="margin-top:3px"><b>' + Math.round(nr.d) + ' m</b> ' + what +
      ' &nbsp;|&nbsp; E ' + Math.round(cursor.x) + ', N ' + Math.round(cursor.y) + '</div>';
    read.innerHTML = h;
  }

  // ---- build ----
  function boxHeight(el) { return Math.round(el.clientWidth * (E[3]-E[1]) / (E[2]-E[0])); }
  D.slots.forEach(async s => {
    const cell = document.createElement('div'); cell.className = 'imap-cell';
    const title = document.createElement('div'); title.className = 'imap-title';
    title.innerHTML = '<b>' + s.day + '</b> (' + s.name + '), valid ' + s.valid +
      (s.missing ? '<br>no forecast' : '<br>P(reach road) = ' + s.p_road.toFixed(2) +
       ', ' + s.n_reached + ' of ' + s.n_scen + ' scenarios reach');
    const bx = document.createElement('div'); bx.className = 'imap-box';
    cell.append(title, bx); row.append(cell);
    if (s.missing) {
      bx.classList.add('imap-none');
      bx.textContent = 'No forecast for ' + s.valid + ': no scenarios were delivered for ' + s.name + '.';
      new ResizeObserver(() => bx.style.height = boxHeight(bx) + 'px').observe(bx);
      return;
    }
    const cv = document.createElement('canvas');
    const tag = document.createElement('div'); tag.className = 'imap-tag'; tag.style.display = 'none';
    bx.append(cv, tag);
    const m = {slot: s, cv, tag, p: null, road: D.roads[s.road] || [],
               shade: shades[s.shade], pimg: img(s.prob_png)};
    maps.push(m);
    m.pimg.onload = () => draw(m); if (m.shade) m.shade.addEventListener('load', () => draw(m));
    new ResizeObserver(() => { bx.style.height = boxHeight(bx) + 'px'; draw(m); }).observe(bx);
    try { m.p = await gunzip(s.prob_gz, Uint16Array);
      if (s.count_gz) m.count = await gunzip(s.count_gz, Uint8Array);
      if (s.damage) m.imp = await Promise.all(s.damage.map(o => gunzip(o.gz, Uint16Array)));
    } catch (e) { m.p = m.p || null; console.warn(e); }
    drawAll();

    // interaction (shared across maps)
    let drag = null;
    cv.addEventListener('mousemove', ev => {
      const r = cv.getBoundingClientRect(), sx = ev.clientX - r.left, sy = ev.clientY - r.top;
      if (drag) { const k = view.k || fit(m); view.k = k;
        view.cx = drag.cx - (sx - drag.sx) / k; view.cy = drag.cy + (sy - drag.sy) / k;
        drag.moved = true; drawAll(); return; }
      if (pinned) return;
      const [x, y] = toWorld(m, sx, sy); cursor = {x, y}; drawAll();
    });
    cv.addEventListener('mouseleave', () => { drag = null; if (!pinned) { cursor = null; drawAll(); } });
    cv.addEventListener('mousedown', ev => { const r = cv.getBoundingClientRect();
      drag = {sx: ev.clientX - r.left, sy: ev.clientY - r.top, cx: view.cx, cy: view.cy, moved: false}; });
    cv.addEventListener('mouseup', ev => {
      if (drag && !drag.moved) { const r = cv.getBoundingClientRect();
        const [x, y] = toWorld(m, ev.clientX - r.left, ev.clientY - r.top);
        pinned = !pinned; cursor = {x, y};
        history.replaceState(null, '', pinned ? '#at=' + Math.round(x) + ',' + Math.round(y)
                                              : location.pathname + location.search);
        drawAll(); }
      drag = null; });
    cv.addEventListener('wheel', ev => { ev.preventDefault();
      const r = cv.getBoundingClientRect(), sx = ev.clientX - r.left, sy = ev.clientY - r.top;
      const [x, y] = toWorld(m, sx, sy), k0 = view.k || fit(m);
      const k = Math.max(fit(m) * 0.8, Math.min(k0 * Math.pow(1.0015, -ev.deltaY), 40));
      view.k = k; view.cx = x - (sx - cv.clientWidth/2) / k; view.cy = y + (sy - cv.clientHeight/2) / k;
      drawAll(); }, {passive: false});
    cv.addEventListener('dblclick', () => { resetView(); drawAll(); });
    // #at=E,N pins the readout there (a shareable link to one spot)
    const at = /#at=(-?[\d.]+),(-?[\d.]+)/.exec(location.hash);
    if (at) { cursor = {x: +at[1], y: +at[2]}; pinned = true; }
    drawAll();
  });

  // ---- histogram: closest approach to the road, # scenarios ----
  const svg = document.getElementById('ihist-svg'), tip = document.getElementById('ihist-tip');
  const have = D.slots.filter(s => !s.missing && s.scenarios && s.scenarios.length);
  if (!have.length) { document.getElementById('ihist').textContent = 'No scenarios.'; return; }
  const B = D.hist_bin_m;
  let dmax = 0; have.forEach(s => s.scenarios.forEach(q => { if (q.d !== null) dmax = Math.max(dmax, q.d); }));
  const nb = Math.max(1, Math.ceil(dmax / B));         // bins (0,B], (B,2B] ... ; bin -1 = reached
  const counts = have.map(s => { const c = new Array(nb + 1).fill(0), w = new Array(nb + 1).fill(0), ids = [...Array(nb + 1)].map(() => []);
    s.scenarios.forEach(q => { if (q.d === null) return;
      const i = q.reached ? 0 : Math.min(nb, Math.max(1, Math.ceil(q.d / B)));
      c[i]++; w[i] += q.w; ids[i].push(q.id); });
    return {s, c, w, ids}; });
  const cmax = Math.max(1, ...counts.flatMap(o => o.c));
  const VW = 900, VH = 300, ML = 48, MR = 12, MT = 26, MB = 44;
  const pw = VW - ML - MR, ph = VH - MT - MB, slotW = pw / (nb + 1), gap = 2;
  const barW = Math.max(2, (slotW - gap * 2) / counts.length);
  const fills = ['#222222', '#7A8893', '#C9CED2', '#555555', '#A0A8AE'];
  svg.setAttribute('viewBox', '0 0 ' + VW + ' ' + VH);
  let out = '';
  const yv = v => MT + ph - v / cmax * ph;
  const step = cmax <= 10 ? 1 : cmax <= 25 ? 5 : 10;
  for (let v = 0; v <= cmax; v += step) out += '<line x1="' + ML + '" x2="' + (VW-MR) + '" y1="' + yv(v) + '" y2="' + yv(v) +
    '" stroke="#DDD" stroke-width="0.6"/><text x="' + (ML-6) + '" y="' + (yv(v)+4) + '" font-size="11" text-anchor="end">' + v + '</text>';
  counts.forEach((o, j) => o.c.forEach((c, i) => { if (!c) return;
    const x = ML + i * slotW + gap + j * barW, y = yv(c);
    out += '<rect data-i="' + i + '" x="' + x + '" y="' + y + '" width="' + (barW - 0.5) + '" height="' + (MT + ph - y) +
      '" fill="' + (i === 0 ? '#A6192E' : fills[j % fills.length]) + '"/>'; }));
  for (let i = 0; i <= nb; i++) {
    const x = ML + i * slotW;
    out += '<rect class="hit" data-i="' + i + '" x="' + x + '" y="' + MT + '" width="' + slotW + '" height="' + ph + '" fill="transparent"/>';
    const lab = i === 0 ? 'reached' : String(i * B);
    if (i === 0 || nb <= 24 || i % 2 === 0)
      out += '<text x="' + (x + (i === 0 ? slotW/2 : slotW)) + '" y="' + (MT + ph + 14) + '" font-size="10.5" text-anchor="middle">' + lab + '</text>';
  }
  out += '<line x1="' + (ML + slotW) + '" x2="' + (ML + slotW) + '" y1="' + MT + '" y2="' + (MT + ph) + '" stroke="#E87722" stroke-width="1.5"/>';
  out += '<line x1="' + ML + '" x2="' + (VW-MR) + '" y1="' + (MT+ph) + '" y2="' + (MT+ph) + '" stroke="#000"/>';
  const xlab = D.dist_ref === 'crossing' ? "Closest approach to where the avalanche path crosses the road (m), bin upper edge"
                                         : 'Closest approach to the road (m), bin upper edge';
  out += '<text x="' + (ML + pw/2) + '" y="' + (VH - 6) + '" font-size="12" text-anchor="middle">' + xlab + '</text>';
  out += '<text transform="translate(13,' + (MT + ph/2) + ') rotate(-90)" font-size="12" text-anchor="middle"># scenarios</text>';
  // legend
  counts.forEach((o, j) => { const lx = ML + 8 + j * 170, ly = 4;
    out += '<rect x="' + lx + '" y="' + ly + '" width="12" height="10" fill="' + fills[j % fills.length] + '"/>' +
      '<text x="' + (lx + 18) + '" y="' + (ly + 9) + '" font-size="11">' + o.s.day + ' (' + o.s.name + ')</text>'; });
  svg.innerHTML = out;
  svg.querySelectorAll('rect.hit').forEach(r => {
    r.addEventListener('mousemove', ev => {
      const i = +r.dataset.i;
      const rng = i === 0 ? 'reached the road' : ((i-1)*B) + '-' + (i*B) + ' m short of ' +
        (D.dist_ref === 'crossing' ? 'the road crossing' : 'the road');
      tip.innerHTML = '<b>' + rng + '</b><br>' + counts.map(o =>
        o.s.name + ': ' + o.c[i] + ' scenario' + (o.c[i] === 1 ? '' : 's') +
        ', weight ' + o.w[i].toFixed(3) + (o.c[i] && o.c[i] <= 4 ? ' (' + o.ids[i].join(', ') + ')' : '')).join('<br>');
      const host = document.getElementById('ihist').getBoundingClientRect();
      tip.style.display = 'block';
      tip.style.left = Math.min(ev.clientX - host.left + 12, host.width - tip.offsetWidth - 4) + 'px';
      tip.style.top = (ev.clientY - host.top - 10) + 'px';
      r.setAttribute('fill', 'rgba(0,0,0,0.06)');
    });
    r.addEventListener('mouseleave', () => { tip.style.display = 'none'; r.setAttribute('fill', 'transparent'); });
  });
})();
"""


def render_parts(payload):
    # The pieces cycle_products drops into REPORT_TEMPLATE
    # @param payload: dict from build_payload, or None
    # @returns: dict of strings (empty when there is no payload)
    if payload is None:
        return {'imap_css': '', 'imap_html': '', 'ihist_html': '', 'imap_js': '',
                'imap_data': ''}
    return {
        'imap_css': CSS,
        'imap_html': HTML,
        'ihist_html': HIST_HTML.replace('{{BIN}}', f"{payload['hist_bin_m']:.0f}"),
        'imap_js': JS,
        'imap_data': payload_json(payload),
    }
