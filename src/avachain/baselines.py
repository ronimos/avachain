"""Reference arrest modes with no snowpack mechanics, for comparison with 'energy'.

slope_only:          connected region from the trigger where psi >= stauchwall and a weak layer exists.
stability_connected: connected region from the trigger where the skier stability index
                     (SNOWPACK's Sk38, column min_sk38) is below 1 and psi >= stauchwall.
                     The standalone calls this 'sk38_connected'; renamed because each
                     cluster's profile is simulated on its own slope, not on 38 deg.
"""
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from release_geometry import (
    STAUCHWALL_DEG,
    MAX_BFS_CLUSTERS,
    _cluster_graph,
    _domain_edge_pixels,
    _reached_polygon,
    cluster_pixel_index,
)

BASELINE_MODES = ('slope_only', 'stability_connected')
# Same threshold as analysis_pipeline's trigger gate MAX_SK38 (function-local
# there): Sk38 >= 1 is stable for skier triggering (Schweizer & Jamieson 2007).
STABILITY_MAX = 1.0


def _row(snap_features, cid):
    if snap_features is None or cid not in snap_features.index:
        return pd.Series(dtype=float)
    row = snap_features.loc[cid]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _num(row, col):
    return float(pd.to_numeric(row.get(col, np.nan), errors='coerce'))


def propagate_baseline(
        mode: str,
        trigger_cluster_id: int,
        cluster_map: np.ndarray,
        dem: np.ndarray,
        slope_grid: np.ndarray,
        transform,
        start_zone_mask: Optional[np.ndarray] = None,
        stauchwall_deg: float = STAUCHWALL_DEG,
        snap_features: 'Optional[pd.DataFrame]' = None,
        k_neighbours: int = 8,
        max_clusters: int = MAX_BFS_CLUSTERS,
        pixel_index: Optional[dict] = None,
        state_out: Optional[dict] = None):
    """Flood-fill on the BFS cluster graph with a per-cell admission test; returns (polygon, reached)."""
    if mode not in BASELINE_MODES:
        raise ValueError(f"unknown baseline mode {mode!r}")
    if pixel_index is None:
        pixel_index = cluster_pixel_index(cluster_map)
    if trigger_cluster_id not in pixel_index:
        return None, set()
    neighbours, centroids = _cluster_graph(pixel_index, transform, k_neighbours)
    edge_px = _domain_edge_pixels(cluster_map, dem)
    sk_col = ('min_sk38' if snap_features is not None and 'min_sk38' in snap_features
              else 'sk38_min')

    def _reason(cid):
        px = pixel_index[cid]
        if start_zone_mask is not None and not start_zone_mask[int(px[0].mean()), int(px[1].mean())]:
            return 'outside_start_zone'
        if float(slope_grid[px[0], px[1]].mean()) < stauchwall_deg:
            return 'stauchwall_slope'
        row = _row(snap_features, cid)
        if mode == 'slope_only':
            if not (np.isfinite(_num(row, 'wl_shear_strength'))
                    and np.isfinite(_num(row, 'slab_thickness'))):
                return 'no_weak_layer'
        elif not _num(row, sk_col) < STABILITY_MAX:     # NaN fails too
            return 'stability_stable'
        return None

    reached, rejected = {trigger_cluster_id}, {}
    queue = deque([trigger_cluster_id])
    while queue and len(reached) < max_clusters:
        c = queue.popleft()
        for n in neighbours.get(c, []):
            if n in reached or n in rejected or n not in centroids:
                continue
            reason = _reason(n)
            if reason is None:
                reached.add(n)
                queue.append(n)
            else:
                rejected[n] = reason
    cap_bound = bool(queue) and len(reached) >= max_clusters
    edge = {c for c in reached if edge_px[pixel_index[c]].any()}
    counts = pd.Series(list(rejected.values()), dtype=object).value_counts().to_dict()
    print(f"  {mode} region: {len(reached)} clusters ({len(edge)} on the domain edge); "
          f"boundary rejections {counts}{'  [CAP-BOUND]' if cap_bound else ''}")
    if state_out is not None:
        state_out.update(rejected=rejected, domain_edge=edge, cap_bound=cap_bound)
    return _reached_polygon(reached, cluster_map, transform,
                            min_clusters=1, min_area=0.0), reached
