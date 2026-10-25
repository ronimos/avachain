"""Frozen Jan 18 2026 inputs for release-geometry regression tests.

tests/data/jan18/ holds copies of outputs/little_prof/ as of the 2026-10-23
rebuild (commit 2f79a8e): DEM, cluster map, and the two feature CSVs. The live
outputs move with every rebuild; this copy does not. The observed crown and the
start-zone KML are read from data/boundaries/, where they are tracked.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src' / 'avachain'))

DATA = REPO / 'tests' / 'data' / 'jan18'
OBSERVED = REPO / 'data' / 'boundaries' / 'avalanche_release_area_20260118.geojson'
START_ZONE_KML = REPO / 'data' / 'boundaries' / 'Litte_prof_start_zone.kml'


@lru_cache(maxsize=1)
def load():
    """dict with dem, transform, cluster_map, start_zone_mask, snap_features, meloche_df, observed."""
    import json
    import rasterio
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from analysis_pipeline import kml_to_mask

    with rasterio.open(DATA / 'dem_1m.tif') as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        transform = src.transform
    feats = pd.read_csv(DATA / 'all_start_zone_features_2026-01-18.csv.gz', index_col=0)
    mel = pd.read_csv(DATA / 'meloche_features_all_2026-01-18.csv.gz', index_col=0)
    gj = json.loads(OBSERVED.read_text())
    observed = unary_union([shape(f['geometry']) for f in gj['features']])
    return dict(
        dem=dem, transform=transform,
        cluster_map=np.load(DATA / 'cluster_map.npy'),
        start_zone_mask=kml_to_mask(START_ZONE_KML, dem.shape, transform),
        snap_features=feats[~feats.index.duplicated()],
        meloche_df=mel[~mel.index.duplicated()],
        observed=observed)


def polygon_for(trigger, A_ca=None, size_factor=1.0, state_out=None):
    """Release polygon for one trigger through make_release_polygon_2d, as step_scenarios calls it."""
    from release_geometry import make_release_polygon_2d
    d = load()
    if A_ca is None:
        v = float(d['meloche_df'].loc[trigger, 'A_ca_brittle'])
        A_ca = 50.0 if np.isnan(v) else min(v, 300.0)
    return make_release_polygon_2d(
        trigger_cluster_id=trigger, A_ca=A_ca, meloche_df=d['meloche_df'],
        cluster_map=d['cluster_map'], dem=d['dem'], transform=d['transform'],
        start_zone_mask=d['start_zone_mask'], snap_features=d['snap_features'],
        size_factor=size_factor, state_out=state_out)


def iou(poly):
    obs = load()['observed']
    if poly is None or poly.is_empty:
        return 0.0
    return poly.intersection(obs).area / poly.union(obs).area
