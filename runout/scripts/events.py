"""
Observed Events

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Reads observed avalanche event records, one folder per event under
    data/static/<slug>/observed/<YYYYMMDD>/. The record (event.yaml) names
    any of a debris polygon, an observed release polygon, a toe point, and
    per reference point outcomes, plus the slab that released. A record
    with only point outcomes is enough to score against.
"""

# --------- Load Libraries --------
import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

from . import profiles as prof


POINT_OUTCOMES = {
    'reached': True, 'yes': True, 'true': True, 'hit': True,
    'not_reached': False, 'no': False, 'false': False, 'missed': False,
    'short': False,
}


class EventError(ValueError):
    pass


def observed_root(avachain_root, slug):
    # Folder holding a path's event records
    # @param avachain_root: Path
    # @param slug: path slug
    # @returns: Path
    return Path(avachain_root) / 'data' / 'static' / slug / 'observed'


def list_events(avachain_root, slug):
    # Every event date with an event.yaml for a path
    # @param avachain_root: Path
    # @param slug: path slug
    # @returns: sorted list of YYYYMMDD strings
    root = observed_root(avachain_root, slug)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / 'event.yaml').exists())


def _read_geom(folder, value, crs):
    # Load a geometry reference from the record
    # @param folder: Path of the event folder
    # @param value: file name, path, or [x, y] list
    # @param crs: CRS to return the geometry in
    # @returns: shapely geometry or None
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return Point(float(value[0]), float(value[1]))
    p = Path(str(value))
    if not p.is_absolute():
        p = folder / p
    if not p.exists():
        raise EventError(f"geometry file not found: {p}")
    gdf = gpd.read_file(p)
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        raise EventError(f"no geometry in {p}")
    return unary_union(geoms).buffer(0) if geoms[0].geom_type.endswith('Polygon') \
        else geoms[0]


def load_event(avachain_root, slug, date, crs=None):
    # Read one event record and its geometries
    # @param avachain_root: Path
    # @param slug: path slug
    # @param date: YYYYMMDD
    # @param crs: CRS to load geometries in (the DEM's)
    # @returns: dict with date, path, folder, debris, release, toe, points
    #          ({name: bool}), slab, obs_types (list of 'polygon', 'toe',
    #          'points' present), source, notes
    # @raises EventError: if the record is missing or malformed
    folder = observed_root(avachain_root, slug) / date
    rec_path = folder / 'event.yaml'
    if not rec_path.exists():
        raise EventError(f"no event record at {rec_path}")
    rec = prof.load_mapping_file(rec_path)

    debris = _read_geom(folder, rec.get('debris_polygon'), crs)
    release = _read_geom(folder, rec.get('release_polygon'), crs)
    toe = _read_geom(folder, rec.get('toe_point'), crs)
    if toe is not None and toe.geom_type != 'Point':
        toe = toe.centroid

    points = {}
    for name, outcome in (rec.get('points') or {}).items():
        key = str(outcome).strip().lower()
        if key not in POINT_OUTCOMES:
            raise EventError(f"{rec_path}: point {name!r} outcome {outcome!r}; "
                             f"use reached or not_reached")
        points[str(name)] = POINT_OUTCOMES[key]

    obs_types = []
    if debris is not None:
        obs_types.append('polygon')
    if toe is not None:
        obs_types.append('toe')
    if points:
        obs_types.append('points')
    if not obs_types:
        raise EventError(f"{rec_path}: no observation (debris_polygon, "
                         f"toe_point or points)")

    return {
        'date': str(rec.get('date') or date),
        'path': rec.get('path') or slug,
        'folder': folder,
        'record': rec,
        'debris': debris,
        'release': release,
        'toe': toe,
        'points': points,
        'slab': rec.get('slab') or {},
        'obs_types': obs_types,
        'source': rec.get('source'),
        'notes': rec.get('notes'),
    }
