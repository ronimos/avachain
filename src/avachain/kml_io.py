"""Read every polygon in a KML file.

Each KML reader used to take the first <coordinates> element only, so a file
holding several polygons (the Seven Sisters: one start zone per path, five in
one file) silently became its first polygon. This module returns all of them:
every <Polygon> in the document, outer ring plus holes, in lon/lat.

A file with no <Polygon> element falls back to the old reading, the first
<coordinates> element as one ring, so a bare LinearRing KML still works.

Imports nothing from the project (like config.py), so any module can use it.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _ring(el) -> list[tuple[float, float]]:
    """lon/lat pairs of the first <coordinates> under el."""
    for c in el.iter():
        if _local(c.tag) == 'coordinates' and c.text:
            return [tuple(map(float, p.split(',')[:2]))
                    for p in c.text.strip().split() if ',' in p]
    return []


def kml_rings(kml_path) -> list[tuple[list, list[list]]]:
    """[(outer, [hole, ...]), ...] in lon/lat, one entry per polygon, in
    document order. Raises ValueError when the file has no coordinates."""
    root = ET.parse(str(kml_path)).getroot()
    out = []
    for poly in root.iter():
        if _local(poly.tag) != 'Polygon':
            continue
        outer, holes = [], []
        for child in poly:
            name = _local(child.tag)
            if name == 'outerBoundaryIs':
                outer = _ring(child)
            elif name == 'innerBoundaryIs':
                h = _ring(child)
                if len(h) >= 3:
                    holes.append(h)
        if len(outer) >= 3:
            out.append((outer, holes))
    if not out:
        ring = _ring(root)
        if len(ring) < 3:
            raise ValueError(f"No polygon coordinates found in {kml_path}")
        out.append((ring, []))
    return out


def kml_polygons(kml_path, target_crs=None) -> list:
    """Shapely Polygons, one per KML polygon, in lon/lat or reprojected to
    target_crs (an EPSG int or any pyproj CRS input)."""
    from shapely.geometry import Polygon
    rings = kml_rings(kml_path)
    if target_crs is not None:
        from pyproj import Transformer
        crs = f"EPSG:{target_crs}" if isinstance(target_crs, int) else target_crs
        t = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        tr = lambda pts: [t.transform(x, y) for x, y in pts]
    else:
        tr = lambda pts: pts
    return [Polygon(tr(o), [tr(h) for h in hs]) for o, hs in rings]


def kml_geometry(kml_path, target_crs=None):
    """The union of every polygon in the file (Polygon or MultiPolygon)."""
    from shapely.ops import unary_union
    polys = kml_polygons(kml_path, target_crs)
    return polys[0] if len(polys) == 1 else unary_union(polys)
