"""Every KML reader takes every polygon in the file (kml_io, 2026-11-17).

The four readers (snowpack_io.kml_to_mask, clustering.build_domain_mask,
avalanche.load_slope_mask, fetch_avalanche_obs) each read the first
<coordinates> element only, so the Seven Sisters start zone, five disjoint
polygons in one KML, would silently have been Sister 1 alone. A single-polygon
file (every Little Professor KML) must give the same mask as before, bit for
bit: the season lock allows only output-identical operational changes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio.features
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "avachain"))
import kml_io                                   # noqa: E402
from snowpack_io import kml_to_mask             # noqa: E402
from clustering import build_domain_mask, parse_kml_polygon   # noqa: E402
from avalanche import load_slope_mask           # noqa: E402

EPSG = 6342
X0, Y0 = 423400.0, 4392400.0                    # NW corner of a 100 x 60 m, 1 m grid
SHAPE = (60, 100)
TF = Affine(1.0, 0, X0, 0, -1.0, Y0)
TO_LL = Transformer.from_crs(EPSG, 4326, always_xy=True)
TO_UTM = Transformer.from_crs(4326, EPSG, always_xy=True)


def _ll(ring):
    return " ".join(f"{lon:.10f},{lat:.10f},0" for lon, lat in (TO_LL.transform(x, y) for x, y in ring))


def _box(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]


A = _box(X0 + 10, Y0 - 50, X0 + 30, Y0 - 10)    # west path
B = _box(X0 + 60, Y0 - 50, X0 + 90, Y0 - 10)    # east path
HOLE = _box(X0 + 70, Y0 - 40, X0 + 80, Y0 - 20)


def _polygon_xml(outer, holes=()):
    inner = "".join(f"<innerBoundaryIs><LinearRing><coordinates>{_ll(h)}</coordinates>"
                    f"</LinearRing></innerBoundaryIs>" for h in holes)
    return (f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{_ll(outer)}</coordinates>"
            f"</LinearRing></outerBoundaryIs>{inner}</Polygon>")


def _kml(body):
    return ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
            f"<Document>{body}</Document></kml>")


@pytest.fixture
def two_placemarks(tmp_path):
    p = tmp_path / "two.kml"
    p.write_text(_kml(f"<Placemark>{_polygon_xml(A)}</Placemark><Placemark>{_polygon_xml(B)}</Placemark>"))
    return p


@pytest.fixture
def multigeometry_with_points(tmp_path):
    """The Google Earth layout of 'Seven Sisters.kml': one placemark, a
    MultiGeometry holding polygons and trigger points."""
    lon, lat = TO_LL.transform(X0 + 45, Y0 - 30)
    p = tmp_path / "mg.kml"
    p.write_text(_kml(f"<Placemark><MultiGeometry>{_polygon_xml(A)}"
                      f"<Point><coordinates>{lon},{lat},0</coordinates></Point>"
                      f"{_polygon_xml(B, [HOLE])}</MultiGeometry></Placemark>"))
    return p


def _expected(polys, holes=()):
    geoms = [Polygon(p, list(holes) if i == len(polys) - 1 else []) for i, p in enumerate(polys)]
    return rasterio.features.geometry_mask([g.__geo_interface__ for g in geoms],
                                           out_shape=SHAPE, transform=TF, invert=True)


def test_kml_rings_reads_every_polygon_and_skips_points(multigeometry_with_points):
    rings = kml_io.kml_rings(multigeometry_with_points)
    assert len(rings) == 2
    assert [len(h) for _, h in rings] == [0, 1]


def test_kml_to_mask_takes_every_polygon(two_placemarks, multigeometry_with_points):
    m = kml_to_mask(two_placemarks, SHAPE, TF)
    assert m[30, 20] and m[30, 75]                      # inside A and inside B
    assert not m[30, 45]                                # the gap between them
    m2 = kml_to_mask(multigeometry_with_points, SHAPE, TF)
    assert m2[30, 20] and m2[15, 75] and not m2[30, 75]  # B's hole is excluded
    assert m2.sum() == _expected([A, B], [HOLE]).sum()


def test_build_domain_mask_takes_every_polygon(two_placemarks, multigeometry_with_points):
    dem = np.full(SHAPE, 3600.0)
    m = build_domain_mask(dem, TF, f"EPSG:{EPSG}", kml_path=str(two_placemarks), min_slope_deg=0.0)
    assert m[30, 20] and m[30, 75] and not m[30, 45]
    m2 = build_domain_mask(dem, TF, f"EPSG:{EPSG}", kml_path=str(multigeometry_with_points), min_slope_deg=0.0)
    assert m2[30, 20] and m2[15, 75] and not m2[30, 75]


def test_load_slope_mask_takes_every_polygon(two_placemarks, multigeometry_with_points):
    m = load_slope_mask(two_placemarks, SHAPE, TF)
    assert m[30, 20] and m[30, 75] and not m[30, 45]
    assert not m.all()                                  # not the full-domain fallback
    m2 = load_slope_mask(multigeometry_with_points, SHAPE, TF)
    assert m2[30, 20] and not m2[30, 75]


def test_parse_kml_polygon_refuses_several_polygons(two_placemarks):
    with pytest.raises(ValueError, match="several polygons"):
        parse_kml_polygon(two_placemarks)


def test_kml_geometry_is_the_union(two_placemarks):
    g = kml_io.kml_geometry(two_placemarks, EPSG)
    assert g.geom_type == "MultiPolygon"
    assert g.area == pytest.approx(20 * 40 + 30 * 40, rel=1e-3)


# --- single-polygon files: identical to the pre-2026-11-17 readers ----------

KMLS = sorted((ROOT / "data" / "boundaries").glob("*.kml"))


def _old_lonlat(kml_path):
    from xml.etree import ElementTree as ET
    tree = ET.parse(str(kml_path))
    ns = '{http://www.opengis.net/kml/2.2}'
    ct = tree.find('.//coordinates') or tree.find(f'.//{ns}coordinates')
    return [tuple(map(float, p.split(',')[:2])) for p in ct.text.strip().split() if ',' in p]


def _grid_around(kml_path, pad=20.0):
    xs, ys = zip(*(TO_UTM.transform(*p) for p in _old_lonlat(kml_path)))
    x0, y1 = np.floor(min(xs)) - pad, np.ceil(max(ys)) + pad
    shape = (int(max(ys) - min(ys) + 2 * pad) + 1, int(max(xs) - min(xs) + 2 * pad) + 1)
    return shape, Affine(1.0, 0, x0, 0, -1.0, y1)


@pytest.mark.parametrize("kml", KMLS, ids=lambda p: p.name)
def test_single_polygon_masks_unchanged(kml):
    assert len(kml_io.kml_rings(kml)) == 1
    shape, tf = _grid_around(kml)
    old_poly = Polygon([TO_UTM.transform(*p) for p in _old_lonlat(kml)])
    old = rasterio.features.geometry_mask([old_poly.__geo_interface__], out_shape=shape,
                                          transform=tf, invert=True)
    assert np.array_equal(kml_to_mask(kml, shape, tf), old)
    assert np.array_equal(load_slope_mask(kml, shape, tf), old)

    from matplotlib.path import Path as MplPath
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.column_stack([(tf[2] + cols * tf[0]).ravel(), (tf[5] + rows * tf[4]).ravel()])
    old_dom = MplPath([TO_UTM.transform(*p) for p in _old_lonlat(kml)]).contains_points(pts).reshape(shape)
    dom = build_domain_mask(np.full(shape, 3600.0), tf, f"EPSG:{EPSG}", kml_path=str(kml), min_slope_deg=0.0)
    assert np.array_equal(dom, old_dom)
    assert parse_kml_polygon(kml) == _old_lonlat(kml)
