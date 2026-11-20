"""
test_forest_overlap.py - the resistance (forest) shapefile com1DFA gets has no
overlapping features.

geojson_to_shapefile(fill_holes=True) rebuilds each polygon from its exterior,
so a forest patch inside a canopy gap ends up inside the gap's filled polygon.
com1DFA refuses overlapping area features ("Features [...] are overlapping -
this is not allowed"), which failed every scenario of the first Sisters
release to reach com1DFA (2026-11-20 T_1/T_2). Such patches are now dropped:
the outer polygon already covers them.

    cd runout && .venv/bin/python -m unittest tests.test_forest_overlap -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import avalanche_helpers as h  # noqa: E402

CRS = "EPSG:6342"


def _overlaps(gdf):
    g = list(gdf.geometry)
    return sum(1 for i in range(len(g)) for j in range(i + 1, len(g))
               if g[i].intersection(g[j]).area > 0)


class ForestOverlap(unittest.TestCase):

    def _convert(self, geoms):
        tmp = Path(tempfile.mkdtemp())
        src = tmp / "forest.geojson"
        gpd.GeoDataFrame({'forest': [1] * len(geoms)}, geometry=geoms,
                         crs=CRS).to_file(src, driver="GeoJSON")
        out = h.geojson_to_shapefile(src, tmp / "RES" / "forest.shp",
                                     fill_holes=True, explode_multipart=True)
        return gpd.read_file(out)

    def test_patch_in_a_filled_gap_is_dropped(self):
        ring = Polygon(box(0, 0, 100, 100).exterior.coords,
                       [box(30, 30, 70, 70).exterior.coords])
        patch = box(40, 40, 60, 60)          # inside the gap, no overlap yet
        lone = box(200, 0, 220, 20)
        out = self._convert([ring, patch, lone])
        self.assertEqual(_overlaps(out), 0)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out.geometry.area.sum(), 100 * 100 + 20 * 20)
        self.assertIn('forest', out.columns)

    def test_nested_twice_keeps_only_the_outer(self):
        outer = Polygon(box(0, 0, 100, 100).exterior.coords,
                        [box(10, 10, 90, 90).exterior.coords])
        middle = Polygon(box(20, 20, 80, 80).exterior.coords,
                         [box(30, 30, 70, 70).exterior.coords])
        inner = box(40, 40, 60, 60)
        out = self._convert([inner, middle, outer])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.geometry.area.iloc[0], 100 * 100)

    def test_layer_without_nesting_is_unchanged(self):
        geoms = [box(0, 0, 10, 10), box(10, 0, 20, 10), box(50, 50, 60, 70)]
        out = self._convert(geoms)
        self.assertEqual(len(out), 3)
        self.assertEqual(_overlaps(out), 0)


if __name__ == '__main__':
    unittest.main()
