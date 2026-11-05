"""dem_check.py on synthetic rasters: CRS, coverage, road and offset checks.

The new GCP DEM has not arrived (2026-11-05), so each check is pinned here on
fixtures where the answer is known by construction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from config import RASTER_EPSG  # noqa: E402
import dem_check  # noqa: E402

X0, Y0 = 424500.0, 4389200.0       # top-left, EPSG:6342
N = 100                            # 100 x 100 cells of 1 m
COMPOUND = CRS.from_user_input(f"EPSG:{RASTER_EPSG}+5703")   # + NAVD88 height


def _plane(n=N, dz=0.0, slope_deg=30.0, ncols=N):
    """Planar surface falling to the south (rows), elevation 3500 m at the top."""
    rows = np.arange(n, dtype="float32")[:, None] * np.ones((1, ncols), "float32")
    return 3500.0 - rows * np.tan(np.radians(slope_deg)) + dz


def _write(path, z, crs=COMPOUND, res=1.0, x0=X0, y0=Y0, nodata=-10000.0):
    z = np.where(np.isnan(z), nodata, z).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1],
                       count=1, dtype="float32", crs=crs.to_wkt(), nodata=nodata,
                       transform=from_origin(x0, y0, res, res)) as dst:
        dst.write(z, 1)
    return path


def _kml(path, x0, y0, x1, y1):
    """A rectangle (EPSG:6342 corners) written as a lon/lat KML polygon."""
    t = Transformer.from_crs(f"EPSG:{RASTER_EPSG}", "EPSG:4326", always_xy=True)
    pts = [t.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    coords = " ".join(f"{lon},{lat},0" for lon, lat in pts)
    path.write_text('<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                    f"<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}"
                    "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></kml>")
    return path


@pytest.fixture
def kmls(tmp_path):
    # boundary: the inner 80 x 80 m; start zone: a 20 x 20 m block near the top
    b = _kml(tmp_path / "boundary.kml", X0 + 10, Y0 - 10, X0 + 90, Y0 - 90)
    s = _kml(tmp_path / "start.kml", X0 + 40, Y0 - 10, X0 + 60, Y0 - 30)
    return b, s


def test_compound_6342_navd88_passes_with_no_warning(tmp_path, kmls):
    dem = _write(tmp_path / "new.tif", _plane())
    rep = dem_check.check(dem, *kmls, with_sha=False)
    assert rep["dem"]["horizontal_epsg"] == RASTER_EPSG
    assert "NAVD88" in rep["dem"]["vertical"]
    assert rep["fail"] == [] and rep["warn"] == []
    assert rep["coverage"] == {"boundary": 1.0, "start_zone": 1.0}


def test_other_horizontal_crs_fails(tmp_path, kmls):
    dem = _write(tmp_path / "wgs.tif", _plane(), crs=CRS.from_epsg(26913))  # NAD83 / UTM 13N
    rep = dem_check.check(dem, *kmls, with_sha=False)
    assert rep["fail"] and "EPSG:26913" in rep["fail"][0]


def test_missing_vertical_crs_warns_only(tmp_path, kmls):
    dem = _write(tmp_path / "h.tif", _plane(), crs=CRS.from_epsg(RASTER_EPSG))
    rep = dem_check.check(dem, *kmls, with_sha=False)
    assert rep["fail"] == [] and rep["warn"] == ["no vertical CRS (expected NAVD88)"]


def test_hole_in_start_zone_fails_coverage(tmp_path, kmls):
    z = _plane()
    z[15:25, 45:55] = np.nan          # 100 of the start zone's 400 cells
    dem = _write(tmp_path / "holey.tif", z)
    rep = dem_check.check(dem, *kmls, with_sha=False)
    assert rep["coverage"]["start_zone"] == pytest.approx(0.75, abs=0.03)
    # the hole is inside the boundary too: 100 of 6400 cells, 0.984 < 0.995
    assert rep["coverage"]["boundary"] == pytest.approx(1 - 100 / 6400, abs=0.005)
    assert {f.split(":")[0] for f in rep["fail"]} == {"start_zone", "boundary"}


def test_path_footprint_and_road(tmp_path, kmls):
    """The new DEM must cover the runout DEM's valid cells and the road inside them.

    The path DEM extends 50 m below the new one; the road runs across that
    extension. The new DEM is also 100 m wider than the path; a road segment
    there has data but is off the path, so it must not count."""
    import geopandas as gpd
    from shapely.geometry import LineString
    new = _write(tmp_path / "new.tif", _plane(ncols=200))
    path = _write(tmp_path / "path.tif", _plane(n=150))       # 50 m further south, same width
    yr = Y0 - 125                                              # road row, outside new
    road = tmp_path / "road.shp"
    gpd.GeoDataFrame(geometry=[LineString([(X0 + 5, yr), (X0 + 95, yr)]),
                               LineString([(X0 + 150, Y0 - 50), (X0 + 190, Y0 - 50)])],  # on new, off the path
                     crs=f"EPSG:{RASTER_EPSG}").to_file(road)
    rep = dem_check.check(new, *kmls, path_dem=path, road=road, with_sha=False)
    assert rep["coverage"]["path_footprint"] == pytest.approx(100 / 150)
    assert rep["coverage"]["road"] == 0.0
    assert 85 <= rep["road_points"] <= 95                      # the off-path line dropped
    assert {f.split(":")[0] for f in rep["fail"]} == {"path_footprint", "road"}

    full = _write(tmp_path / "full.tif", _plane(n=150, ncols=120))
    rep = dem_check.check(full, *kmls, path_dem=path, road=road, with_sha=False)
    assert rep["fail"] == [] and rep["coverage"]["road"] == 1.0


def test_offset_and_slope_against_an_old_dem(tmp_path, kmls):
    """A surface 1.1 m below an identical one: dz = -1.1 everywhere, no slope change.
    The old DEM is finer (0.25 m), as 251110 is, so this also covers averaging."""
    new = _write(tmp_path / "new.tif", _plane(dz=-1.1))
    old_z = 3500.0 - (np.arange(N * 4, dtype="float32")[:, None] + 0.5) * 0.25 \
        * np.tan(np.radians(30.0)) * np.ones((1, N * 4), "float32")
    old = _write(tmp_path / "old.tif", old_z + 0.5 * np.tan(np.radians(30.0)), res=0.25)
    rep = dem_check.check(new, *kmls, compare=[old], with_sha=False)
    s = rep["compare"]["old.tif"]["start_zone"]
    assert s["dz_median_m"] == pytest.approx(-1.1, abs=0.01)
    assert s["dz_p05_m"] == pytest.approx(-1.1, abs=0.01)
    assert s["slope_absdiff_median_deg"] == pytest.approx(0.0, abs=0.05)


def test_sha256_matches_file(tmp_path):
    import hashlib
    p = _write(tmp_path / "a.tif", _plane(n=10, ncols=10))
    assert dem_check.sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()
