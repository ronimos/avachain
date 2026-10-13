"""Guard tests for the project raster CRS.

Every reprojection target in the pipeline must agree on one horizontal CRS.
They did not before 2026-10-13: rasters are NAD83(2011)/UTM 13N (EPSG:6342)
while seven code sites declared WGS84/UTM 13N (EPSG:32613). pyproj treats the
two as a sub-millimetre null transform at this location, so the mismatch was
latent rather than visibly broken — which is exactly why it survived. These
tests fail if any of those sites drifts away from config.RASTER_EPSG again.
"""
import inspect
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from config import RASTER_EPSG      # noqa: E402
import snowpack_io                  # noqa: E402
import snowpack_analysis            # noqa: E402
import scenario_writer              # noqa: E402


def test_raster_epsg_is_nad83_2011_utm13n():
    """The rasters on disk are EPSG:6342, not 32613. Pin it."""
    assert RASTER_EPSG == 6342


@pytest.mark.parametrize("func, param", [
    (snowpack_io.load_boundaries,        "target_epsg"),
    (snowpack_io.kml_to_mask,            "target_epsg"),
    (snowpack_analysis.geojson_to_mask,  "dst_epsg"),
    (snowpack_analysis._reproject_lonlat_to_utm, "epsg"),
])
def test_reprojection_default_matches_raster_epsg(func, param):
    """Callers rely on these defaults; they must not diverge from config."""
    default = inspect.signature(func).parameters[param].default
    assert default == RASTER_EPSG, (
        f"{func.__name__}({param}=) defaults to {default}, "
        f"expected config.RASTER_EPSG={RASTER_EPSG}"
    )


def test_depth_prj_wkt_authority_matches_raster_epsg():
    """The .prj sidecar WKT is a hand-written literal that AvaFrame parses.

    It is deliberately not generated from RASTER_EPSG — the exact bytes go to
    an external consumer — so this is the only thing keeping it in step.
    """
    src = inspect.getsource(scenario_writer)
    codes = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]\]', src)
    assert codes, "no AUTHORITY[EPSG,...] found in the .prj WKT literal"
    for code in codes:
        assert int(code) == RASTER_EPSG, (
            f".prj WKT declares EPSG:{code}, expected {RASTER_EPSG}"
        )


def test_no_wgs84_utm13n_reprojection_targets_remain():
    """32613 must not reappear as a reprojection target in the package.

    Comments are allowed to mention it (one documents the input formats the
    GeoJSON CRS regex accepts), so only non-comment code is checked.
    """
    pkg = Path(__file__).resolve().parent.parent / "src" / "avachain"
    offenders = []
    for py in sorted(pkg.glob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "32613" in line.split("#")[0]:
                offenders.append(f"{py.name}:{lineno}: {line.strip()}")
    assert not offenders, "EPSG:32613 used as a CRS in:\n" + "\n".join(offenders)
