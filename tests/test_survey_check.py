"""
tests/test_survey_check.py — the survey vertical-frame check (survey_check.py).

Snow height is snow DSM − ground DEM. The DEM series is NAD83(2011)
ellipsoidal; at Little Professor GEOID18 N = −12.42 m, so a survey with a
geoid applied reads ~12.4 m of snow everywhere, and nothing downstream would
stop it. These pin the verdicts on synthetic surveys and that the operational
script stops a FAIL at Step 0, marked FAILED, before any state changes.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import survey_check as sc                       # noqa: E402
import survey_markers as sm                     # noqa: E402

X0, Y0, RES, N = 424500.0, 4389200.0, 0.5, 200   # 100 x 100 m at 0.5 m, EPSG:6342
D1 = dt.date(2026, 11, 20)


def _kml(path: Path, x0, y0, x1, y1) -> Path:
    t = Transformer.from_crs("EPSG:6342", "EPSG:4326", always_xy=True)
    pts = [t.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    coords = " ".join(f"{lon},{lat},0" for lon, lat in pts)
    path.write_text('<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                    f"<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}"
                    "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></kml>")
    return path


def _hs(dz: float = 0.0, seed: int = 0) -> np.ndarray:
    """A plausible survey: a quarter of the slope scoured near bare, the rest
    0.5-2 m, like 2025-26 (p10 -0.57..+0.68 m)."""
    rng = np.random.default_rng(seed)
    hs = rng.uniform(0.5, 2.0, (N, N))
    hs[: N // 4] = rng.normal(0.0, 0.15, (N // 4, N))
    return (hs + dz).astype("float32")


def _survey(path: Path, hs: np.ndarray, nodata=-9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=hs.shape[0], width=hs.shape[1],
                       count=1, dtype="float32", crs="EPSG:6342", nodata=nodata,
                       transform=from_origin(X0, Y0, RES, RES)) as d:
        d.write(np.where(np.isnan(hs), nodata, hs).astype("float32"), 1)
    return path


@pytest.fixture
def boundary(tmp_path):
    return _kml(tmp_path / "boundary.kml", X0 + 10, Y0 - 10, X0 + 90, Y0 - 90)


@pytest.mark.parametrize("dz, verdict", [
    (0.0, "PASS"),
    (sc.GEOID_N_M * -1, "FAIL"),     # geoid applied to the snow DSM only
    (sc.GEOID_N_M, "FAIL"),          # geoid applied to the ground only
    (2.0, "WARN"),                   # no near-bare ground
])
def test_verdicts(tmp_path, boundary, dz, verdict):
    st = sc.survey_stats(_survey(tmp_path / "s.tif", _hs(dz)), boundary)
    assert st["cover"] == pytest.approx(1.0)
    v, msgs = sc.judge(st)
    assert v == verdict, st
    if verdict == "FAIL":
        assert "vertical frames" in msgs[0]


def test_negative_tail_does_not_trip(tmp_path, boundary):
    # Last season's rasters have minima of -11 to -18 m (edges, trees); a few
    # such cells must not look like a frame mix.
    hs = _hs()
    hs[100:104, 50:150] = -15.0          # 2 % of the slope
    st = sc.survey_stats(_survey(tmp_path / "s.tif", hs), boundary)
    assert sc.judge(st)[0] == "PASS"


def test_survey_missing_most_of_the_slope_fails(tmp_path, boundary):
    hs = _hs()
    hs[:, : int(N * 0.6)] = np.nan
    st = sc.survey_stats(_survey(tmp_path / "s.tif", hs), boundary)
    v, msgs = sc.judge(st)
    assert v == "FAIL" and "covers" in msgs[0]


def test_cli_exit_codes(tmp_path, boundary):
    toml = _toml(tmp_path, boundary)
    for dz, rc in ((0.0, 0), (2.0, 3), (12.42, 1)):
        s = _survey(tmp_path / f"s{dz}.tif", _hs(dz))
        assert sc.main(["--toml", str(toml), "--survey", str(s)]) == rc


def _toml(tmp_path: Path, boundary: Path) -> Path:
    p = tmp_path / "slope_config.toml"
    p.write_text(f'[slope]\nname = "Professor_PTC"\n\n'
                 f'[paths]\nproject_dir = "{tmp_path}"\nsurvey_dir = "surveys"\n'
                 f'output_dir = "out"\nslope_dir = "work"\nboundary_kml = "{boundary}"\n\n'
                 f'[surveys]\nmarkers = true\n')
    return p


class TestOperationalStep0:
    """The rendered run_<slope>.sh: claim, then Step 0 as rendered, then a
    stand-in for Step 1 that leaves a trace if reached."""

    def _script(self, tmp_path, boundary) -> Path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("install_mod_check", REPO / "install.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with open(REPO / "slopes" / "little_prof" / "slope_config.toml", "rb") as fh:
            sh = mod._render_run_operational(tomllib.load(fh))
        head = sh[:sh.index("exec > >(tee")]
        step0 = sh[sh.index("# Step 0:"):sh.index('echo ">>> Step 1: Resample"')]
        head = head.replace('PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                            f'PROJECT_DIR="{REPO}"')
        head = "\n".join(f"TOML={_toml(tmp_path, boundary)}" if l.startswith("TOML=") else l
                         for l in head.splitlines())
        tail = ('SURVEY_DATE=$CLAIMED\nALERT="echo ALERT-SENT"\n' + step0
                + f'touch {tmp_path}/step1_reached\n')
        p = tmp_path / "run.sh"
        p.write_text(head + "\n" + tail)
        return p

    def _deliver(self, tmp_path, dz):
        tif = _survey(tmp_path / "surveys" / f"{D1:%y%m%d}_Professor_PTC_snowHeight.tif",
                      _hs(dz))
        sm.marker_path(tif, "READY").write_text("")
        return tif

    def _run(self, script):
        return subprocess.run(["bash", str(script)], capture_output=True, text=True)

    def test_frame_mix_fails_at_step0_before_any_state_change(self, tmp_path, boundary):
        tif = self._deliver(tmp_path, 12.42)
        r = self._run(self._script(tmp_path, boundary))
        assert r.returncode == 1, r.stdout + r.stderr
        assert not (tmp_path / "step1_reached").exists()
        assert sm.state_of(tif) == "FAILED"
        rec = json.loads(sm.marker_path(tif, "FAILED").read_text())
        assert rec["failed_step"] == "Step 0: Survey vertical-frame check"

    def test_warn_alerts_and_continues(self, tmp_path, boundary):
        tif = self._deliver(tmp_path, 2.0)
        r = self._run(self._script(tmp_path, boundary))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "ALERT-SENT" in r.stdout and (tmp_path / "step1_reached").exists()
        assert sm.state_of(tif) == "DONE"

    def test_good_survey_passes_quietly(self, tmp_path, boundary):
        tif = self._deliver(tmp_path, 0.0)
        r = self._run(self._script(tmp_path, boundary))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "ALERT-SENT" not in r.stdout and (tmp_path / "step1_reached").exists()
        assert sm.state_of(tif) == "DONE"
        rep = json.loads(next((tmp_path / "out").rglob(f"survey_check_{D1}.json")).read_text())
        assert rep["verdict"] == "PASS"
