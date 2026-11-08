"""survey_check.py — is a new snow-height survey in the ground DEM's vertical frame?

    python src/avachain/survey_check.py --toml T [--survey-date YYYY-MM-DD]
                                        [--survey FILE] [--json out.json]

Ron, 2026-11-07. Snow height is snow DSM − ground DEM, so both must share a
vertical frame. The DEM series is NAD83(2011) ellipsoidal heights with no
geoid (docs/deployment.md §2.1). At Little Professor GEOID18 N = −12.42 m, so
a snow DSM with a geoid applied, differenced against the ellipsoidal ground,
reads about +12.4 m of snow everywhere (−12.4 m the other way round). Nothing
downstream would stop it: SNOWPACK takes HS as given.

The check reads the survey inside the slope boundary and takes the shallow
end of the distribution, p10, as its bare-ground level. Surveys don't reach
the road, and their minima are edge and tree artefacts (−11 to −18 m on last
season's rasters), so neither can serve as the reference. On Little
Professor some wind-scoured ground is always near bare. Over the 23 surveys
of 2025-26, p10 ran from −0.57 to +0.68 m and the median from −0.06 to
+1.43 m.

  FAIL  |p10| > FRAME_FAIL_M (6 m, half the geoid separation): the survey
        and the DEM are in different frames. The run stops before any state
        changes and the survey ends FAILED with an alert.
  FAIL  fewer than MIN_BOUNDARY_COVER of the boundary cells have data, so
        the level cannot be read.
  WARN  |p10| > SHALLOW_WARN_M (1.5 m): no near-bare ground. This is possible
        after a slope-wide storm, but it is also what a partial datum shift
        (a different base or control) looks like. The run continues and
        alerts.

The thresholds are physical (the geoid separation) or set with a wide margin
around last season's range. They are not fitted to any event. Exit codes:
0 PASS, 3 WARN, 1 FAIL, 2 cannot run (no survey, unreadable file).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ProjectConfig  # noqa: E402

GEOID_N_M = -12.42           # NGS GEOID18 at Little Professor (2026-11-07)
FRAME_FAIL_M = 6.0           # half of |GEOID_N_M|
SHALLOW_WARN_M = 1.5
MIN_BOUNDARY_COVER = 0.5
SHALLOW_PCT = 10
READ_RES_M = 1.0             # surveys are ~0.05 m; 1 m is plenty for a level

EXIT = {"PASS": 0, "WARN": 3, "FAIL": 1}


def survey_stats(survey: Path, boundary_kml: Path) -> dict:
    """p10/p50 of HS (m) inside the boundary, read at ~READ_RES_M."""
    from snowpack_io import kml_to_mask
    with rasterio.open(survey) as s:
        f = max(1, int(round(READ_RES_M / abs(s.res[0]))))
        shape = (max(1, s.height // f), max(1, s.width // f))
        a = s.read(1, out_shape=shape, resampling=Resampling.average,
                   masked=True).astype("float64").filled(np.nan)
        tf = s.transform * s.transform.scale(s.width / shape[1], s.height / shape[0])
    inside = kml_to_mask(boundary_kml, shape, tf)
    v = a[inside & np.isfinite(a)]
    out = {"survey": str(survey), "boundary_cells": int(inside.sum()),
           "cells": int(v.size),
           "cover": float(v.size / inside.sum()) if inside.sum() else 0.0}
    if v.size:
        p10, p50 = np.percentile(v, [SHALLOW_PCT, 50])
        out.update({"p10_m": round(float(p10), 3), "p50_m": round(float(p50), 3)})
    return out


def judge(st: dict) -> tuple[str, list[str]]:
    msgs = []
    if st["cover"] < MIN_BOUNDARY_COVER:
        return "FAIL", [f"survey covers {st['cover']:.0%} of the slope boundary "
                        f"(< {MIN_BOUNDARY_COVER:.0%}); cannot read its level"]
    p10 = st["p10_m"]
    if abs(p10) > FRAME_FAIL_M:
        return "FAIL", [
            f"shallowest 10% of the slope reads {p10:+.2f} m of snow (median "
            f"{st['p50_m']:+.2f} m). A frame mix reads ~{-GEOID_N_M:.1f} m here: the snow "
            "DSM and the ground DEM are probably in different vertical frames (geoid vs "
            "ellipsoid). Ask the producer to reprocess it in NAD83(2011) ellipsoidal "
            "heights against the current ground DEM (docs/deployment.md 2.1)."]
    if abs(p10) > SHALLOW_WARN_M:
        msgs.append(f"no near-bare ground: shallowest 10% reads {p10:+.2f} m (2025-26 range "
                    "-0.57..+0.68 m). Possible after a slope-wide storm, or a datum shift "
                    "from a processing change; check before trusting it.")
        return "WARN", msgs
    return "PASS", msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--toml", required=True, type=Path)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--survey-date", type=dt.date.fromisoformat,
                   help="default: the latest visible survey")
    g.add_argument("--survey", type=Path, help="a snow-height raster, by path")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    cfg = ProjectConfig.from_toml(a.toml)
    survey = a.survey
    if survey is None:
        from pipeline_io import discover_surveys
        surveys = discover_surveys(cfg)
        if a.survey_date is not None:
            surveys = [s for s in surveys if s[0] == a.survey_date]
        if not surveys:
            print("survey_check: no matching survey", file=sys.stderr)
            return 2
        survey = surveys[-1][1]
    try:
        st = survey_stats(Path(survey), cfg.boundary_kml)
    except Exception as e:  # unreadable raster or KML
        print(f"survey_check: cannot read {survey}: {e}", file=sys.stderr)
        return 2
    verdict, msgs = judge(st)
    st.update({"verdict": verdict, "messages": msgs,
               "thresholds": {"frame_fail_m": FRAME_FAIL_M, "shallow_warn_m": SHALLOW_WARN_M,
                              "min_boundary_cover": MIN_BOUNDARY_COVER}})
    print(f"Survey frame check: {Path(survey).name}")
    if "p10_m" in st:
        print(f"  inside boundary: {st['cover']:.0%} covered, p10 {st['p10_m']:+.2f} m, "
              f"median {st['p50_m']:+.2f} m")
    for m in msgs:
        print(f"  {verdict}: {m}")
    print(f"  RESULT {verdict}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(st, indent=2))
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
