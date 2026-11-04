"""One snapshot time for every read of a run (Ron, 2026-11-04: 18 UTC).

Before, one `scenarios` run read features at 00:00 (analyze), release depth
at 12:00 and scour at 18:00 of the snapshot date. [scenarios]
snapshot_hour_utc (or --snapshot-ts) now sets one time for all of them;
without it the historical hours stay, which the Jan 18 validation relies on
(checked on real data: analyze CSVs and all 458 scenario files identical).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "avachain"))
import analysis_pipeline as ap  # noqa: E402
from release_geometry import depth_from_snowpack  # noqa: E402
from config import ProjectConfig  # noqa: E402


def _args(**kw):
    return SimpleNamespace(**{"snapshot_date": "2026-12-02", "snapshot_ts": None, **kw})


def test_hour_from_the_toml():
    a = _args()
    ap.resolve_snapshot_ts(SimpleNamespace(snapshot_hour_utc=18), a)
    assert a.snapshot_ts == "2026-12-02T18:00"


def test_no_hour_keeps_the_historical_reads():
    a = _args()
    ap.resolve_snapshot_ts(SimpleNamespace(snapshot_hour_utc=None), a)
    assert a.snapshot_ts is None


def test_an_explicit_time_wins():
    a = _args(snapshot_ts="2026-12-04T06:00")          # a clamped T+2
    ap.resolve_snapshot_ts(SimpleNamespace(snapshot_hour_utc=18), a)
    assert a.snapshot_ts == "2026-12-04T06:00"


def test_operational_toml_sets_18():
    cfg = ProjectConfig.from_toml(REPO / "slopes" / "Professor_PTC" / "slope_config.toml")
    assert cfg.snapshot_hour_utc == 18
    assert ProjectConfig.from_toml(REPO / "slopes" / "little_prof" / "slope_config.toml") \
        .snapshot_hour_utc is None


def test_release_depth_is_read_at_the_snapshot_time():
    times = pd.to_datetime(["2026-12-02T12:00", "2026-12-02T18:00"]).values
    ds = xr.Dataset({"HS": (("location", "time"), np.array([[1.0, 2.0]]))},
                    coords={"location": ["cluster_0001"], "time": times})
    cm = np.ones((2, 2), dtype=int)
    old = depth_from_snowpack(cm, ds, "2026-12-02")                       # 12:00
    new = depth_from_snowpack(cm, ds, "2026-12-02", snapshot_ts="2026-12-02T18:00")
    # HS is in metres in the store; the fallback is 0.8 x HS, in metres
    # (it divided by 100 until 2026-11-04).
    assert np.allclose(old, 0.8) and np.allclose(new, 1.6)
