"""The 0.8 x HS release-depth fallback is in metres (HS in the store is m).

depth_from_snowpack divided HS by 100 until 2026-11-04, so every cluster
without a slab thickness got a release depth 100x too small.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
from release_geometry import depth_from_snowpack  # noqa: E402


def test_fallback_is_0p8_hs_in_metres():
    ds = xr.Dataset({"HS": (("location", "time"), np.array([[1.25]]))},
                    coords={"location": ["cluster_0001"],
                            "time": pd.to_datetime(["2026-12-02T12:00"]).values})
    d = depth_from_snowpack(np.ones((2, 2), dtype=int), ds, "2026-12-02")
    assert np.allclose(d, 1.0)


def test_slab_thickness_still_wins():
    ds = xr.Dataset({"HS": (("location", "time"), np.array([[1.25]]))},
                    coords={"location": ["cluster_0001"],
                            "time": pd.to_datetime(["2026-12-02T12:00"]).values})
    feats = pd.DataFrame({"slab_thickness": [0.6]}, index=[1])
    d = depth_from_snowpack(np.ones((2, 2), dtype=int), ds, "2026-12-02", snap_features=feats)
    assert np.allclose(d, 0.6)
