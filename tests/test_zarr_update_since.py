"""Incremental Zarr update after a SNOWPACK re-run from a restart point.

build_zarr_cache resumes by skipping locations already in the store and
appends only along location. So after the first build, no incremental run
(daily, survey or reinit) ever extended the store in time, and analyze read
the newest stored profile without complaint.

update_zarr_since(since) must leave the store exactly as a fresh build of the
same .pro files would: everything after `since` replaced, everything up to it
untouched, and steps the re-run no longer produces (e.g. yesterday's forecast
tail) gone.

The .pro files are synthetic: one real SNOWPACK block (tests/fixtures)
repeated at 6-hourly times, with a marker added to its temperature line
(0503) that identifies the cluster, the timestep and the run. That way a
stale or misplaced step is detectable.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from snowpack_io import build_zarr_cache, update_zarr_since   # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pro_header_block.pro"
T0 = datetime(2026, 2, 5, 12)
STEP = timedelta(hours=6)
MAXL = 120


def _split_fixture():
    text = FIXTURE.read_text()
    head, block = text.split("[DATA]\n", 1)
    return head + "[DATA]\n", block


def _write_pro(d: Path, cid: int, steps: list[tuple[int, int]]):
    """steps: (step index, run id). Temperature marker = cid*100 + run*10 + i."""
    head, block = _split_fixture()
    name = f"cluster_{cid:04d}"
    out = [head.replace("cluster_2859", name)]
    for i, run in steps:
        t = T0 + i * STEP
        lines = []
        for ln in block.splitlines():
            if ln.startswith("0500,"):
                ln = "0500," + t.strftime("%d.%m.%Y %H:%M:%S")
            elif ln.startswith("0503,"):
                parts = ln.split(",")
                mark = cid * 100 + run * 10 + i
                parts[2:] = [f"{float(v) + mark:.2f}" for v in parts[2:]]
                ln = ",".join(parts)
            lines.append(ln)
        out.append("\n".join(lines) + "\n")
    (d / f"{name}_{name}.pro").write_text("".join(out))


def _times(store):
    with xr.open_zarr(store) as z:
        return [str(t)[:16] for t in z["time"].values]


def _same_as_fresh_build(store, pro_dir, tmp_path):
    fresh = tmp_path / "fresh.zarr"
    build_zarr_cache(pro_dir, fresh, n_cpus=1, max_layers=MAXL, rebuild=True)
    a, b = xr.open_zarr(store).load(), xr.open_zarr(fresh).load()
    assert list(a["time"].values) == list(b["time"].values)
    assert list(a["location"].values) == list(b["location"].values)
    for v in list(b.data_vars) + ["z"]:
        np.testing.assert_array_equal(a[v].values, b[v].values, err_msg=v)


@pytest.fixture
def pro_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d


def _temp0(store, loc, i):
    with xr.open_zarr(store) as z:
        return float(z["temperature"].sel(location=loc).isel(time=i)
                     .dropna("layer").values[0])


def test_rerun_replaces_only_the_window_after_since(pro_dir, tmp_path):
    store = tmp_path / "s.zarr"
    for cid in (1, 2, 3):
        _write_pro(pro_dir, cid, [(i, 0) for i in range(4)])          # run 0
    build_zarr_cache(pro_dir, store, n_cpus=1, max_layers=MAXL, rebuild=True)
    before = [_temp0(store, "cluster_0002", i) for i in range(4)]

    # Re-run from step 1: SNOWPACK rewrites the .pro from there (run 1).
    for cid in (1, 2, 3):
        _write_pro(pro_dir, cid, [(0, 0), (1, 0)] + [(i, 1) for i in range(2, 6)])
    n = update_zarr_since(pro_dir, store, T0 + STEP, batch_size=2, n_cpus=1,
                          max_layers=MAXL)

    assert n == 4
    after = [_temp0(store, "cluster_0002", i) for i in range(6)]
    assert after[:2] == before[:2]                       # <= since untouched
    assert after[2] - before[2] == pytest.approx(10.0)   # run 1 replaced step 2
    _same_as_fresh_build(store, pro_dir, tmp_path)


def test_forecast_tail_no_longer_produced_is_dropped(pro_dir, tmp_path):
    """Yesterday's +48 h forecast steps must not survive today's update."""
    store = tmp_path / "s.zarr"
    for cid in (1, 2):
        _write_pro(pro_dir, cid, [(i, 0) for i in range(8)])          # to +42 h
    build_zarr_cache(pro_dir, store, n_cpus=1, max_layers=MAXL, rebuild=True)
    for cid in (1, 2):
        _write_pro(pro_dir, cid, [(i, 0) for i in range(2)] + [(i, 1) for i in range(2, 5)])
    update_zarr_since(pro_dir, store, T0 + STEP, n_cpus=1, max_layers=MAXL)

    assert len(_times(store)) == 5
    _same_as_fresh_build(store, pro_dir, tmp_path)


def test_location_without_new_blocks_is_nan_in_the_window(pro_dir, tmp_path):
    store = tmp_path / "s.zarr"
    for cid in (1, 2):
        _write_pro(pro_dir, cid, [(i, 0) for i in range(3)])
    build_zarr_cache(pro_dir, store, n_cpus=1, max_layers=MAXL, rebuild=True)
    _write_pro(pro_dir, 1, [(i, 0) for i in range(3)] + [(3, 1), (4, 1)])  # cluster 2 not re-run
    update_zarr_since(pro_dir, store, T0 + 2 * STEP, n_cpus=1, max_layers=MAXL)

    with xr.open_zarr(store) as z:
        hs2 = z["HS"].sel(location="cluster_0002").values
    assert np.isfinite(hs2[:3]).all() and np.isnan(hs2[3:]).all()


def test_since_before_the_store_is_refused(pro_dir, tmp_path):
    store = tmp_path / "s.zarr"
    _write_pro(pro_dir, 1, [(i, 0) for i in range(2)])
    build_zarr_cache(pro_dir, store, n_cpus=1, max_layers=MAXL, rebuild=True)
    with pytest.raises(ValueError, match="no timestep"):
        update_zarr_since(pro_dir, store, T0 - STEP, n_cpus=1, max_layers=MAXL)


def test_windowed_build_holds_only_steps_after_since(pro_dir, tmp_path):
    """The NWP forecast store: only T_stable < t."""
    _write_pro(pro_dir, 1, [(i, 0) for i in range(6)])
    fc = tmp_path / "forecast.zarr"
    build_zarr_cache(pro_dir, fc, n_cpus=1, max_layers=MAXL, rebuild=True,
                     since=T0 + 3 * STEP)
    assert _times(fc) == [(T0 + i * STEP).strftime("%Y-%m-%dT%H:%M")
                          for i in (4, 5)]
