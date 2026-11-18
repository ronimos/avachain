"""Zarr builds on a slope with 10 000 clusters or more (Widow, 2026-11-17).

Widow's bootstrap ended SETUP EXIT 0 with 9 553 of its 12 681 locations in
the store. Two separate faults, both silent:

1. The `location` coordinate is a fixed-width string. The first batch fixed
   it at <U12 ("cluster_9999"), and every later batch holding a 5-digit id
   ("cluster_10000", <U13) failed to append: "Mismatched dtypes for variable
   location ... <U12 ... <U13".

2. SNOWPACK runs with NUMBER_SLOPES = 5. For station S it also runs a
   virtual slope n (1-4) whenever a snow file S<n>.sno exists, and writes it
   as S<n>_S.pro. Below 10 000 clusters no such file exists. On Widow,
   cluster_1000's run picked up the real clusters 10001-10004 and wrote
   cluster_10001_cluster_1000.pro and so on (1 072 files). The builder takes
   the location from the part before "_cluster_", so those files duplicated
   real locations. xsnow then failed the batch ("cannot select a dimension to
   squeeze out"), and because files sort lexically each failed batch also
   took ~25 four-digit ids down with it. update_zarr_since kept whichever of
   the two files the glob listed last.

The builder logged ERROR for each and still returned normally, so
season_start reported success. These tests use real xsnow on the
pro_header_block fixture.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import snowpack_io                                             # noqa: E402
from snowpack_io import build_zarr_cache, update_zarr_since   # noqa: E402

from test_zarr_update_since import (MAXL, STEP, T0,            # noqa: E402
                                    _split_fixture)


def _write(d: Path, station: int, experiment: int, n_steps=3, offset=0.0,
           slope=None):
    """A .pro named <station>_<experiment>.pro, StationName <experiment>
    (what SNOWPACK writes for a virtual slope), temperature marked."""
    head, block = _split_fixture()
    st, ex = f"cluster_{station:04d}", f"cluster_{experiment:04d}"
    head = head.replace("cluster_2859", ex)
    if slope is not None:          # a virtual slope carries S<n>'s slope
        head = head.replace("SlopeAngle= 31.08", f"SlopeAngle= {slope:.2f}")
    out = [head]
    for i in range(n_steps):
        t = T0 + i * STEP
        lines = []
        for ln in block.splitlines():
            if ln.startswith("0500,"):
                ln = "0500," + t.strftime("%d.%m.%Y %H:%M:%S")
            elif ln.startswith("0503,"):
                p = ln.split(",")
                p[2:] = [f"{float(v) + station + offset:.2f}" for v in p[2:]]
                ln = ",".join(p)
            lines.append(ln)
        out.append("\n".join(lines) + "\n")
    (d / f"{st}_{ex}.pro").write_text("".join(out))


def _real(d, cid, **kw):
    _write(d, cid, cid, **kw)


@pytest.fixture
def pro_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d


def _locs(store):
    with xr.open_zarr(store) as z:
        return [str(v) for v in z["location"].values]


def _temp0(store, loc, i=0):
    with xr.open_zarr(store) as z:
        return float(z["temperature"].sel(location=loc).isel(time=i)
                     .dropna("layer").values[0])


def test_five_digit_ids_after_four_digit_batch(pro_dir, tmp_path):
    """Batch 1 is all 4-digit ids, batch 2 holds 5-digit ones."""
    ids = [9997, 9998, 9999, 10000, 10001]
    for c in ids:
        _real(pro_dir, c)
    store = tmp_path / "s.zarr"
    build_zarr_cache(pro_dir, store, batch_size=3, n_cpus=1, max_layers=MAXL,
                     rebuild=True)
    assert sorted(_locs(store)) == sorted(f"cluster_{c:04d}" for c in ids)


def test_virtual_slope_files_are_not_locations(pro_dir, tmp_path):
    for c in (1000, 10001, 10002):
        _real(pro_dir, c)
    # cluster_1000's virtual slopes 1 and 2, as SNOWPACK names them.
    _write(pro_dir, 10001, 1000, offset=500.0, slope=45.44)
    _write(pro_dir, 10002, 1000, offset=500.0, slope=40.0)
    store = tmp_path / "s.zarr"
    build_zarr_cache(pro_dir, store, batch_size=100, n_cpus=1,
                     max_layers=MAXL, rebuild=True)
    assert sorted(_locs(store)) == ["cluster_1000", "cluster_10001",
                                    "cluster_10002"]
    # the real cluster_10001, not the virtual slope (marker +500)
    t_real = _temp0(store, "cluster_10001")

    _real(pro_dir, 10001, n_steps=4)
    _write(pro_dir, 10001, 1000, n_steps=4, offset=900.0, slope=45.44)
    update_zarr_since(pro_dir, store, T0 + STEP, batch_size=2, n_cpus=1,
                      max_layers=MAXL)
    assert _temp0(store, "cluster_10001") == t_real
    assert _temp0(store, "cluster_10001", 3) == pytest.approx(
        _temp0(store, "cluster_10001", 2))     # real file again, not +900


def test_failed_batch_fails_the_build(pro_dir, tmp_path, monkeypatch):
    for c in (1, 2, 3, 4):
        _real(pro_dir, c)
    real_read = snowpack_io._read_pro_batch

    def flaky(files, *a, **kw):
        if any("cluster_0003" in f.name for f in files):
            raise ValueError("synthetic read failure")
        return real_read(files, *a, **kw)

    monkeypatch.setattr(snowpack_io, "_read_pro_batch", flaky)
    store = tmp_path / "s.zarr"
    with pytest.raises(RuntimeError, match="cluster_0003"):
        build_zarr_cache(pro_dir, store, batch_size=2, n_cpus=1,
                         max_layers=MAXL, rebuild=True)
    # the batches that could be written still were
    assert _locs(store) == ["cluster_0001", "cluster_0002"]


def test_resume_into_narrow_store_refuses_wide_ids(pro_dir, tmp_path):
    for c in (9998, 9999):
        _real(pro_dir, c)
    store = tmp_path / "s.zarr"
    build_zarr_cache(pro_dir, store, n_cpus=1, max_layers=MAXL, rebuild=True)
    _real(pro_dir, 10000)
    with pytest.raises(RuntimeError, match="rebuild"):
        build_zarr_cache(pro_dir, store, n_cpus=1, max_layers=MAXL)
    assert _locs(store) == ["cluster_9998", "cluster_9999"]


def test_four_digit_store_keeps_its_dtype(pro_dir, tmp_path):
    """Byte-identity for Professor and Sisters: < 10 000 ids stay <U12."""
    for c in (1, 2, 3):
        _real(pro_dir, c)
    store = tmp_path / "s.zarr"
    build_zarr_cache(pro_dir, store, batch_size=2, n_cpus=1, max_layers=MAXL,
                     rebuild=True)
    with xr.open_zarr(store) as z:
        assert z["location"].dtype == np.dtype("<U12")
