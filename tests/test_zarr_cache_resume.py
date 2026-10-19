"""Regression tests for build_zarr_cache() resume behaviour.

The production store at snowpack/little_prof/output/slope_snowpack.zarr grew to
78 GB holding only 6565 distinct clusters across 59280 `location` entries —
every cluster present 9-12 times with byte-identical data, at index strides of
exactly 6565 (the full cluster set appended again, start to finish).

Two independent holes in the resume path produce that:

1. `mode` was taken from `zarr_out.exists()` while the skip-list came from a
   separate `xr.open_zarr()` inside a try/except. When that read failed the
   function printed "starting fresh" and then appended anyway, because an empty
   `written_locs` makes the skip guard `if written_locs and all(...)` False.
   One unreadable store turns into a second full copy.

2. The skip guard is all-or-nothing per batch. A batch interrupted part-way
   through its write leaves some of its locations in the store; on resume
   `all(...)` is False, so the whole batch is appended again and the locations
   that did land are duplicated.

Both are tested here with a stubbed xsnow.read, because reproducing them
against real .pro files means an hour of SNOWPACK output per run.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import snowpack_io                  # noqa: E402


N_TIME = 4
N_LAYER = 3


def _synthetic_batch(pro_tmp_dir):
    """Stand in for xsnow.read(): one location per .pro file in the directory.

    Mirrors the shape build_zarr_cache has to cope with — singleton slope and
    realization dims to squeeze, fewer layers than max_layers to pad — and
    makes each location's data a function of its own id so duplicate entries
    are detectable by value, not just by label.
    """
    stems = sorted(p.stem.split('_cluster_')[0]
                   for p in Path(pro_tmp_dir).glob("*.pro"))
    ids = np.array([int(s.split('_')[-1]) for s in stems], dtype=float)

    density = (ids[:, None, None]
               + np.zeros((len(stems), N_TIME, N_LAYER), dtype=float))
    return xr.Dataset(
        data_vars={
            "density": (("location", "time", "layer", "slope", "realization"),
                        density[..., None, None].astype("float32")),
        },
        coords={
            "location": stems,
            "time": np.arange(N_TIME),
            "layer": np.arange(N_LAYER),
            "slope": [0],
            "realization": [0],
            "latitude": ("location", 39.6 + ids / 1e4),
        },
    )


@pytest.fixture
def pro_dir(tmp_path):
    """Four empty .pro files named the way SNOWPACK writes them."""
    d = tmp_path / "pro"
    d.mkdir()
    for i in range(1, 5):
        (d / f"cluster_{i:04d}_cluster_{i:04d}.pro").touch()
    return d


@pytest.fixture(autouse=True)
def stub_xsnow_read(monkeypatch):
    monkeypatch.setattr(snowpack_io.xsnow, "read",
                        lambda tmp, **kw: _synthetic_batch(tmp))


def _locations(zarr_out):
    with xr.open_zarr(str(zarr_out)) as ds:
        return [str(v) for v in ds.coords['location'].values]


def _build(pro_dir, zarr_out, **kw):
    snowpack_io.build_zarr_cache(pro_dir=pro_dir, zarr_out=zarr_out,
                                 batch_size=2, n_cpus=1, max_layers=N_LAYER,
                                 **kw)


# ---------------------------------------------------------------------------
# Baseline: a clean build, and a resume that has nothing left to do
# ---------------------------------------------------------------------------

def test_clean_build_writes_each_location_once(pro_dir, tmp_path):
    zarr_out = tmp_path / "store.zarr"
    _build(pro_dir, zarr_out)

    locs = _locations(zarr_out)
    assert locs == [f"cluster_{i:04d}" for i in range(1, 5)]


def test_resume_over_complete_store_is_a_noop(pro_dir, tmp_path):
    zarr_out = tmp_path / "store.zarr"
    _build(pro_dir, zarr_out)
    _build(pro_dir, zarr_out)

    assert len(_locations(zarr_out)) == 4


# ---------------------------------------------------------------------------
# Hole 1 — unreadable existing store appended instead of rebuilt
# ---------------------------------------------------------------------------

def test_unreadable_store_does_not_silently_duplicate(pro_dir, tmp_path,
                                                      monkeypatch):
    """This is the 9x duplication, reproduced.

    With the resume read failing, the pre-fix code appended all four locations
    on top of the four already there.
    """
    zarr_out = tmp_path / "store.zarr"
    _build(pro_dir, zarr_out)
    assert len(_locations(zarr_out)) == 4

    real_open_zarr = snowpack_io.xr.open_zarr

    def failing_open_zarr(*a, **kw):
        raise RuntimeError("simulated corrupt/unreadable store")

    monkeypatch.setattr(snowpack_io.xr, "open_zarr", failing_open_zarr)
    with pytest.raises(RuntimeError, match="could not be read"):
        _build(pro_dir, zarr_out)
    monkeypatch.setattr(snowpack_io.xr, "open_zarr", real_open_zarr)

    # Refusing is only useful if it also left the store alone.
    assert len(_locations(zarr_out)) == 4


def test_rebuild_flag_overwrites_instead_of_appending(pro_dir, tmp_path):
    """The escape hatch for a store you have decided to throw away."""
    zarr_out = tmp_path / "store.zarr"
    _build(pro_dir, zarr_out)
    _build(pro_dir, zarr_out, rebuild=True)

    assert len(_locations(zarr_out)) == 4


# ---------------------------------------------------------------------------
# Hole 2 — partially written batch re-appended whole
# ---------------------------------------------------------------------------

def test_partially_written_batch_does_not_duplicate(pro_dir, tmp_path):
    """Batch 1 = clusters 1,2; batch 2 = clusters 3,4.

    Simulate an interrupt that landed cluster 3 but not cluster 4 by building
    from a pro_dir that holds only clusters 1-3, then resuming with all four.
    Pre-fix, batch 2 fails `all(... in written_locs)` and is appended entire,
    so cluster_0003 lands twice.
    """
    zarr_out = tmp_path / "store.zarr"
    partial = tmp_path / "pro_partial"
    partial.mkdir()
    for i in range(1, 4):
        (partial / f"cluster_{i:04d}_cluster_{i:04d}.pro").touch()

    _build(partial, zarr_out)
    assert _locations(zarr_out) == ["cluster_0001", "cluster_0002",
                                    "cluster_0003"]

    _build(pro_dir, zarr_out)

    locs = _locations(zarr_out)
    assert sorted(locs) == [f"cluster_{i:04d}" for i in range(1, 5)]
    assert len(locs) == len(set(locs)), f"duplicate locations: {locs}"


def test_resumed_store_keeps_one_row_per_location_with_right_data(pro_dir,
                                                                  tmp_path):
    """Duplicates were byte-identical in production, so check values too."""
    zarr_out = tmp_path / "store.zarr"
    partial = tmp_path / "pro_partial"
    partial.mkdir()
    (partial / "cluster_0001_cluster_0001.pro").touch()
    _build(partial, zarr_out)
    _build(pro_dir, zarr_out)

    with xr.open_zarr(str(zarr_out)) as ds:
        got = {str(loc): float(ds['density'].sel(location=loc).mean())
               for loc in ds.coords['location'].values}

    assert got == {f"cluster_{i:04d}": float(i) for i in range(1, 5)}


# ---------------------------------------------------------------------------
# Chunking: one timestep per chunk
# ---------------------------------------------------------------------------

class TestChunkEncoding:
    """A rebuild must not reintroduce the slow layout.

    Every consumer reads a single snapshot (isel(time=tidx) / sel(time=ts) in
    analysis_pipeline, analyze_release_zone, release_geometry,
    snowpack_analysis), so chunks must be time=1. build_zarr_cache used to
    call to_zarr() with no encoding and inherit the .pro reader's
    (25, 126, 85) chunks, which makes a one-timestep read decompress 126
    timesteps. The good layout existed only inside dedup_zarr's rechunk, so
    any plain rebuild silently undid it -- which is exactly what the
    2026-10-19 per-cluster-aspect re-baseline did.

    Note the badly-chunked store is SMALLER on disk (wide-time chunks compress
    better), so size is not a usable signal here.
    """

    def test_time_dim_is_chunked_to_one(self, pro_dir, tmp_path):
        out = tmp_path / "chunks.zarr"
        _build(pro_dir, out)
        with xr.open_zarr(str(out)) as ds:
            chunks = ds["density"].encoding["chunks"]
            dims = ds["density"].dims
        assert chunks[dims.index("time")] == 1, (
            f"time chunk is {chunks[dims.index('time')]}, not 1 — a snapshot "
            f"read will decompress that many timesteps")

    def test_layer_dim_is_one_whole_column(self, pro_dir, tmp_path):
        out = tmp_path / "chunks.zarr"
        _build(pro_dir, out)
        with xr.open_zarr(str(out)) as ds:
            chunks = ds["density"].encoding["chunks"]
            dims = ds["density"].dims
        assert chunks[dims.index("layer")] == N_LAYER, (
            "a profile should not be split across layer chunks")

    def test_encoding_helper_covers_every_data_var(self):
        ds = _synthetic_batch_like()
        enc = snowpack_io._chunk_encoding(ds, max_layers=338)
        assert set(enc) == set(ds.data_vars)
        for name, spec in enc.items():
            assert len(spec["chunks"]) == len(ds[name].dims)

    def test_encoding_helper_shapes(self):
        ds = _synthetic_batch_like()
        enc = snowpack_io._chunk_encoding(ds, max_layers=338, loc_chunk=256)
        dims = ds["density"].dims
        chunks = enc["density"]["chunks"]
        assert chunks[dims.index("location")] == 256
        assert chunks[dims.index("time")] == 1
        assert chunks[dims.index("layer")] == 338


def _synthetic_batch_like():
    """A squeezed (location, time, layer) frame, as written to the store."""
    return xr.Dataset(
        data_vars={"density": (("location", "time", "layer"),
                               np.zeros((4, N_TIME, N_LAYER), dtype="float32")),
                   "HS": (("location", "time"),
                          np.zeros((4, N_TIME), dtype="float32"))},
        coords={"location": [f"cluster_{i:04d}" for i in range(1, 5)],
                "time": np.arange(N_TIME), "layer": np.arange(N_LAYER)},
    )
