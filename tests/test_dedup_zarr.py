"""Tests for the duplicate-location store repair.

This tool rewrites operational input — the SNOWPACK store that takes hours to
rebuild from .pro files — so the parts that must not be wrong are the ones
tested here: that it keeps every cluster exactly once, that it preserves time,
layers and values untouched, that its verification actually fails when the
output is wrong rather than rubber-stamping it, and that the swap keeps the
original instead of deleting it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import dedup_zarr                    # noqa: E402


N_TIME = 8
N_LAYER = 3


def _store(tmp_path, name="src.zarr", n_clusters=6, n_copies=3,
           tail_partial=0):
    """A store shaped like production: the cluster set appended n_copies times
    at a stride of n_clusters, optionally followed by a partial batch — the
    interrupted-write case that left the odd stride-65 tail in the real store.
    """
    ids = list(np.tile(np.arange(1, n_clusters + 1), n_copies))
    ids += list(np.arange(1, tail_partial + 1))
    ids = np.array(ids)

    base = ids[:, None, None].astype("float32")
    density = base + np.zeros((len(ids), N_TIME, N_LAYER), dtype="float32")
    # A second variable with a different dtype and a time-varying signal, so a
    # mix-up between locations or timesteps cannot pass unnoticed.
    temp = (base * 100
            + np.arange(N_TIME, dtype="float32")[None, :, None]
            + np.zeros((len(ids), N_TIME, N_LAYER), dtype="float32"))
    hs = base[:, :, 0] + np.zeros((len(ids), N_TIME), dtype="float32")

    ds = xr.Dataset(
        data_vars={"density": (("location", "time", "layer"), density),
                   "temperature": (("location", "time", "layer"), temp),
                   "HS": (("location", "time"), hs)},
        coords={"location": [f"cluster_{i:04d}" for i in ids],
                "time": np.array(
                    [np.datetime64('2026-01-14') + np.timedelta64(6 * i, 'h')
                     for i in range(N_TIME)]),
                "layer": np.arange(N_LAYER),
                "latitude": ("location", 39.6 + ids / 1e4)},
    )
    path = tmp_path / name
    ds.chunk({'location': 4, 'time': N_TIME, 'layer': N_LAYER}) \
      .to_zarr(str(path), mode='w')
    return path


# ---------------------------------------------------------------------------
# as_slice — the contiguous fast path
# ---------------------------------------------------------------------------

def test_as_slice_collapses_a_contiguous_index():
    assert dedup_zarr.as_slice(np.arange(0, 6565)) == slice(0, 6565)
    assert dedup_zarr.as_slice(np.array([3, 4, 5])) == slice(3, 6)


def test_as_slice_leaves_a_scattered_index_alone():
    idx = np.array([0, 2, 7])
    out = dedup_zarr.as_slice(idx)
    assert isinstance(out, np.ndarray) and np.array_equal(out, idx)


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

def test_dedup_keeps_each_cluster_once_with_values_intact(tmp_path):
    src = _store(tmp_path, n_clusters=6, n_copies=3)
    out = tmp_path / "out.zarr"
    dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)

    with xr.open_zarr(str(out)) as ds:
        names = [str(v) for v in ds.coords['location'].values]
        assert names == [f"cluster_{i:04d}" for i in range(1, 7)]
        assert ds.sizes['time'] == N_TIME        # nothing dropped on time
        assert ds.sizes['layer'] == N_LAYER
        assert set(ds.data_vars) == {"density", "temperature", "HS"}

        for cid in range(1, 7):
            assert np.allclose(
                ds['density'].sel(location=f"cluster_{cid:04d}").values,
                float(cid))
            # temperature varies along time: catches a transposed or
            # mis-sliced time axis, which equal-per-location data would hide.
            assert np.allclose(
                ds['temperature'].sel(location=f"cluster_{cid:04d}").values,
                cid * 100 + np.arange(N_TIME)[:, None])


def test_dedup_rechunks_to_one_timestep(tmp_path):
    """The whole point of rewriting the store rather than only shrinking it."""
    src = _store(tmp_path)
    out = tmp_path / "out.zarr"
    dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)

    with xr.open_zarr(str(out)) as ds:
        assert ds['density'].encoding['chunks'][1] == 1
        assert ds['density'].encoding['chunks'][0] == 4


def test_dedup_handles_a_partial_trailing_batch(tmp_path):
    """The real store ends in a partial re-append at a different stride."""
    src = _store(tmp_path, n_clusters=6, n_copies=2, tail_partial=4)
    out = tmp_path / "out.zarr"
    dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)

    with xr.open_zarr(str(out)) as ds:
        names = [str(v) for v in ds.coords['location'].values]
        assert names == [f"cluster_{i:04d}" for i in range(1, 7)]


def test_dedup_on_a_clean_store_is_a_faithful_copy(tmp_path):
    src = _store(tmp_path, n_clusters=6, n_copies=1)
    out = tmp_path / "out.zarr"
    dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)

    with xr.open_zarr(str(src)) as a, xr.open_zarr(str(out)) as b:
        assert a.sizes['location'] == b.sizes['location'] == 6
        for v in a.data_vars:
            assert np.array_equal(a[v].values, b[v].values, equal_nan=True)


def test_dedup_refuses_to_clobber_an_existing_output(tmp_path):
    src = _store(tmp_path)
    out = tmp_path / "out.zarr"
    dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)
    with pytest.raises(FileExistsError):
        dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)


# ---------------------------------------------------------------------------
# verify — must be able to fail
# ---------------------------------------------------------------------------

def test_verify_passes_on_a_correct_dedup(tmp_path):
    src = _store(tmp_path)
    out = tmp_path / "out.zarr"
    dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)

    assert dedup_zarr.verify(src, out, n_clusters=6, n_times=N_TIME) is True


def test_verify_fails_when_a_cluster_is_missing(tmp_path):
    """A rubber-stamp verification would be worse than none, so prove it fails."""
    src = _store(tmp_path, n_clusters=6, n_copies=3)
    out = tmp_path / "out.zarr"
    with xr.open_zarr(str(src)) as ds:
        ds.isel(location=slice(0, 5)).to_zarr(str(out), mode='w')

    assert dedup_zarr.verify(src, out, n_clusters=6, n_times=N_TIME) is False


def test_verify_fails_when_values_are_corrupted(tmp_path):
    src = _store(tmp_path, n_clusters=6, n_copies=3)
    out = tmp_path / "out.zarr"
    with xr.open_zarr(str(src)) as ds:
        bad = ds.isel(location=slice(0, 6)).compute()
        bad['density'][2, 0, 0] += 1.0
        bad.to_zarr(str(out), mode='w')

    assert dedup_zarr.verify(src, out, n_clusters=6, n_times=N_TIME) is False


def test_verify_fails_when_timesteps_are_dropped(tmp_path):
    src = _store(tmp_path, n_clusters=6, n_copies=3)
    out = tmp_path / "out.zarr"
    with xr.open_zarr(str(src)) as ds:
        ds.isel(location=slice(0, 6), time=slice(0, 4)).to_zarr(str(out),
                                                                mode='w')

    assert dedup_zarr.verify(src, out, n_clusters=6, n_times=4) is False


def test_verify_fails_when_duplicates_survive(tmp_path):
    src = _store(tmp_path, n_clusters=6, n_copies=3)
    out = tmp_path / "out.zarr"
    with xr.open_zarr(str(src)) as ds:
        ds.isel(location=slice(0, 12)).to_zarr(str(out), mode='w')

    assert dedup_zarr.verify(src, out, n_clusters=6, n_times=N_TIME) is False


# ---------------------------------------------------------------------------
# swap — must not destroy the original
# ---------------------------------------------------------------------------

def test_swap_moves_the_original_aside_rather_than_deleting_it(tmp_path):
    src = _store(tmp_path, n_clusters=6, n_copies=3)
    out = tmp_path / "out.zarr"
    dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)

    dedup_zarr.swap(src, out)

    aside = src.with_name(src.name + ".duplicated")
    assert aside.exists(), "the original store must be kept"
    assert src.exists() and not out.exists()

    with xr.open_zarr(str(src)) as ds:          # deduped, now in place
        assert ds.sizes['location'] == 6
    with xr.open_zarr(str(aside)) as ds:        # original, untouched
        assert ds.sizes['location'] == 18


def test_swap_refuses_when_the_aside_path_is_taken(tmp_path):
    """Swapping twice would otherwise overwrite the one surviving original."""
    src = _store(tmp_path, n_clusters=6, n_copies=2)
    out = tmp_path / "out.zarr"
    dedup_zarr.dedup(src, out, loc_chunk=4, time_chunk=1)
    dedup_zarr.swap(src, out)

    out2 = tmp_path / "out2.zarr"
    dedup_zarr.dedup(src, out2, loc_chunk=4, time_chunk=1)
    with pytest.raises(FileExistsError):
        dedup_zarr.swap(src, out2)
