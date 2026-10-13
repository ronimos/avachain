"""Tests for the dev Zarr fixture builder.

The fixture exists to make the analyze loop fast enough to iterate on. Two of
its three jobs are easy to get silently wrong, so they are pinned here:

  - Deduplicating `location`. The production store holds 9-12 byte-identical
    copies of every cluster, and selecting by
    `int(name.split('_')[-1]) in ids` matches all of them. If the builder ever
    stops collapsing them, the fixture inherits the 9x over-selection it was
    written to remove and nothing downstream complains, because the per-cluster
    CSV keys by cluster id and the duplicate rows just overwrite each other.

  - Rechunking to time=1. The whole speedup is here: the source chunks span
    126 timesteps and analyze reads one, so a fixture that keeps the source
    encoding is a smaller store that is no faster per snapshot.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import make_dev_zarr                 # noqa: E402


N_TIME = 10
N_LAYER = 4


def _duplicated_store(tmp_path, n_clusters=5, n_copies=3):
    """A store shaped like production: the full cluster set appended n_copies
    times, at a stride of n_clusters, with identical data in each copy."""
    ids = np.tile(np.arange(1, n_clusters + 1), n_copies)
    names = [f"cluster_{i:04d}" for i in ids]

    density = (ids[:, None, None].astype("float32")
               + np.zeros((len(ids), N_TIME, N_LAYER), dtype="float32"))
    # Last two layers are NaN pad, as in the real store.
    z = density.copy()
    z[:, :, 2:] = np.nan

    ds = xr.Dataset(
        data_vars={"density": (("location", "time", "layer"), density),
                   "z":       (("location", "time", "layer"), z)},
        coords={"location": names,
                "time": np.array(
                    [np.datetime64('2026-01-14') + np.timedelta64(6 * i, 'h')
                     for i in range(N_TIME)]),
                "layer": np.arange(N_LAYER)},
    )
    path = tmp_path / "src.zarr"
    # Chunk time wide, like the source store that makes snapshot reads slow.
    ds.chunk({'location': 2, 'time': N_TIME, 'layer': N_LAYER}) \
      .to_zarr(str(path), mode='w')
    return path, ds


def test_unique_location_index_collapses_duplicates(tmp_path):
    path, _ = _duplicated_store(tmp_path, n_clusters=5, n_copies=3)
    with xr.open_zarr(str(path)) as ds:
        assert ds.sizes['location'] == 15
        idx = make_dev_zarr.unique_location_index(ds, {1, 2, 3, 4, 5})

        assert len(idx) == 5
        got = [str(v) for v in ds.coords['location'].values[idx]]
        assert sorted(got) == [f"cluster_{i:04d}" for i in range(1, 6)]


def test_unique_location_index_honours_the_cluster_subset(tmp_path):
    path, _ = _duplicated_store(tmp_path)
    with xr.open_zarr(str(path)) as ds:
        idx = make_dev_zarr.unique_location_index(ds, {2, 4})
        got = [str(v) for v in ds.coords['location'].values[idx]]
        assert got == ["cluster_0002", "cluster_0004"]


def test_unique_location_index_warns_on_absent_clusters(tmp_path, capsys):
    """A requested cluster the store does not have must be reported, not
    silently dropped — downstream it would look like a missing physics case."""
    path, _ = _duplicated_store(tmp_path)
    with xr.open_zarr(str(path)) as ds:
        idx = make_dev_zarr.unique_location_index(ds, {1, 999})

    assert len(idx) == 1
    assert "999" in capsys.readouterr().out


def test_time_index_keeps_the_padded_window(tmp_path):
    path, _ = _duplicated_store(tmp_path)
    with xr.open_zarr(str(path)) as ds:
        idx = make_dev_zarr.time_index(ds, ['2026-01-15'], pad_days=0)
        kept = ds.coords['time'].values[idx]
        assert len(kept) == 4                       # the four 6-hourly steps
        assert str(kept[0])[:10] == '2026-01-15'


def test_time_index_falls_back_to_nearest_when_window_is_empty(tmp_path,
                                                               capsys):
    path, _ = _duplicated_store(tmp_path)
    with xr.open_zarr(str(path)) as ds:
        idx = make_dev_zarr.time_index(ds, ['2027-03-01'], pad_days=1)

    assert len(idx) == 1
    assert "nearest" in capsys.readouterr().out


def test_measure_max_layer_finds_the_pad_boundary(tmp_path):
    path, _ = _duplicated_store(tmp_path)
    with xr.open_zarr(str(path)) as ds:
        assert make_dev_zarr.measure_max_layer(ds, probe_var='z') == 2


def test_build_writes_deduplicated_store_chunked_at_one_timestep(tmp_path,
                                                                 monkeypatch):
    path, src = _duplicated_store(tmp_path, n_clusters=5, n_copies=3)
    out = tmp_path / "dev.zarr"

    monkeypatch.setattr(make_dev_zarr, "dilated_start_zone_clusters",
                        lambda cfg, buffer_m: {1, 2, 3, 4, 5})

    make_dev_zarr.build(path, out, cfg=None, dates=['2026-01-15'],
                        pad_days=0, buffer_m=0.0, loc_chunk=2,
                        trim_layers=True)

    with xr.open_zarr(str(out)) as ds:
        names = [str(v) for v in ds.coords['location'].values]
        assert len(names) == len(set(names)) == 5
        assert ds.sizes['time'] == 4
        assert ds.sizes['layer'] == 2               # NaN pad trimmed
        # The point of the exercise: one timestep per chunk.
        assert ds['density'].encoding['chunks'][1] == 1

        # Values must survive the round trip unchanged.
        for cid in range(1, 6):
            v = ds['density'].sel(location=f"cluster_{cid:04d}").values
            assert np.allclose(v, float(cid))


def test_build_refuses_to_clobber_an_existing_fixture(tmp_path, monkeypatch):
    path, _ = _duplicated_store(tmp_path)
    out = tmp_path / "dev.zarr"
    monkeypatch.setattr(make_dev_zarr, "dilated_start_zone_clusters",
                        lambda cfg, buffer_m: {1, 2})

    kw = dict(cfg=None, dates=['2026-01-15'], pad_days=0, buffer_m=0.0,
              loc_chunk=2, trim_layers=False)
    make_dev_zarr.build(path, out, **kw)
    with pytest.raises(FileExistsError):
        make_dev_zarr.build(path, out, **kw)


def test_build_raises_when_nothing_is_selected(tmp_path, monkeypatch):
    path, _ = _duplicated_store(tmp_path)
    monkeypatch.setattr(make_dev_zarr, "dilated_start_zone_clusters",
                        lambda cfg, buffer_m: {4242})

    with pytest.raises(ValueError, match="No locations selected"):
        make_dev_zarr.build(path, tmp_path / "dev.zarr", cfg=None,
                            dates=['2026-01-15'], pad_days=0, buffer_m=0.0,
                            loc_chunk=2, trim_layers=False)
