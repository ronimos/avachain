"""
tests/test_survey_resample.py — survey errors are removed at 4.6 cm, before
the 1 m grid.

Run with:  pytest tests/test_survey_resample.py -v

Survey nodata was always excluded, but pixels that are errors rather than
nodata were averaged into the 1 m cell: 22% of 03-02's pixels are below
-0.5 m (p0.1 -10.9 m even on clean surveys), and they moved 5% of its 1 m
cells by more than 5 cm (max 7.1 m). load_and_resample_survey drops
implausible pixels, takes the median of the rest, and leaves cells with too
few valid pixels as NaN instead of a clipped 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from spatial_model import load_and_resample, load_and_resample_survey  # noqa: E402
from survey_validity import compute_validity                          # noqa: E402
from clustering import build_survey_hs_matrix                         # noqa: E402

CRS = "EPSG:6342"
X0, Y0 = 400000.0, 4400000.0
NODATA = np.float32(-3.4028234663852886e+38)


def _write(path, arr, res, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32", crs=CRS,
                       transform=from_origin(X0, Y0, res, res), nodata=nodata) as dst:
        dst.write(arr.astype(np.float32), 1)


@pytest.fixture
def ref(tmp_path):
    p = tmp_path / "ref.tif"
    _write(p, np.zeros((1, 4)), 1.0)                 # four 1 m cells
    return p


def _survey(tmp_path, cells):
    """cells: four 20x20 pixel blocks (5 cm pixels -> 1 m cells)."""
    arr = np.concatenate(cells, axis=1)
    p = tmp_path / "survey.tif"
    _write(p, arr, 0.05, nodata=NODATA)
    return p


def _block(value=0.8, bad_share=0.0, bad=-5.0, rng=np.random.default_rng(0)):
    b = np.full((20, 20), value, dtype=np.float32)
    n_bad = int(round(bad_share * b.size))
    idx = rng.choice(b.size, n_bad, replace=False)
    b.flat[idx] = bad
    return b


class TestLoadAndResampleSurvey:

    def test_partial_negative_outliers_do_not_move_the_cell(self, tmp_path, ref):
        p = _survey(tmp_path, [_block(), _block(bad_share=0.3), _block(), _block()])
        new = load_and_resample_survey(str(p), str(ref), clip_min=0.0, clip_max=12.0)
        old = load_and_resample(str(p), str(ref), clip_min=0.0, clip_max=12.0)
        assert new[0, 1] == pytest.approx(0.8)
        assert old[0, 1] == pytest.approx(0.0)       # mean 0.7*0.8 - 0.3*5 < 0, clipped

    def test_positive_spikes_do_not_move_the_cell(self, tmp_path, ref):
        p = _survey(tmp_path, [_block(bad_share=0.2, bad=11.0), _block(), _block(), _block()])
        new = load_and_resample_survey(str(p), str(ref), clip_min=0.0, clip_max=12.0)
        assert new[0, 0] == pytest.approx(0.8)

    def test_mostly_bad_cell_is_nan_not_zero(self, tmp_path, ref):
        p = _survey(tmp_path, [_block(), _block(bad_share=0.6), _block(), _block()])
        new = load_and_resample_survey(str(p), str(ref), clip_min=0.0, clip_max=12.0)
        assert np.isnan(new[0, 1])

    def test_nodata_pixels_ignored(self, tmp_path, ref):
        b = _block(0.5)
        b[:5] = NODATA                                  # 25% nodata
        p = _survey(tmp_path, [b, _block(), _block(), _block()])
        new = load_and_resample_survey(str(p), str(ref), clip_min=0.0, clip_max=12.0)
        assert new[0, 0] == pytest.approx(0.5)

    def test_outside_footprint_is_nan(self, tmp_path, ref):
        b = np.full((20, 20), NODATA, dtype=np.float32)
        p = _survey(tmp_path, [_block(), b, _block(), _block()])
        new = load_and_resample_survey(str(p), str(ref), clip_min=0.0, clip_max=12.0)
        assert np.isnan(new[0, 1])

    def test_small_noise_negative_still_clips_to_zero(self, tmp_path, ref):
        # near-bare ground: noise around 0, inside the plausible band
        p = _survey(tmp_path, [_block(-0.1), _block(), _block(), _block()])
        new = load_and_resample_survey(str(p), str(ref), clip_min=0.0, clip_max=12.0)
        assert new[0, 0] == 0.0


class TestDownstreamNoData:

    def test_nan_cells_count_as_no_data_in_validity(self):
        cm = np.array([[1] * 10])
        g = {"2026-02-21": np.full((1, 10), 0.8),
             "2026-03-02": np.array([[np.nan] * 3 + [0.8] * 7])}
        assert compute_validity(g, cm)["2026-03-02"] == [1]

    def test_clustering_fills_instead_of_dropping(self):
        grids = {"a": np.array([[0.5, 0.6]]), "b": np.array([[np.nan, 0.7]]),
                 "c": np.array([[0.9, 0.8]])}
        M, idx, _ = build_survey_hs_matrix(grids, np.array([[True, True]]))
        assert len(idx) == 2
        assert M[0].tolist() == pytest.approx([0.5, 0.7, 0.9])

    def test_clustering_drops_cells_with_no_data_at_all(self):
        grids = {"a": np.array([[np.nan, 0.6]]), "b": np.array([[np.nan, 0.7]])}
        M, idx, _ = build_survey_hs_matrix(grids, np.array([[True, True]]))
        assert idx.tolist() == [[0, 1]]
