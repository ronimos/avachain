"""
tests/test_survey_anchor.py — a new survey sets every cluster's state to the
surveyed HS (reinitialize_snowpack.anchor_to_survey).

Run with:  pytest tests/test_survey_anchor.py -v

Why: SNOWPACK's HS enforcement only adds snow. In the season run at 03-05,
HS_mod sat a median 122 cm above its own forcing (4010 of 6565 clusters more
than 1 m above), because every high step was deposited and every low one
ignored. Re-running from the previous survey cannot fix that: its state
carries the excess. So at each survey the excess is removed (top-down for
event-scoured clusters, proportional thinning otherwise) and the forcing
after the survey starts at the survey HS, so a deficit is deposited.
"""

from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from reinitialize_snowpack import (anchor_to_survey, read_sno,      # noqa: E402
                                   survey_cluster_hs, thin_sno)
import clustering                                                    # noqa: E402

T = pd.Timestamp("2026-02-10T18:00")


# ---------------------------------------------------------------------------
# Cell median
# ---------------------------------------------------------------------------

class TestSurveyClusterHs:

    def test_median_ignores_one_wild_cell(self):
        cm = np.array([[1, 1, 1, 1, 1]])
        hs = np.array([[0.6, 0.7, 0.65, 0.62, 4.5]])
        assert survey_cluster_hs(hs, cm)[1] == pytest.approx(0.65)

    def test_nan_cells_ignored_and_clusters_split(self):
        cm = np.array([[1, 1, 2, 2, 0]])
        hs = np.array([[0.5, np.nan, 1.0, 3.0, 9.0]])
        out = survey_cluster_hs(hs, cm, min_cells=1)
        assert out == {1: pytest.approx(0.5), 2: pytest.approx(2.0)}

    def test_subset(self):
        cm = np.array([[1, 2, 3]])
        hs = np.array([[0.1, 0.2, 0.3]])
        assert set(survey_cluster_hs(hs, cm, {2, 3}, min_cells=1)) == {2, 3}

    def test_forcing_uses_the_median_too(self):
        cm = np.array([[1, 1, 1, 1, 1]])
        hs = np.array([[0.6, 0.7, 0.65, 0.62, 4.5]])
        rep = clustering.compute_cluster_representatives(cm, hs[None], [T])
        assert rep[1]["hs_series"][0] == pytest.approx(survey_cluster_hs(hs, cm)[1])

    def test_exact_zero_is_no_data(self):
        # clipped negatives (survey bias), not bare ground
        cm = np.array([[1, 1, 1, 1, 1]])
        hs = np.array([[0.0, 0.0, 0.8, 0.9, 0.85]])
        assert survey_cluster_hs(hs, cm)[1] == pytest.approx(0.85)

    def test_mostly_zero_cluster_not_trusted(self):
        cm = np.array([[1, 1, 1, 1, 1, 1, 1]])
        hs = np.array([[0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.85]])
        assert 1 not in survey_cluster_hs(hs, cm)

    def test_too_few_cells_not_trusted(self):
        cm = np.array([[1, 1]])
        hs = np.array([[0.8, 0.9]])
        assert 1 not in survey_cluster_hs(hs, cm)
        assert survey_cluster_hs(hs, cm, min_cells=1)[1] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Proportional thinning
# ---------------------------------------------------------------------------

def _sno_text(slope_deg: float, thick: list[float], ice=0.3) -> str:
    head = textwrap.dedent(f"""\
        SMET 1.1 ASCII
        [HEADER]
        ProfileDate      = 2026-02-10T18:00:00
        HS_Last          = {sum(thick):.6f}
        SlopeAngle       = {slope_deg}
        nSnowLayerData   = {len(thick)}
        ErosionLevel     = {len(thick) - 1}
        fields           = timestamp Layer_Thick Vol_Frac_I
        [DATA]
    """)
    return head + "".join(f"2025-11-26T18:00:00     {t:.6f}     {ice}\n" for t in thick)


class TestThinSno:

    def test_every_layer_scaled_and_fractions_kept(self, tmp_path):
        p = tmp_path / "a.sno"
        p.write_text(_sno_text(0.0, [0.2, 0.4]))
        out = thin_sno(read_sno(str(p)), 0.5)
        assert [float(l["Layer_Thick"]) for l in out["layers"]] == \
            pytest.approx([0.1, 0.2])
        assert all(l["Vol_Frac_I"] == "0.3" for l in out["layers"])
        assert len(out["layers"]) == 2                 # stratigraphy kept
        assert float(out["header"]["HS_Last"]) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# anchor_to_survey on a synthetic slope
# ---------------------------------------------------------------------------

def _smet(hs_cm: dict) -> str:
    rows = "".join(f"{t}:00  -5.0  {h:.3f}\n" for t, h in hs_cm.items())
    return ("SMET 1.1 ASCII\n[HEADER]\nnodata = -999\n"
            "fields = timestamp TA HS\n[DATA]\n" + rows)


@pytest.fixture
def slope(tmp_path):
    """Three 3-cell clusters on a 60 deg slope (cos = 0.5), survey 0.6 m:
    1: model 2.0 m vertical, not event-scoured  -> thinned
    2: model 2.0 m vertical, event-scoured      -> top-down
    3: model 0.4 m vertical                     -> below, forcing deposits
    """
    out, smet, an = tmp_path / "out", tmp_path / "smet", tmp_path / "an"
    for d in (out, smet, an):
        d.mkdir()
    cm = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3]])
    np.save(an / "cluster_map.npy", cm)
    np.save(an / "hs_2026-02-10.npy", np.full(cm.shape, 0.6))
    layers = {1: [0.25] * 4, 2: [0.25] * 4, 3: [0.1, 0.1]}   # slope-normal
    for cid, th in layers.items():
        name = f"cluster_{cid:04d}"
        (out / f"{name}_{name}.sno").write_text(_sno_text(60.0, th))
        (smet / f"{name}.smet").write_text(_smet(
            {"2026-02-10T17:00": 150.0, "2026-02-10T18:00": 150.0,
             "2026-02-10T19:00": 152.0}))
    cfg = SimpleNamespace(pro_dir=out, smet_dir=smet, analysis_dir=an,
                          cluster_map_path=an / "cluster_map.npy",
                          resampled_dir=an)
    return cfg


def _hs_vertical(cfg, cid):
    name = f"cluster_{cid:04d}"
    sno = read_sno(str(cfg.pro_dir / f"{name}_{name}.sno"))
    return sum(float(l["Layer_Thick"]) for l in sno["layers"]) / 0.5, sno


def _forcing(cfg, cid):
    name = f"cluster_{cid:04d}"
    return {l.split()[0][:16]: float(l.split()[2])
            for l in (cfg.smet_dir / f"{name}.smet").read_text().splitlines()
            if l.startswith("2026")}


class TestAnchorToSurvey:

    def test_actions(self, slope):
        r = anchor_to_survey(slope, T, top_down_cids={2})
        assert {s["cid"]: s["action"] for s in r["stats"]} == \
            {1: "thinned", 2: "top_down", 3: "below"}

    def test_excess_removed_to_the_survey(self, slope):
        anchor_to_survey(slope, T, top_down_cids={2})
        for cid in (1, 2):
            assert _hs_vertical(slope, cid)[0] == pytest.approx(0.6)

    def test_thinning_keeps_layers_top_down_removes_them(self, slope):
        anchor_to_survey(slope, T, top_down_cids={2})
        assert len(_hs_vertical(slope, 1)[1]["layers"]) == 4
        assert len(_hs_vertical(slope, 2)[1]["layers"]) < 4

    def test_below_survey_state_untouched(self, slope):
        anchor_to_survey(slope, T, top_down_cids={2})
        assert _hs_vertical(slope, 3)[0] == pytest.approx(0.4)

    def test_forcing_starts_at_survey_and_keeps_increments(self, slope):
        anchor_to_survey(slope, T, top_down_cids={2})
        for cid in (1, 2, 3):
            f = _forcing(slope, cid)
            assert f["2026-02-10T17:00"] == pytest.approx(150.0)   # before: as is
            assert f["2026-02-10T18:00"] == pytest.approx(60.0)
            assert f["2026-02-10T19:00"] == pytest.approx(62.0)

    def test_stats_saved(self, slope):
        anchor_to_survey(slope, T)
        assert (slope.analysis_dir / "anchor_stats_2026-02-10.json").exists()


    def test_untrusted_cluster_left_as_modelled(self, slope):
        hs = np.load(slope.resampled_dir / "hs_2026-02-10.npy")
        hs[0, 0:3] = 0.0                                   # cluster 1: no data
        np.save(slope.resampled_dir / "hs_2026-02-10.npy", hs)
        before = (slope.pro_dir / "cluster_0001_cluster_0001.sno").read_text()
        r = anchor_to_survey(slope, T)
        assert r["counts"]["no_survey"] == 1
        assert (slope.pro_dir / "cluster_0001_cluster_0001.sno").read_text() == before
        assert _forcing(slope, 1)["2026-02-10T18:00"] == pytest.approx(150.0)
