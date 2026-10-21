"""
tests/test_reinit_forcing_alignment.py — reinit must leave the scoured .sno
and the HS forcing agreeing, in the frame SNOWPACK enforces HS in.

Run with:  pytest tests/test_reinit_forcing_alignment.py -v

Two defects, both found on the 2026-03-05 replay survey tick:

1. Reinit scoured the .sno to the post-event survey HS but left the SMET HS
   forcing at the gap-fill value. SNOWPACK enforces measured HS, so it put
   the difference straight back as new snow: a median 80% of each scour
   within 6 h, and a 78 cm "snowfall" that crashed cluster 5859.
2. Frames. SNOWPACK enforces the forcing HS against the slope-normal layer
   sum (hn = forcing - sum, measured), and reports both scaled by 1/cos as
   HS_meas / HS_mod. The survey HS (what the forcing carries) therefore
   compares with the layer sum as is; the .pro-derived slab thickness is in
   the 1/cos-scaled frame and must be multiplied by cos(slope).
"""

from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from reinitialize_snowpack import align_smet_hs, read_sno, scour_cluster


# ---------------------------------------------------------------------------
# align_smet_hs
# ---------------------------------------------------------------------------

def _make_smet(tmp_path: Path, hs_cm: dict) -> Path:
    """Hourly SMET; rows up to 12:00 in write_smet style (spaces), later
    ones in smet_append style (tabs, 4 decimals), as in the live files."""
    head = textwrap.dedent("""\
        SMET 1.1 ASCII
        [HEADER]
        station_id       = cluster_0001
        nodata           = -999.0
        units_multiplier = 1 1 0.01
        fields           = timestamp TA HS
        [DATA]
    """)
    rows = []
    for ts, hs in hs_cm.items():
        if ts <= "2026-03-03T12:00":
            rows.append(f"{ts}:00  -6.833  {hs:.3f}\n")
        else:
            rows.append(f"{ts}\t-2.5000\t{hs:.4f}\n")
    p = tmp_path / "cluster_0001.smet"
    p.write_text(head + "".join(rows))
    return p


def _read_hs(p: Path) -> dict:
    out, data = {}, False
    for line in p.read_text().splitlines():
        if data:
            parts = line.split()
            out[parts[0][:16]] = float(parts[2])
        data = data or line.strip() == "[DATA]"
    return out


T_EVENT = pd.Timestamp("2026-03-03T12:00")
T_SURVEY = pd.Timestamp("2026-03-03T16:00")


@pytest.fixture
def forcing(tmp_path):
    # 144 cm at the event, +2 cm/h of snowfall afterwards
    hs = {f"2026-03-03T{h:02d}:00": 144.0 + 2.0 * max(0, h - 12)
          for h in range(10, 19)}
    return _make_smet(tmp_path, hs)


class TestAlignSmetHs:

    def test_forcing_equals_scoured_hs_at_event(self, forcing):
        align_smet_hs(forcing, T_EVENT, 0.66, T_SURVEY, 0.80)
        assert _read_hs(forcing)["2026-03-03T12:00"] == pytest.approx(66.0)

    def test_forcing_equals_survey_at_survey(self, forcing):
        align_smet_hs(forcing, T_EVENT, 0.66, T_SURVEY, 0.80)
        assert _read_hs(forcing)["2026-03-03T16:00"] == pytest.approx(80.0)

    def test_offset_linear_between_event_and_survey(self, forcing):
        before = _read_hs(forcing)
        align_smet_hs(forcing, T_EVENT, 0.66, T_SURVEY, 0.80)
        after = _read_hs(forcing)
        d_event, d_survey = 144.0 - 66.0, 152.0 - 80.0
        d_mid = after["2026-03-03T14:00"] - before["2026-03-03T14:00"]
        assert d_mid == pytest.approx(-(d_event + d_survey) / 2)

    def test_increments_kept_after_survey(self, forcing):
        before = _read_hs(forcing)
        align_smet_hs(forcing, T_EVENT, 0.66, T_SURVEY, 0.80)
        after = _read_hs(forcing)
        for h in (17, 18):
            t = f"2026-03-03T{h:02d}:00"
            assert after[t] - after["2026-03-03T16:00"] == \
                pytest.approx(before[t] - before["2026-03-03T16:00"])

    def test_no_survey_holds_event_offset(self, forcing):
        before = _read_hs(forcing)
        align_smet_hs(forcing, T_EVENT, 0.66)
        after = _read_hs(forcing)
        for t in ("2026-03-03T12:00", "2026-03-03T15:00", "2026-03-03T18:00"):
            assert before[t] - after[t] == pytest.approx(78.0)

    def test_rows_before_event_untouched(self, forcing):
        before = forcing.read_text().splitlines()
        align_smet_hs(forcing, T_EVENT, 0.66, T_SURVEY, 0.80)
        after = forcing.read_text().splitlines()
        for b, a in zip(before, after):
            if b.startswith("2026-03-03T1") and b[:16] < "2026-03-03T12:00":
                assert a == b

    def test_only_hs_token_changes(self, forcing):
        before = forcing.read_text().splitlines()
        align_smet_hs(forcing, T_EVENT, 0.66, T_SURVEY, 0.80)
        after = forcing.read_text().splitlines()
        assert len(after) == len(before)
        for b, a in zip(before, after):
            sep = "\t" if "\t" in b else None
            bt, at = b.split(sep), a.split(sep)
            assert bt[:-1] == at[:-1] or not b.startswith("2026")
            if b.startswith("2026"):
                # same decimals as the original token
                assert len(bt[-1].split(".")[1]) == len(at[-1].split(".")[1])

    def test_clipped_at_zero(self, forcing):
        align_smet_hs(forcing, T_EVENT, 0.0, T_SURVEY, 0.0)
        assert min(_read_hs(forcing).values()) >= 0.0

    def test_returns_rows_changed(self, forcing):
        assert align_smet_hs(forcing, T_EVENT, 0.66, T_SURVEY, 0.80) == 7


# ---------------------------------------------------------------------------
# scour_cluster — vertical vs slope-normal
# ---------------------------------------------------------------------------

def _make_sno(tmp_path: Path, slope_deg: float, thick: list[float]) -> dict:
    hs = sum(thick)
    head = textwrap.dedent(f"""\
        SMET 1.1 ASCII
        [HEADER]
        ProfileDate      = 2026-03-03T12:00:00
        HS_Last          = {hs:.6f}
        SlopeAngle       = {slope_deg}
        nSnowLayerData   = {len(thick)}
        ErosionLevel     = {len(thick) - 1}
        fields           = timestamp Layer_Thick
        [DATA]
    """)
    rows = "".join(f"2025-11-26T18:00:00     {t:.6f}\n" for t in thick)
    p = tmp_path / "cluster_0001_cluster_0001.sno"
    p.write_text(head + rows)
    return read_sno(str(p))


class TestScourClusterFrame:
    """SNOWPACK enforces forcing HS against the slope-normal layer sum, so
    the survey HS is compared with that sum directly; only the .pro-derived
    slab thickness (scaled by 1/cos) is converted."""

    def test_survey_target_is_layer_sum(self, tmp_path):
        sno = _make_sno(tmp_path, 60.0, [0.25] * 4)
        scoured, hs_before, scour_m, source = scour_cluster(
            sno, "2026-03-03T12:00", survey_hs_m=0.6)
        assert source == "survey_HS"
        assert hs_before == pytest.approx(1.0)
        normal = sum(float(l["Layer_Thick"]) for l in scoured["layers"])
        # equal to what the aligned forcing will carry: no deposition
        assert normal == pytest.approx(0.6)

    def test_survey_above_layer_sum_falls_back(self, tmp_path):
        sno = _make_sno(tmp_path, 60.0, [0.25] * 4)
        _, _, _, source = scour_cluster(sno, "2026-03-03T12:00",
                                        survey_hs_m=1.5, slab_thickness_m=0.3)
        assert source == "features"

    def test_slab_thickness_scaled_by_cos(self, tmp_path):
        # 0.5 m of .pro height on a 60 deg slope is 0.25 m of layer sum
        sno = _make_sno(tmp_path, 60.0, [0.25] * 4)
        scoured, _, scour_m, source = scour_cluster(
            sno, "2026-03-03T12:00", slab_thickness_m=0.5)
        assert source == "features"
        assert scour_m == pytest.approx(0.25)
        normal = sum(float(l["Layer_Thick"]) for l in scoured["layers"])
        assert normal == pytest.approx(0.75)

    def test_hs_fraction_fallback_frame_free(self, tmp_path):
        sno = _make_sno(tmp_path, 45.0, [0.25] * 4)
        _, _, scour_m, source = scour_cluster(sno, "2026-03-03T12:00")
        assert source == "HS×0.80"
        assert scour_m == pytest.approx(0.8)
