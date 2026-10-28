"""
tests/test_reinit_forcing_alignment.py — reinit must leave the scoured .sno
and the HS forcing agreeing, in vertical HS.

Run with:  pytest tests/test_reinit_forcing_alignment.py -v

Two defects, both found on the 2026-03-05 replay survey tick:

1. Reinit scoured the .sno to the post-event survey HS but left the SMET HS
   forcing at the gap-fill value. (A first fix that also ramped the forcing
   back to the survey HS at the survey time turned settlement into fresh
   snow and crashed cluster 0727.) SNOWPACK enforces measured HS, so it put
   the difference straight back as new snow: a median 80% of each scour
   within 6 h, and a 78 cm "snowfall" that crashed cluster 5859.
2. Frames. SNOWPACK enforces the forcing HS against the slope-normal layer
   sum (hn = forcing - sum, measured) and reports both scaled by 1/cos as
   HS_meas / HS_mod. run_snowpack.sh now projects the vertical forcing x cos,
   so the forcing, survey HS, HS_mod and .pro heights are all vertical and
   only the .sno layers are slope-normal.
"""

from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from reinitialize_snowpack import (align_scoured_forcing, align_smet_hs,
                                   read_sno, scour_cluster)


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


@pytest.fixture
def forcing(tmp_path):
    # 144 cm at the event, settling 2 cm/h afterwards (no snowfall)
    hs = {f"2026-03-03T{h:02d}:00": 144.0 - 2.0 * max(0, h - 12)
          for h in range(10, 19)}
    return _make_smet(tmp_path, hs)


class TestAlignSmetHs:

    def test_forcing_equals_scoured_hs_at_event(self, forcing):
        align_smet_hs(forcing, T_EVENT, 0.66)
        assert _read_hs(forcing)["2026-03-03T12:00"] == pytest.approx(66.0)

    def test_constant_offset_keeps_settlement(self, forcing):
        # A ramp back to the survey HS turned settlement into a rising
        # forcing, which SNOWPACK deposited as fresh snow (cluster 0727).
        before = _read_hs(forcing)
        align_smet_hs(forcing, T_EVENT, 0.66)
        after = _read_hs(forcing)
        for h in range(12, 19):
            t = f"2026-03-03T{h:02d}:00"
            assert before[t] - after[t] == pytest.approx(78.0)
        rising = [after[f"2026-03-03T{h + 1:02d}:00"] - after[f"2026-03-03T{h:02d}:00"]
                  for h in range(12, 18)]
        assert max(rising) <= 0.0

    def test_event_between_rows_is_interpolated(self, forcing):
        align_smet_hs(forcing, pd.Timestamp("2026-03-03T13:30"), 0.66)
        after = _read_hs(forcing)
        # forcing at 13:30 was 141 cm, so the offset is 75 cm from 14:00 on
        assert after["2026-03-03T14:00"] == pytest.approx(140.0 - 75.0)
        assert after["2026-03-03T13:00"] == pytest.approx(142.0)

    def test_rows_before_event_untouched(self, forcing):
        before = forcing.read_text().splitlines()
        align_smet_hs(forcing, T_EVENT, 0.66)
        after = forcing.read_text().splitlines()
        for b, a in zip(before, after):
            if b.startswith("2026-03-03T1") and b[:16] < "2026-03-03T12:00":
                assert a == b

    def test_only_hs_token_changes(self, forcing):
        before = forcing.read_text().splitlines()
        align_smet_hs(forcing, T_EVENT, 0.66)
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
        align_smet_hs(forcing, T_EVENT, 0.0)
        assert min(_read_hs(forcing).values()) >= 0.0

    def test_returns_rows_changed(self, forcing):
        assert align_smet_hs(forcing, T_EVENT, 0.66) == 7

    def test_reinit_does_not_pin_forcing_to_the_survey(self, forcing):
        # The stats carry the survey HS; aligning to it at the survey time
        # (the first fix) made this settling forcing rise.
        stats = [{"cid": 1, "hs_after_m": 0.66, "survey_hs_m": 0.66}]
        assert align_scoured_forcing(forcing.parent, stats, T_EVENT) == 1
        after = _read_hs(forcing)
        assert after["2026-03-03T18:00"] == pytest.approx(66.0 - 12.0)

    def test_event_after_last_row_changes_nothing(self, forcing):
        before = forcing.read_text()
        assert align_smet_hs(forcing, pd.Timestamp("2026-03-04T00:00"), 0.66) == 0
        assert forcing.read_text() == before


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
    """.sno layers are slope-normal; survey HS, slab_thickness (.pro) and the
    forcing (projected x cos by run_snowpack.sh) are vertical. scour_cluster
    works in vertical and cuts in slope-normal."""

    def test_survey_target_reached_in_vertical(self, tmp_path):
        # 60 deg: 1.0 m slope-normal = 2.0 m vertical
        sno = _make_sno(tmp_path, 60.0, [0.25] * 4)
        scoured, hs_before, scour_m, source = scour_cluster(
            sno, "2026-03-03T12:00", survey_hs_m=1.2)
        assert source == "survey_HS"
        assert hs_before == pytest.approx(2.0)
        assert scour_m == pytest.approx(0.8)
        assert scoured["scour_stats"]["hs_new_m"] == pytest.approx(1.2)
        # what SNOWPACK reports on restart: layer sum / cos(slope)
        normal = sum(float(l["Layer_Thick"]) for l in scoured["layers"])
        assert normal / math.cos(math.radians(60.0)) == pytest.approx(1.2)

    def test_survey_at_or_above_model_hs_means_no_scour(self, tmp_path):
        # 2.0 m vertical modelled; the survey sees 2.5 m (or exactly 2.0 m).
        # The survey says the snow is still there, so the slab thickness must
        # not be removed instead. Jan 18 2026: 9 release clusters had survey
        # 2.0-2.9 m and were stripped to 0.1-0.25 m by the slab fallback.
        sno = _make_sno(tmp_path, 60.0, [0.25] * 4)
        for survey in (2.5, 2.0):
            assert scour_cluster(sno, "2026-03-03T12:00", survey_hs_m=survey,
                                 slab_thickness_m=1.5) is None

    def test_survey_between_normal_and_vertical_hs_is_a_loss(self, tmp_path):
        # 1.0 m normal = 2.0 m vertical, so a 1.5 m survey is a real loss
        sno = _make_sno(tmp_path, 60.0, [0.25] * 4)
        _, _, scour_m, source = scour_cluster(sno, "2026-03-03T12:00",
                                              survey_hs_m=1.5, slab_thickness_m=0.3)
        assert source == "survey_HS"
        assert scour_m == pytest.approx(0.5)

    def test_slab_thickness_is_vertical(self, tmp_path):
        sno = _make_sno(tmp_path, 60.0, [0.25] * 4)
        scoured, _, scour_m, source = scour_cluster(
            sno, "2026-03-03T12:00", slab_thickness_m=0.5)
        assert source == "features"
        assert scour_m == pytest.approx(0.5)
        # 0.5 m vertical removes 0.25 m of slope-normal thickness
        normal = sum(float(l["Layer_Thick"]) for l in scoured["layers"])
        assert normal == pytest.approx(0.75)

    def test_hs_fraction_fallback(self, tmp_path):
        sno = _make_sno(tmp_path, 45.0, [0.25] * 4)
        scoured, hs_before, scour_m, source = scour_cluster(sno, "2026-03-03T12:00")
        assert source == "HS×0.80"
        assert scour_m == pytest.approx(0.8 * hs_before)
        assert scoured["scour_stats"]["hs_new_m"] == pytest.approx(0.2 * hs_before)

    def test_flat_slope_unchanged(self, tmp_path):
        sno = _make_sno(tmp_path, 0.0, [0.25] * 4)
        scoured, _, scour_m, _ = scour_cluster(
            sno, "2026-03-03T12:00", survey_hs_m=0.6)
        assert scour_m == pytest.approx(0.4)
        assert scoured["scour_stats"]["hs_new_m"] == pytest.approx(0.6)
