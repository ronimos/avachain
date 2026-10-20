"""Tests that install.py's templates and the scripts on disk agree.

Two fixes have now been lost to generator/generated divergence:

  * the reinit_stats path was corrected in run_little_prof.sh but not in
    install.py, so `install.py --force` silently reverted it to the
    un-namespaced outputs/analysis/ -- which makes reinit pass 2 rerun every
    cluster instead of the scoured set;
  * the MeteoIO METEO/SNOW plugin lines existed in the pre-rename
    master_config.ini but not in the template that replaced it, so SNOWPACK
    could not start at all.

Both were invisible until something actually ran. So these tests assert the
generated artifacts match what install.py would produce *now*, and that the
per-cluster terrain wiring is present in the template rather than only in a
file someone hand-edited.
"""
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SLOPES = ["little_prof", "little_prof_replay"]


@pytest.fixture(scope="module")
def install():
    spec = importlib.util.spec_from_file_location("install_mod", REPO / "install.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["install_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfg(slope: str) -> dict:
    with open(REPO / "slopes" / slope / "slope_config.toml", "rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize("slope", SLOPES)
class TestOnDiskMatchesTemplate:
    """The committed scripts must be exactly what install.py renders."""

    def test_run_snowpack_matches(self, install, slope):
        on_disk = (REPO / "slopes" / slope / "run_snowpack.sh").read_text()
        assert on_disk == install._render_run_snowpack(_cfg(slope)), (
            f"slopes/{slope}/run_snowpack.sh has drifted from install.py's "
            f"template -- a hand edit here is reverted by --force")

    def test_run_daily_matches(self, install, slope):
        on_disk = (REPO / f"run_daily_{slope}.sh").read_text()
        assert on_disk == install._render_run_daily(_cfg(slope)), (
            f"run_daily_{slope}.sh has drifted from install.py's template")

    def test_run_operational_matches(self, install, slope):
        on_disk = (REPO / f"run_{slope}.sh").read_text()
        assert on_disk == install._render_run_operational(_cfg(slope)), (
            f"run_{slope}.sh has drifted from install.py's template")

    def test_master_config_matches(self, install, slope):
        on_disk = (REPO / "slopes" / slope / "config" / "master_config.ini").read_text()
        assert on_disk == install._render_master_ini(_cfg(slope)), (
            f"slopes/{slope}/config/master_config.ini has drifted")


class TestMeteoIOPluginsDeclared:
    """Without these SNOWPACK aborts before reading any data."""

    @pytest.mark.parametrize("slope", SLOPES)
    def test_input_plugins(self, install, slope):
        ini = install._render_master_ini(_cfg(slope))
        inp = ini.split("[INPUT]", 1)[1].split("[", 1)[0]
        assert "METEO      = SMET" in inp, "No plugin defined for METEO"
        assert "SNOW       = SMET" in inp

    @pytest.mark.parametrize("slope", SLOPES)
    def test_output_plugins(self, install, slope):
        ini = install._render_master_ini(_cfg(slope))
        out = ini.split("[OUTPUT]", 1)[1].split("\n[", 1)[0]
        assert "METEO                = SMET" in out
        assert "SNOW                 = SMET" in out


class TestPerClusterTerrain:
    """Each cluster must run on its own slope angle and aspect.

    Before this, template.sno baked in the slope-level 38 deg / 90 deg, so all
    6565 clusters were simulated at that geometry while their true terrain
    spanned 8-81 deg and 1-356 deg -- inflating tau_g = rho g h sin(psi) by a
    median 1.28x (p95 2.0x).
    """

    def test_template_sno_uses_placeholders(self, install):
        sno = install._render_template_sno(_cfg("little_prof"))
        assert "SlopeAngle       = WRFSLOPE" in sno
        assert "SlopeAzi         = WRFAZI" in sno
        # The old slope-wide bake-in must be gone.
        assert "SlopeAngle       = 38.0" not in sno

    @pytest.mark.parametrize("slope", SLOPES)
    def test_run_snowpack_reads_terrain_from_smet(self, install, slope):
        sh = install._render_run_snowpack(_cfg(slope))
        assert "/^slope_angle/" in sh, "slope angle not read from the SMET header"
        assert "/^slope_azi/" in sh

    @pytest.mark.parametrize("slope", SLOPES)
    def test_run_snowpack_substitutes_terrain(self, install, slope):
        sh = install._render_run_snowpack(_cfg(slope))
        assert 's/WRFSLOPE/$sang/g' in sh
        assert 's/WRFAZI/$sazi/g' in sh

    @pytest.mark.parametrize("slope", SLOPES)
    def test_steep_slopes_are_clamped(self, install, slope):
        """46 of 6565 clusters exceed 55 deg; p99 is 49.7 deg."""
        sh = install._render_run_snowpack(_cfg(slope))
        assert "MAX_SLOPE_DEG=55.0" in sh
        assert "n_clamped" in sh, "clamping must be reported, not silent"

    @pytest.mark.parametrize("slope", SLOPES)
    def test_fallback_to_slope_level_values(self, install, slope):
        """A SMET with no slope headers falls back, rather than emitting empty."""
        sh = install._render_run_snowpack(_cfg(slope))
        assert "DEFAULT_SLOPE_ANGLE" in sh and "DEFAULT_SLOPE_AZI" in sh


class TestSlopeNamespacedPaths:
    @pytest.mark.parametrize("slope", SLOPES)
    def test_no_unnamespaced_output_paths(self, install, slope):
        """outputs/analysis/ (no slope) once made reinit pass 2 rerun everything."""
        sh = install._render_run_operational(_cfg(slope))
        assert "outputs/analysis/" not in sh
        assert f"outputs/{slope}/analysis/cluster_map.npy" in sh

    @pytest.mark.parametrize("slope", SLOPES)
    def test_every_python_entry_point_gets_toml(self, install, slope):
        """Without --toml a subprocess silently uses the default slope's paths."""
        sh = install._render_run_operational(_cfg(slope))
        for mod in ("forcing_pipeline", "analysis_pipeline", "nwp_ingest",
                    "smet_append", "snowpack_cycle"):
            assert f"src/avachain/{mod}.py --toml $TOML" in sh, mod


class TestSurveyFlow:
    """Survey arrival: detect events on the new pair, re-run from the last survey."""

    @pytest.mark.parametrize("slope", SLOPES)
    def test_detection_runs_before_gap_fill(self, install, slope):
        sh = install._render_run_operational(_cfg(slope))
        i_tr, i_av = sh.index("$PIPELINE transport"), sh.index("$PIPELINE avalanche")
        assert i_tr < i_av < sh.index("$PIPELINE gap_fill")

    @pytest.mark.parametrize("slope", SLOPES)
    def test_snowpack_goes_through_the_survey_cycle(self, install, slope):
        sh = install._render_run_operational(_cfg(slope))
        assert "survey --survey-date $SURVEY_DATE" in sh
        assert "run_snowpack.sh" not in sh          # no direct, unanchored runs

    @pytest.mark.parametrize("slope", SLOPES)
    def test_observations_after_the_survey_are_appended(self, install, slope):
        sh = install._render_run_operational(_cfg(slope))
        assert sh.index("$PIPELINE smet") < sh.index("$SMET_APPEND") < sh.index("$NWP --lead-hours")

    @pytest.mark.parametrize("slope", SLOPES)
    def test_smet_dir_default_exists(self, install, slope):
        """Defaulted to a nonexistent outputs/smet, so SNOWPACK found no input."""
        sh = install._render_run_snowpack(_cfg(slope))
        assert 'SMET_DIR=${SMET_DIR:-$SLOPE_DIR/input/smet}' in sh
        assert "outputs/smet" not in sh


class TestDateHandling:
    @pytest.mark.parametrize("slope", SLOPES)
    def test_edate_is_conditional_like_bdate(self, install, slope):
        """Unconditional T18:00 made "...T12:00T18:00" -> Date object undefined."""
        sh = install._render_run_snowpack(_cfg(slope))
        assert '[[ "$EDATE" != *T* ]] && EDATE="${EDATE}T18:00"' in sh
        assert '[[ "$BDATE" != *T* ]] && BDATE="${BDATE}T18:00"' in sh


class TestSuccessDetection:
    @pytest.mark.parametrize("slope", SLOPES)
    def test_checks_the_string_snowpack_actually_prints(self, install, slope):
        """SNOWPACK 3.7.x never prints "done!"; the season logs have 0 matches."""
        sh = install._render_run_snowpack(_cfg(slope))
        assert 'grep -q "FINISHED running"' in sh
        assert 'grep -q "done!"' not in sh

    @pytest.mark.parametrize("slope", SLOPES)
    def test_also_requires_the_restart_artifact_at_the_end_date(self, install, slope):
        """A non-empty .sno is not enough: a run stalled on HS=-999 leaves one
        at its begin date and still prints FINISHED (behaviour pinned in
        test_run_snowpack_restart.py)."""
        sh = install._render_run_snowpack(_cfg(slope))
        assert '[[ "$pdate" == "${EDATE:0:16}" ]]' in sh


class TestDailyFlow:
    """The daily forward was a hand-written run_daily.sh hardwired to little_prof."""

    @pytest.mark.parametrize("slope", SLOPES)
    def test_daily_is_slope_namespaced(self, install, slope):
        sh = install._render_run_daily(_cfg(slope))
        assert f"slopes/{slope}/slope_config.toml" in sh
        assert f"outputs/{slope}/logs" in sh
        assert "outputs/analysis/" not in sh

    @pytest.mark.parametrize("slope", SLOPES)
    def test_daily_restarts_through_the_cycle(self, install, slope):
        sh = install._render_run_daily(_cfg(slope))
        assert 'CYCLE_ARGS="daily"' in sh
        assert "run_snowpack.sh" not in sh

    @pytest.mark.parametrize("slope", SLOPES)
    def test_daily_appends_before_nwp_and_scenarios_are_optional(self, install, slope):
        sh = install._render_run_daily(_cfg(slope))
        assert sh.index("$SMET_APPEND\n") < sh.index("$NWP --lead-hours")
        assert "--no-scenarios" in sh

    @pytest.mark.parametrize("slope", SLOPES)
    def test_every_daily_python_entry_point_gets_toml(self, install, slope):
        sh = install._render_run_daily(_cfg(slope))
        for mod in ("analysis_pipeline", "aws_ingest", "smet_append",
                    "nwp_ingest", "snowpack_cycle"):
            assert f"src/avachain/{mod}.py --toml $TOML" in sh, mod
