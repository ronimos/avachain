"""
Project configuration for distributed SNOWPACK forcing generation.

For multi-slope deployments, use ProjectConfig.from_toml() instead of editing
this file directly.  See slopes/template/slope_config.toml for the format.
"""

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Horizontal CRS of all project rasters (cluster_map.tif, DEM, scenario outputs).
# NAD83(2011) / UTM zone 13N — the actual files are COMPD with NAVD88 vertical.
# rasterio returns EPSG=None for compound CRS; the horizontal component is 6342.
RASTER_EPSG = 6342

# Directories a slope TOML may relocate individually under [paths]. Each
# defaults to its historical place under output_dir (weather_cache_dir: the
# shared project_dir/outputs/weather). Relative values resolve against
# project_dir; absolute ones are used as given.
OVERRIDABLE_DIRS = (
    "analysis_dir", "plots_dir", "grids_dir", "resampled_dir", "scenarios_dir",
    "models_dir", "logs_dir", "event_masks_dir", "weather_cache_dir",
)

def wrf_season(wrf: dict, season_start: Optional[str]) -> Optional[str]:
    """The WRF forecast season directory name for a slope.

    [wrf] season wins when set. Otherwise it is the year of [snowpack]
    season_start: the forecast archive is <fcst_base_dir>/<YYYY>/ per season
    (2025 holds 2025-09-23 .. 2026-06), so the season SNOWPACK runs is the
    season the WRF forcing comes from. "Latest year directory" was the old
    rule and is wrong as soon as next season's directory exists (2026/ did
    by 2026-10), which would move every 2025-26 slope and replay to it.
    """
    if wrf.get("season"):
        return str(wrf["season"])
    try:
        return str(_dt.date.fromisoformat(str(season_start)).year)
    except (TypeError, ValueError):
        return None


def resolve_wrf_smet(wrf: dict, season_start: Optional[str]) -> Optional[Path]:
    """The WRF-forced SMET for this slope's season.

    Built as <fcst_base_dir>/<season>/<wrf_zone>/<wrf_grid_id>/<file name>,
    keeping only the file name (the aspect variant install.py chose) from the
    stamped wrf_smet_file, so its season directory cannot go stale. Falls
    back to wrf_smet_file as written when the parts are missing. The file
    may not exist; callers that need it say so (forcing_pipeline.step_smet).
    """
    stamped = wrf.get("wrf_smet_file")
    season = wrf_season(wrf, season_start)
    parts = (wrf.get("fcst_base_dir"), season, wrf.get("wrf_zone"), wrf.get("wrf_grid_id"))
    if stamped and all(parts):
        return Path(*parts) / Path(stamped).name
    return Path(stamped) if stamped else None


# Environment variable holding the simulated "now" for operational replay.
AS_OF_ENV = "AVACHAIN_AS_OF"


def replay_as_of(explicit: "str | _dt.datetime | None" = None) -> Optional[_dt.datetime]:
    """Simulated UTC 'now' for operational replay, or None in live operation.

    Resolution order: the explicit argument, then $AVACHAIN_AS_OF. The env var
    is the harness's lever because the operational shell scripts spawn many
    Python subprocesses, and an exported variable reaches all of them without
    threading a flag through every call site.

    Returns a naive datetime; callers that need tz-awareness treat it as UTC.
    """
    val = explicit if explicit is not None else os.environ.get(AS_OF_ENV)
    if val is None or val == "":
        return None
    if isinstance(val, _dt.datetime):
        return val
    return _dt.datetime.fromisoformat(str(val))


@dataclass
class ProjectConfig:
    # --- Paths ---
    project_dir: Path = Path(".")
    dem_path: Path = Path("data/dem/251110_Professor_Ground-DSM_aligned.tiff")
    survey_dir: Path = Path("data/surveys")
    weather_csv: Path = Path("data/weather/weather_data.csv")
    boundary_kml: Path = Path("data/boundaries/Little_Proff.kml")
    start_zone_kml: Path = Path("data/boundaries/Litte_prof_start_zone.kml")
    output_dir: Path = Path("outputs")
    windninja_library_dir: Path = Path("windninja/library")

    # --- Per-slope runtime directory (outside the repo) ---
    # Source-controlled slope config lives in slopes/<slope_name>/;
    # runtime data (input/smet/, output/*.pro, output/*.zarr) lives here.
    # Default points to the legacy in-repo location so existing runs
    # keep working until the data is physically moved to /data/snowpack/.
    slope_name: str = "little_prof"
    slope_dir: Path = Path("snowpack/little_prof")

    # --- Station list (full dicts from [[stations]] in slope_config.toml) ---
    # Used by sql_util.stations_from_cfg() to know which SQL columns to fetch.
    # Each entry: {role, sql_id, name, lat, lon, elev_m, sql_columns}
    stations: list = field(default_factory=lambda: [
        {"role": "summit", "sql_id": "CAABT", "name": "A-Basin SA-Summit",
         "lat": 39.6424, "lon": -105.8718, "elev_m": 3798.3,
         "sql_columns": ["swin", "temp", "dewp", "rh", "wspd", "wdir",
                         "gust", "mxtemp24h", "mntemp24h"]},
        {"role": "base", "sql_id": "CAABM", "name": "A-Basin SA-Base",
         "lat": 39.6424, "lon": -105.8718, "elev_m": 3554.0,
         "sql_columns": ["pcpac", "depth", "snow24h"]},
    ])

    # --- WRF forecast forcing ---
    # Closest WRF grid cell; populated by install.py from the [wrf] TOML section.
    wrf_smet_file: Optional[Path] = None   # aspect-matched .smet file for this slope
    wrf_smet_dir:  Optional[Path] = None   # grid-cell directory (all aspect variants)
    release_geojson: Path = Path("data/boundaries/avalanche_release_area.geojson")

    # --- Scenario defaults ---
    default_snapshot: str = "2026-01-18"
    n_triggers: int = 5
    max_bfs_clusters: int = 2000
    size_factors: list = field(default_factory=lambda: [0.70, 0.85, 1.00, 1.15, 1.30])
    depth_percentiles: list = field(default_factory=lambda: [10, 50, 90])
    depth_scales: dict = field(default_factory=lambda: {10: 0.75, 50: 1.00, 90: 1.30})
    mu: float = 0.155           # Voellmy dry friction (uncalibrated default)
    xi: float = 1500.0          # Voellmy turbulent friction m/s² (uncalibrated default)
    stauchwall_deg: float = 29.0   # Ron, 2026-10-27 (was 28.0)

    # --- Survey file pattern ---
    # Expected format: YYMMDD_*_snowHeight.tif
    # The date prefix is extracted automatically
    survey_glob: str = "*_snowHeight.tif"
    # [surveys] markers: a survey counts only once its producer's marker says
    # so (survey_markers.py: RUNNING or DONE). Off for trees whose surveys
    # carry no markers (dev, replays); on in a deployment.
    survey_markers: bool = False
    bare_ground_date: str = "251126"  # YYMMDD of the bare ground DEM

    # --- Station metadata ---
    summit_id: str = "CAABT"
    summit_lat: float = 39.6424
    summit_lon: float = -105.8718
    summit_alt_m: float = 3798.3

    base_id: str = "CAABM"
    base_lat: float = 39.6424
    base_lon: float = -105.8718
    base_alt_m: float = 3554.0

    # --- Processing parameters ---
    target_resolution_m: float = 1.0
    flight_hour_utc: int = 18
    sx_search_distance_m: float = 300.0
    sx_azimuths_deg: list = field(default_factory=lambda: [
        0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
        180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5
    ])
    transport_smoothing_window_m: int = 15
    min_valid_hs_m: float = 0.0
    max_valid_hs_m: float = 12.0
    # Clean survey pixels before resampling to 1 m (spatial_model.
    # load_and_resample_survey: drop pixels outside [-0.5 m, max], median per
    # cell, NaN under 50% valid). False resamples with the plain mean and
    # clips, as before 1e71ebb. [processing] clean_survey_pixels in the TOML;
    # off only for ablations.
    clean_survey_pixels: bool = True

    # --- Domain mask ---
    min_slope_deg: float = 15.0

    # --- Avalanche / scour event detection ---
    frac_loss_threshold: float = 0.60   # start-zone loss fraction above which event is settlement/melt
    min_scour_depth_m: float = 0.05     # minimum ΔHS loss to count a cell as "losing" snow
    caic_obs_window_days: int = 3       # ± days around period midpoint to search CAIC API
    caic_spatial_buffer_m: float = 200.0  # buffer around KML polygon for CAIC obs matching

    # --- Clustering ---
    target_cells_per_cluster: int = 50  # auto-determines initial cluster count
    max_cells_per_cluster: int = 20     # recursively split clusters larger than this
    min_cluster_size: int = 4       # minimum cluster size to avoid excessive splitting
    n_pca_components: float = 0.99  # Explained variance threshold for PCA dimensionality reduction before clustering
    n_clusters_override: Optional[int] = None  # set to force a specific initial count
    max_cluster_std_m: float = 0.08  # 8 cm
    # The slope_config.toml this was loaded from, so orchestrators can pass
    # --toml on to subprocesses (None for the default config).
    toml_path: Optional[Path] = None
    # Known-avalanche-events file ([paths] avalanche_events). The avalanche
    # step reads it as manual labels AND appends its auto-detected events to
    # it, so a replay must not share the live slope's copy. None = the
    # historical default, <boundaries_dir>/avalanche_events.json.
    avalanche_events_file: Optional[Path] = None
    # Validation release polygons ([paths] validation_dir), named like the
    # boundaries_dir ones (avalanche_release_area_<YYYYMMDD>.geojson). A file
    # here replaces boundaries_dir's for the same date in
    # release_geojsons_for_date, i.e. for everything a model is scored
    # against, but never in event_mask_for: reinit keeps the mapped crown.
    # None (the default) = off, so a blind replay slope cannot pick it up.
    validation_dir: Optional[Path] = None
    # Per-directory overrides ([paths] <name> for each name in OVERRIDABLE_DIRS).
    # Unset keys keep the historical layout under output_dir, so the dev tree
    # is unchanged; a deployment points them at work/, handoff/, outputs/ and
    # logs/ separately, which fixed sub-paths of output_dir cannot express.
    dir_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        """Resolve paths relative to project_dir."""
        self.dem_path = self.project_dir / self.dem_path
        self.survey_dir = self.project_dir / self.survey_dir
        self.weather_csv = self.project_dir / self.weather_csv
        self.boundary_kml  = self.project_dir / self.boundary_kml
        self.start_zone_kml = self.project_dir / self.start_zone_kml
        self.output_dir = self.project_dir / self.output_dir
        self.windninja_library_dir = self.project_dir / self.windninja_library_dir
        self.slope_dir = self.project_dir / self.slope_dir
        self.release_geojson = self.project_dir / self.release_geojson
        if self.avalanche_events_file is not None:
            self.avalanche_events_file = self.project_dir / self.avalanche_events_file
        if self.validation_dir is not None:
            self.validation_dir = self.project_dir / self.validation_dir
        unknown = set(self.dir_overrides) - set(OVERRIDABLE_DIRS)
        if unknown:
            raise ValueError(f"unknown [paths] directory key(s): {sorted(unknown)}")
        self.dir_overrides = {k: self.project_dir / Path(v)
                              for k, v in self.dir_overrides.items()}

    def _dir(self, name: str, default: Path) -> Path:
        return self.dir_overrides.get(name, default)

    @property
    def boundaries_dir(self) -> Path:
        return self.boundary_kml.parent

    @property
    def weather_cache_dir(self) -> Path:
        """Shared weather-station parquet cache — not slope-namespaced."""
        return self._dir("weather_cache_dir", self.project_dir / "outputs" / "weather")

    @property
    def avalanche_events_path(self) -> Path:
        return self.avalanche_events_file or self.boundaries_dir / "avalanche_events.json"

    @property
    def event_masks_dir(self) -> Path:
        """Auto-detected event masks (avalanche_release_area_<YYYYMMDD>.geojson).

        Slope-namespaced output, unlike boundaries_dir, which holds hand-drawn
        and observed polygons (the Jan 18 crown among them). The avalanche step
        used to write its masks there, where a replay's masks would have been
        read back by a later live reinit as if they were observed.
        """
        return self._dir("event_masks_dir", self.output_dir / "event_masks")

    def event_mask_for(self, event_date) -> Optional[Path]:
        """Release polygon for an event: an observed/hand-drawn one in
        boundaries_dir wins over this slope's auto-detected mask."""
        name = f"avalanche_release_area_{event_date:%Y%m%d}.geojson"
        for d in (self.boundaries_dir, self.event_masks_dir):
            if (d / name).exists():
                return d / name
        return None

    def release_geojsons_for_date(self, snapshot_date: str) -> list:
        """
        Return all avalanche_release_area_{YYYYMMDD}.geojson paths whose
        embedded date is ≤ snapshot_date, sorted chronologically.

        A validation_dir file wins over the boundaries_dir file of the same
        date. This is the validation seam: the comparison plot, the
        release/adjacent/reference groups, --use-observed-release and the
        energy experiments all read the observed polygon through here.

        snapshot_date : YYYY-MM-DD string
        """
        from datetime import date as _date
        try:
            snap = _date.fromisoformat(snapshot_date)
        except ValueError:
            return []
        by_name = {p.name: p for p in self.boundaries_dir.glob(
            "avalanche_release_area_????????.geojson")}
        if self.validation_dir is not None:
            by_name.update({p.name: p for p in self.validation_dir.glob(
                "avalanche_release_area_????????.geojson")})
        paths = []
        for p in (by_name[n] for n in sorted(by_name)):
            date_str = p.stem.replace("avalanche_release_area_", "")
            if len(date_str) == 8:
                try:
                    ev_date = _date(int(date_str[:4]),
                                    int(date_str[4:6]),
                                    int(date_str[6:8]))
                    if ev_date <= snap:
                        paths.append(p)
                except ValueError:
                    continue
        return paths

    @property
    def analysis_dir(self) -> Path:
        return self._dir("analysis_dir", self.output_dir / "analysis")

    @property
    def plots_dir(self) -> Path:
        return self._dir("plots_dir", self.output_dir / "plots")
    

    @property
    def smet_dir(self) -> Path:
        # SMETs live under the slope runtime dir (SNOWPACK input), not outputs/.
        # Falls back to outputs/smet/ if the slope input dir doesn't exist yet
        # so existing runs keep working before the data is physically moved.
        candidate = self.slope_dir / "input" / "smet"
        return candidate if candidate.parent.exists() else self.output_dir / "smet"

    @property
    def grids_dir(self) -> Path:
        return self._dir("grids_dir", self.output_dir / "hourly_grids")

    @property
    def resampled_dir(self) -> Path:
        return self._dir("resampled_dir", self.output_dir / "resampled_1m")

    @property
    def pro_dir(self) -> Path:
        return self.slope_dir / "output"

    @property
    def zarr_path(self) -> Path:
        return self.pro_dir / "slope_snowpack.zarr"

    @property
    def forecast_zarr_path(self) -> Path:
        """NWP-forced profiles after T_stable, rebuilt every cycle.

        Kept apart from zarr_path so a forecast step can never be read as
        analysis: the main store only ever holds observation-driven profiles.
        """
        return self.pro_dir / "forecast_snowpack.zarr"

    @property
    def slopes_config_dir(self) -> Path:
        return self.project_dir / "slopes" / self.slope_name

    @property
    def scenarios_dir(self) -> Path:
        return self._dir("scenarios_dir", self.output_dir / "scenarios")

    @property
    def models_dir(self) -> Path:
        return self._dir("models_dir", self.output_dir / "models")

    @property
    def logs_dir(self) -> Path:
        return self._dir("logs_dir", self.output_dir / "logs")

    @property
    def dem_1m_path(self) -> Path:
        return self.resampled_dir / "dem_1m.tif"

    @property
    def cluster_map_path(self) -> Path:
        return self.analysis_dir / "cluster_map.npy"

    @property
    def cluster_map_tif_path(self) -> Path:
        return self.analysis_dir / "cluster_map.tif"

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [self.output_dir, self.analysis_dir, self.plots_dir,
                  self.smet_dir, self.grids_dir, self.resampled_dir,
                  self.scenarios_dir, self.models_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_toml(cls, toml_path: "str | Path") -> "ProjectConfig":
        """Load a ProjectConfig from a slope_config.toml file."""
        import tomllib
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        paths     = data.get("paths", {})
        slope     = data.get("slope", {})
        scenarios = data.get("scenarios", {})
        processing = data.get("processing", {})
        surveys   = data.get("surveys", {})
        wrf       = data.get("wrf", {})
        snowpack  = data.get("snowpack", {})
        wrf_smet  = resolve_wrf_smet(wrf, snowpack.get("season_start"))
        stations  = {s["role"]: s for s in data.get("stations", [])}
        summit    = stations.get("summit", {})
        base      = stations.get("base", {})

        slope_name = slope.get("name", "slope")

        return cls(
            toml_path=Path(toml_path).resolve(),
            project_dir=Path(paths.get("project_dir", ".")),
            dem_path=Path(paths.get("dem", "data/dem/dem.tif")),
            survey_dir=Path(paths.get("survey_dir", "data/surveys")),
            weather_csv=Path(paths.get("weather_csv", "data/weather/weather_data.csv")),
            boundary_kml=Path(paths.get("boundary_kml", "data/boundaries/boundary.kml")),
            start_zone_kml=Path(paths.get("start_zone_kml", "data/boundaries/start_zone.kml")),
            release_geojson=Path(paths.get("release_geojson",
                                           "data/boundaries/avalanche_release_area.geojson")),
            output_dir=Path(paths.get("output_dir", "outputs")),
            avalanche_events_file=(Path(paths["avalanche_events"])
                                   if "avalanche_events" in paths else None),
            validation_dir=(Path(paths["validation_dir"])
                            if "validation_dir" in paths else None),
            windninja_library_dir=Path(paths.get("windninja_library_dir", "windninja/library")),
            clean_survey_pixels=bool(processing.get("clean_survey_pixels", True)),
            survey_markers=bool(surveys.get("markers", False)),
            slope_name=slope_name,
            slope_dir=Path(paths.get("slope_dir", f"snowpack/{slope_name}")),
            summit_id=summit.get("sql_id", ""),
            summit_lat=float(summit.get("lat", 0.0)),
            summit_lon=float(summit.get("lon", 0.0)),
            summit_alt_m=float(summit.get("elev_m", 0.0)),
            base_id=base.get("sql_id", ""),
            base_lat=float(base.get("lat", 0.0)),
            base_lon=float(base.get("lon", 0.0)),
            base_alt_m=float(base.get("elev_m", 0.0)),
            n_triggers=int(scenarios.get("n_triggers", 5)),
            max_bfs_clusters=int(scenarios.get("max_bfs_clusters", 2000)),
            size_factors=list(scenarios.get("size_factors", [0.70, 0.85, 1.00, 1.15, 1.30])),
            depth_percentiles=list(scenarios.get("depth_percentiles", [10, 50, 90])),
            stauchwall_deg=float(scenarios.get("stauchwall_deg", 29.0)),
            stations=data.get("stations", []),
            wrf_smet_file=wrf_smet,
            wrf_smet_dir=wrf_smet.parent if wrf_smet else None,
            dir_overrides={k: paths[k] for k in OVERRIDABLE_DIRS if k in paths},
        )
            

def shell_paths(cfg: ProjectConfig) -> dict:
    """Directories the shell scripts need, as AVA_<NAME> -> absolute path.

    The scripts used to rebuild these from $PROJECT_DIR/outputs/<slope>/...,
    which is wrong as soon as a TOML relocates a directory. They eval
    `config.py --toml X --print-paths` instead, so the TOML stays the one
    source of truth for where things live.
    """
    return {
        "AVA_SLOPE_NAME":    cfg.slope_name,
        "AVA_SLOPE_DIR":     cfg.slope_dir,
        "AVA_OUTPUT_DIR":    cfg.output_dir,
        "AVA_SURVEY_DIR":    cfg.survey_dir,
        "AVA_SURVEY_MARKERS": int(cfg.survey_markers),
        "AVA_ANALYSIS_DIR":  cfg.analysis_dir,
        "AVA_PLOTS_DIR":     cfg.plots_dir,
        "AVA_SCENARIOS_DIR": cfg.scenarios_dir,
        "AVA_LOGS_DIR":      cfg.logs_dir,
        "AVA_WEATHER_CACHE_DIR": cfg.weather_cache_dir,
        "AVA_PRO_DIR":       cfg.pro_dir,
        "AVA_ZARR_PATH":     cfg.zarr_path,
    }


if __name__ == "__main__":
    import argparse
    import shlex

    ap = argparse.ArgumentParser(description="Inspect a slope configuration.")
    ap.add_argument("--toml", required=True, help="slope_config.toml")
    ap.add_argument("--print-paths", action="store_true",
                    help="print AVA_*=<path> lines for `eval` in a shell script")
    args = ap.parse_args()
    if args.print_paths:
        for k, v in shell_paths(ProjectConfig.from_toml(args.toml)).items():
            print(f"{k}={shlex.quote(str(Path(v).resolve() if isinstance(v, Path) else v))}")
