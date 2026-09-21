"""
Project configuration for distributed SNOWPACK forcing generation.

For multi-slope deployments, use ProjectConfig.from_toml() instead of editing
this file directly.  See slopes/template/slope_config.toml for the format.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    size_factors: list = field(default_factory=lambda: [0.70, 0.85, 1.00, 1.15, 1.30])
    depth_percentiles: list = field(default_factory=lambda: [10, 50, 90])
    depth_scales: dict = field(default_factory=lambda: {10: 0.75, 50: 1.00, 90: 1.30})
    mu: float = 0.155           # Voellmy dry friction (uncalibrated default)
    xi: float = 1500.0          # Voellmy turbulent friction m/s² (uncalibrated default)
    stauchwall_deg: float = 28.0

    # --- Survey file pattern ---
    # Expected format: YYMMDD_*_snowHeight.tif
    # The date prefix is extracted automatically
    survey_glob: str = "*_snowHeight.tif"
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

    @property
    def boundaries_dir(self) -> Path:
        return self.boundary_kml.parent

    @property
    def weather_cache_dir(self) -> Path:
        """Shared weather-station parquet cache — not slope-namespaced."""
        return self.project_dir / "outputs" / "weather"

    @property
    def avalanche_events_path(self) -> Path:
        return self.boundaries_dir / "avalanche_events.json"

    def release_geojsons_for_date(self, snapshot_date: str) -> list:
        """
        Return all avalanche_release_area_{YYYYMMDD}.geojson paths whose
        embedded date is ≤ snapshot_date, sorted chronologically.

        snapshot_date : YYYY-MM-DD string
        """
        from datetime import date as _date
        try:
            snap = _date.fromisoformat(snapshot_date)
        except ValueError:
            return []
        paths = []
        for p in sorted(self.boundaries_dir.glob(
                "avalanche_release_area_????????.geojson")):
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
        return self.output_dir / "analysis"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"
    

    @property
    def smet_dir(self) -> Path:
        # SMETs live under the slope runtime dir (SNOWPACK input), not outputs/.
        # Falls back to outputs/smet/ if the slope input dir doesn't exist yet
        # so existing runs keep working before the data is physically moved.
        candidate = self.slope_dir / "input" / "smet"
        return candidate if candidate.parent.exists() else self.output_dir / "smet"

    @property
    def grids_dir(self) -> Path:
        return self.output_dir / "hourly_grids"

    @property
    def resampled_dir(self) -> Path:
        return self.output_dir / "resampled_1m"

    @property
    def pro_dir(self) -> Path:
        return self.slope_dir / "output"

    @property
    def zarr_path(self) -> Path:
        return self.pro_dir / "slope_snowpack.zarr"

    @property
    def slopes_config_dir(self) -> Path:
        return self.project_dir / "slopes" / self.slope_name

    @property
    def scenarios_dir(self) -> Path:
        return self.output_dir / "scenarios"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

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
        wrf       = data.get("wrf", {})
        stations  = {s["role"]: s for s in data.get("stations", [])}
        summit    = stations.get("summit", {})
        base      = stations.get("base", {})

        slope_name = slope.get("name", "slope")

        return cls(
            project_dir=Path(paths.get("project_dir", ".")),
            dem_path=Path(paths.get("dem", "data/dem/dem.tif")),
            survey_dir=Path(paths.get("survey_dir", "data/surveys")),
            weather_csv=Path(paths.get("weather_csv", "data/weather/weather_data.csv")),
            boundary_kml=Path(paths.get("boundary_kml", "data/boundaries/boundary.kml")),
            start_zone_kml=Path(paths.get("start_zone_kml", "data/boundaries/start_zone.kml")),
            release_geojson=Path(paths.get("release_geojson",
                                           "data/boundaries/avalanche_release_area.geojson")),
            output_dir=Path(paths.get("output_dir", "outputs")),
            windninja_library_dir=Path(paths.get("windninja_library_dir", "windninja/library")),
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
            size_factors=list(scenarios.get("size_factors", [0.70, 0.85, 1.00, 1.15, 1.30])),
            depth_percentiles=list(scenarios.get("depth_percentiles", [10, 50, 90])),
            stauchwall_deg=float(scenarios.get("stauchwall_deg", 28.0)),
            stations=data.get("stations", []),
            wrf_smet_file=Path(wrf["wrf_smet_file"]) if wrf.get("wrf_smet_file") else None,
            wrf_smet_dir= Path(wrf["wrf_smet_dir"])  if wrf.get("wrf_smet_dir")  else None,
        )
            