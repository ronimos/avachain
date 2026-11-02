"""
avalanche_simulation.py

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Per-scenario simulation pipeline. For one scenario:
        1. Build an AvaFrame-layout directory under scenarios/scenario_NNN/avaframe/
        2. Auto-detect the entrainment area (in avalanche_helpers)
        3. Save the steepest-descent flowline for the viewer
        4. Write the per-scenario com1DFA config (.ini)
        5. Run AvaFrame com1DFA through its Python API
        6. Copy peak rasters into scenario outputs/
        7. Build the reach mask (flow depth >= threshold)
        8. Check whether the flow reaches the road
        9. Write summary.json

    No command line. Imported by avalanche_runout.py.

Requirements:
    - avaframe >= 2.1 (install separately per AvaFrame docs)
    - rasterio, fiona, geopandas, shapely, numpy
"""

# --------- Load Libraries --------
import os
import json
import time
import logging
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import LineString, Point

import avalanche_helpers as h


# com1DFA friction models (AvaFrame 2.1) and the .ini keys each one reads.
# A model ignores any parameter it has no key for.
# samosATAuto picks a samosAT variant at run time, so values go to all three.
# spatialVoellmy reads mu/xi rasters from Inputs/RASTERS instead of the ini.
FRICTION_MODELS = {
    'samosat':         {'mu': ('musamosat',), 'tau0': ('tau0samosat',)},
    'samosatsmall':    {'mu': ('musamosatsmall',), 'tau0': ('tau0samosatsmall',)},
    'samosatmedium':   {'mu': ('musamosatmedium',), 'tau0': ('tau0samosatmedium',)},
    'samosatauto':     {'mu': ('musamosat', 'musamosatsmall', 'musamosatmedium'),
                        'tau0': ('tau0samosat', 'tau0samosatsmall',
                                 'tau0samosatmedium')},
    'voellmy':         {'mu': ('muvoellmy',), 'xi': ('xsivoellmy',)},
    'voellmyminshear': {'mu': ('muvoellmyminshear',),
                        'xi': ('xsivoellmyminshear',),
                        'tau0': ('tau0voellmyminshear',)},
    'coulomb':         {'mu': ('mucoulomb',)},
    'coulombminshear': {'mu': ('mucoulombminshear',),
                        'tau0': ('tau0coulombminshear',)},
    'wetsnow':         {'mu': ('mu0wetsnow',), 'xi': ('xsiwetsnow',)},
    'spatialvoellmy':  {},
}

# Models that take per-scenario mu/xi from params.json when no config
# override is given. samosAT-family models keep their internal calibration
# unless friction.mu is set on purpose in the config.
PARAMS_MU_XI_MODELS = ('voellmy', 'voellmyminshear', 'coulomb',
                       'coulombminshear', 'wetsnow')


def materialize_ava_dir(scenario, shared_paths, config, preserve_outputs=False):
    # Build the AvaFrame Inputs/ folder for one scenario
    # @param scenario: dict with scenario_id, scenario_dir, release_geojson,
    #                  depth_raster, density, params
    # @param shared_paths: dict with 'dem', 'forest', 'road' Path objects
    # @param config: top-level config dict
    # @param preserve_outputs: If True, keep avaframe/Outputs from a prior
    #                          run and only rebuild Inputs (post-only mode)
    # @returns: tuple (ava_dir_path, entrainment_info_dict_or_None)
    # wipes any prior avaframe/ subdirectory before rebuilding,
    #       except Outputs when preserve_outputs is True
    
    sid = scenario['scenario_id']
    # All per-scenario writes go to work_dir. Defaults to the scenario
    # folder itself (original layout). Set scenario['work_dir'] to keep
    # the scenario input folder read-only (forecast runs).
    work_dir = Path(scenario.get('work_dir') or scenario['scenario_dir'])
    ava_dir = work_dir / 'avaframe'
    
    # Wipe and rebuild. In post-only mode the com1DFA Outputs from the
    # completed run are the whole point, so only Inputs and Work go.
    if ava_dir.exists():
        import shutil
        if preserve_outputs:
            for sub in ('Inputs', 'Work'):
                if (ava_dir / sub).exists():
                    shutil.rmtree(ava_dir / sub)
        else:
            shutil.rmtree(ava_dir)
    inputs_dir = ava_dir / 'Inputs'
    for sub in ('REL', 'RELTH', 'RES'):
        (inputs_dir / sub).mkdir(parents=True)
    
    # Per-scenario outputs dir (created early so flowpath has somewhere to go)
    scenario_outputs = work_dir / 'outputs'
    scenario_outputs.mkdir(parents=True, exist_ok=True)
    
    # ---- DEM (shared across scenarios) ----
    h.convert_raster_format(shared_paths['dem'], inputs_dir / 'DEM.tif')
    
    # ---- Release polygon: GeoJSON -> Shapefile ----
    # AvaFrame requires a 'thickness' attribute on every release feature
    # even when the raster supplies the real values (relThFromFile=True).
    rel_thickness = scenario['params'].get(
        'mean_depth_m',
        config['defaults']['scour_depth_fallback_m']
    )
    
    # Check for and log holes
    release_check = gpd.read_file(scenario['release_geojson'])
    n_holes = sum(
        len(g.interiors) for g in release_check.geometry
        if g is not None and g.geom_type == 'Polygon'
    )
    if n_holes > 0:
        logging.info(
            f"  Release polygon has {n_holes} hole(s) - filled for com1DFA. "
            "(AvaFrame rejects polygons with interior rings.)"
        )
    
    h.geojson_to_shapefile(
        scenario['release_geojson'],
        inputs_dir / 'REL' / 'release_01.shp',
        attributes={'thickness': float(rel_thickness)},
        fill_holes=True
    )
    
    # Save the filled polygon as GeoJSON in scenario outputs/ for the viewer
    
    from shapely.geometry import Polygon, MultiPolygon
    rel_filled_gdf = gpd.read_file(scenario['release_geojson'])
    filled_geoms = []
    for geom in rel_filled_gdf.geometry:
        if geom is None or geom.is_empty:
            filled_geoms.append(geom)
        elif geom.geom_type == 'Polygon':
            filled_geoms.append(Polygon(geom.exterior.coords))
        elif geom.geom_type == 'MultiPolygon':
            filled_geoms.append(MultiPolygon(
                [Polygon(p.exterior.coords) for p in geom.geoms]
            ))
        else:
            filled_geoms.append(geom)
    rel_filled_gdf = rel_filled_gdf.set_geometry(filled_geoms)
    rel_filled_gdf.to_file(
        scenario_outputs / 'release_filled.geojson', driver='GeoJSON'
    )
    
    # ---- Release thickness raster ----
    # DEM and RELTH must share an identical grid. Resample the upstream
    # depth raster onto the DEM grid; cells outside the source get 0.
    h.resample_to_reference_grid(
        src_path=scenario['depth_raster'],
        ref_path=inputs_dir / 'DEM.tif',
        dst_path=inputs_dir / 'RELTH' / 'release_01.asc',
        resampling_method='bilinear',
        fill_value=0.0
    )
    
    # ---- Forest as resistance area ----
    # Clip to the DEM valid-data footprint (com1DFA aborts on features
    # outside the DEM). fill_holes: com1DFA rejects features with interior
    # rings. Filling small canopy gaps treats them as forest, which errs
    # toward stopping flow.
    h.geojson_to_shapefile(
        shared_paths['forest'],
        inputs_dir / 'RES' / 'forest.shp',
        clip_to_raster=inputs_dir / 'DEM.tif',
        fill_holes=True,
        explode_multipart=True
    )
    
    # ---- Auto-detect entrainment (only if sim_type='ent') ----
    # Always save the flowpath even if not running entrainment, so the
    # viewer can show the path for every scenario regardless of sim_type
    ent_info = None
    flowpath_out = scenario_outputs / 'flowpath.geojson'
    
    # Load the release polygon (single feature) once for downstream use
    release_gdf = gpd.read_file(scenario['release_geojson'])
    release_polygon = release_gdf.geometry.iloc[0]
    
    if 'ent' in config['com1dfa']['sim_type']:
        (inputs_dir / 'ENT').mkdir()
        (inputs_dir / 'ENTH').mkdir()
        
        ab_cfg = config['entrainment']['auto_detect']['alpha_beta']
        fa_cfg = config['entrainment']['auto_detect']['flow_accumulation']
        
        # Determine entrainment depth (scour cap) for this scenario.
        # Priority order:
        #   1. params.json scour_depth_m (if > 0 - real per-scenario value)
        #   2. config.defaults.scour_depth_fallback_m (when scour=0; happens
        #      on hard slabs where the failure-plane-to-first-hard-layer
        #      measurement misses the entrainable bed below the slab)
        defaults_cfg = config['defaults']
        scour_from_params = scenario['params'].get('scour_depth_m', 0.0)
        if scour_from_params and scour_from_params > 0:
            entrainment_depth = float(scour_from_params)
            ent_source = 'params.json (per-scenario)'
        else:
            entrainment_depth = float(defaults_cfg['scour_depth_fallback_m'])
            ent_source = 'config defaults (fallback - scour=0 in params)'
        
        logging.info(
            f"  [{scenario['scenario_id']}] entrainment depth (scour cap) = "
            f"{entrainment_depth:.2f} m from {ent_source}"
        )
        
        # Entrainment mode (set by the profile):
        #   auto     corridor along the flowline at the scour cap
        #   raster   corridor with per-cell min(HS, scour cap) from the
        #            snow height raster
        #   polygon  hand-drawn polygon from the path profile at the cap
        # A config without a mode key falls back on use_snow_height_dsm.
        ent_cfg_top = config['entrainment']
        ent_mode = ent_cfg_top.get('mode') or (
            'raster' if ent_cfg_top.get('use_snow_height_dsm') else 'auto'
        )
        min_slope = float(ent_cfg_top.get('min_slope_deg', 0.0))

        if ent_mode == 'polygon':
            ent_info = h.build_polygon_entrainment(
                polygon_path=ent_cfg_top['polygon'],
                release_polygon=release_polygon,
                dem_path=inputs_dir / 'DEM.tif',
                polygon_out=inputs_dir / 'ENT' / 'entrainment.shp',
                thickness_out=inputs_dir / 'ENTH' / 'entrainment.asc',
                crs=h.get_crs_of(shared_paths['dem']),
                depth_m=entrainment_depth,
                min_slope_deg=min_slope,
                flowline_out=flowpath_out,
            )
        else:
            snow_height_path = None
            if ent_mode == 'raster':
                sh_path = ent_cfg_top.get('snow_height_raster')
                if sh_path and Path(sh_path).exists():
                    snow_height_path = Path(sh_path)
                    logging.info(
                        f"  [{scenario['scenario_id']}] using snow height DSM: "
                        f"{snow_height_path.name}"
                    )
                else:
                    logging.warning(
                        f"  [{scenario['scenario_id']}] entrainment mode is "
                        f"raster but snow_height_raster is missing or not "
                        f"found ({sh_path}). Falling back to constant scour cap."
                    )

            ent_info = h.auto_detect_entrainment(
                release_polygon=release_polygon,
                dem_path=inputs_dir / 'DEM.tif',
                polygon_out=inputs_dir / 'ENT' / 'entrainment.shp',
                thickness_out=inputs_dir / 'ENTH' / 'entrainment.asc',
                crs=h.get_crs_of(shared_paths['dem']),
                depth_m=entrainment_depth,
                alpha_min=ab_cfg['min_alpha_deg'],
                alpha_max=ab_cfg['max_alpha_deg'],
                alpha_buffer_m=ab_cfg.get('buffer_m', 15.0),
                fallback_buffer_m=fa_cfg['buffer_m'],
                flowline_out=flowpath_out,
                snow_height_raster=snow_height_path,
                min_slope_deg=min_slope
            )
        # Annotate ent_info with source so the report can reference it
        ent_info['mode'] = ent_mode
        ent_info['depth_source'] = ent_source
        ent_info['scour_depth_from_params'] = scour_from_params
    else:
        # No entrainment - still trace and save the flowline for the viewer
        flowline = h.steepest_descent_flowline(
            release_polygon, inputs_dir / 'DEM.tif'
        )
        flow_gdf = gpd.GeoDataFrame(
            {'name': ['flowpath']}, geometry=[flowline],
            crs=h.get_crs_of(shared_paths['dem'])
        )
        flow_gdf.to_file(flowpath_out, driver='GeoJSON')
    
    return ava_dir, ent_info


def write_com1dfa_config(scenario, config, out_path):
    # Write the per-scenario com1DFA config (.ini)
    # @param scenario: scenario dict (with params, density)
    # @param config: top-level config dict
    # @param out_path: Path for the .ini file
    # @returns: Path to written config
    # @raises ValueError: if the friction model is not a known com1DFA model

    cfg_cfg = config['com1dfa']
    ent_cfg = config['entrainment']
    defaults_cfg = config['defaults']
    friction_cfg = config.get('friction', {}) or {}

    # Density: prefer density.json mean, fall back to params.json rho_kgm3
    rho = scenario['density'].get(
        'mean_kgm3', scenario['params'].get('rho_kgm3', 300.0)
    )

    # Entrainment thickness scalar. com1DFA requires 'entTh' to parse as a
    # float even when the raster route (entThFromFile) supplies the real
    # per-cell values.
    scour_from_params = scenario['params'].get('scour_depth_m', 0.0)
    if scour_from_params and scour_from_params > 0:
        ent_thickness_placeholder = float(scour_from_params)
    else:
        ent_thickness_placeholder = float(defaults_cfg['scour_depth_fallback_m'])

    # AvaFrame reads case-sensitive camelCase keys. Python's ConfigParser
    # lowercases option names on write, which makes AvaFrame silently fall
    # back to defaults. Subclass to preserve case.
    class _CaseSensitiveCfg(ConfigParser):
        optionxform = staticmethod(str)

    cp = _CaseSensitiveCfg()
    cp['GENERAL'] = {
        'simTypeList': cfg_cfg['sim_type'],
        'frictModel': cfg_cfg['friction_model'],
        'rho': str(rho),
        'meshCellSize': str(cfg_cfg['mesh_cell_size_m']),
        'stopCrit': str(cfg_cfg['stop_crit']),
        # AvaFrame default tEnd of 400 s is too short for long low-friction
        # paths - flow truncates with the tail still moving.
        'tEnd': str(cfg_cfg.get('t_end_s', 1200)),
        'resType': '|'.join(cfg_cfg['result_types']),
        # Release thickness comes from the Inputs/RELTH raster.
        # relTh must stay unset when relThFromFile is True, and
        # relThFromShp must be False (com1DFA defaults it to True).
        'relThFromFile': 'True',
        'relThFromShp': 'False',
        # No secondary release areas in this workflow.
        'secRelArea': 'False',
        # Wipe stale remeshed rasters from any previous run.
        'cleanRemeshedRasters': 'True',
    }

    # ---- Friction parameters ----
    # Values in config['friction'] win when set. When mu/xi are unset,
    # models listed in PARAMS_MU_XI_MODELS take per-scenario mu/xi from
    # params.json; samosAT-family models keep their internal calibration.
    fm = cfg_cfg['friction_model'].lower()
    if fm not in FRICTION_MODELS:
        # throw error if the model name is not a com1DFA friction model
        raise ValueError(
            f"Unknown friction model '{cfg_cfg['friction_model']}'. "
            f"Valid options: {sorted(FRICTION_MODELS.keys())}"
        )
    model_keys = FRICTION_MODELS[fm]

    # friction.sources names where each parameter comes from:
    #   fixed   the config value (null = AvaFrame internal default)
    #   params  per-scenario params.json
    # Without a sources block the old rule applies: a config value wins,
    # otherwise Voellmy-type models take mu/xi from params.json.
    sources = friction_cfg.get('sources') or {}

    def _resolve(name):
        # Pick the value and its source for one friction parameter
        # @param name: 'mu', 'xi' or 'tau0'
        # @returns: tuple (value_or_None, source_label_or_None)
        src = sources.get(name)
        if src == 'params':
            return scenario['params'].get(name), 'params.json'
        if src == 'fixed':
            v = friction_cfg.get(name)
            return (v, 'config') if v is not None else (None, None)
        if friction_cfg.get(name) is not None:
            return friction_cfg[name], 'config'
        if name in ('mu', 'xi') and fm in PARAMS_MU_XI_MODELS:
            return scenario['params'].get(name), 'params.json'
        return None, None

    for name in ('mu', 'xi', 'tau0'):
        if name not in model_keys:
            continue
        value, source = _resolve(name)
        if value is None:
            continue
        for ini_key in model_keys[name]:
            cp['GENERAL'][ini_key] = str(value)
        logging.info(
            f"  [{scenario['scenario_id']}] friction {name}={value} "
            f"({fm}, from {source})"
        )

    if fm not in PARAMS_MU_XI_MODELS and friction_cfg.get('mu') is None:
        logging.info(
            f"  [{scenario['scenario_id']}] '{cfg_cfg['friction_model']}' "
            f"uses its internal calibration; per-scenario mu/xi from "
            f"params.json are not used."
        )

    # ---- Resistance (forest) parameters ----
    # com1DFA only reads the RES/ area when sim_type includes resistance
    # ('res' or 'entres'). Resistance settings work under any friction model.
    # Feistl et al. (2014) band logic per cell each timestep:
    #   FV or FT below min  -> detrainment only
    #   FV and FT in band   -> added friction only (cResH)
    #   FV or FT above max  -> forest overrun, nothing applied
    res_cfg = config.get('resistance')
    if res_cfg and 'res' in cfg_cfg['sim_type']:
        for key in ('cResH', 'detrainment', 'forestVMin', 'forestThMin',
                    'forestVMax', 'forestThMax'):
            if key in res_cfg and res_cfg[key] is not None:
                cp['GENERAL'][key] = str(res_cfg[key])
        logging.info(
            f"  [{scenario['scenario_id']}] resistance active: "
            f"cResH={res_cfg.get('cResH', 'default')}, "
            f"detrainment={res_cfg.get('detrainment', 'default')}"
        )
    elif res_cfg and 'res' not in cfg_cfg['sim_type']:
        logging.warning(
            f"  [{scenario['scenario_id']}] config has a 'resistance' block "
            f"but sim_type='{cfg_cfg['sim_type']}' does not include "
            f"resistance; the forest area will be IGNORED. "
            f"Set sim_type to 'res' or 'entres'."
        )

    # ---- Entrainment keys (only if sim_type includes entrainment) ----
    if 'ent' in cfg_cfg['sim_type']:
        # Raster route (entThFromFile=True): per-cell thickness from the
        # ENTH raster, used when the snow-height DSM is enabled.
        # Shapefile route (entThFromFile=False): single 'entTh' value.
        ent_mode = ent_cfg.get('mode') or (
            'raster' if ent_cfg.get('use_snow_height_dsm') else 'auto'
        )
        use_dsm = ent_mode == 'raster' and bool(
            ent_cfg.get('snow_height_raster') and
            Path(ent_cfg['snow_height_raster']).exists()
        )
        cp['GENERAL']['entThFromFile'] = 'True' if use_dsm else 'False'
        # Thickness never comes from the shapefile attribute. It comes from
        # the ENTH raster (entThFromFile=True) or the entTh scalar. com1DFA
        # defaults entThFromShp to True, which forbids setting entTh at all,
        # so it must be flipped off explicitly.
        cp['GENERAL']['entThFromShp'] = 'False'
        # entTh must be float-parseable in both routes; the raster
        # overrides per-cell when entThFromFile is True.
        cp['GENERAL']['entTh'] = str(ent_thickness_placeholder)
        # Fallback when the shapefile is missing the thickness attribute.
        cp['GENERAL']['entThIfMissingInShp'] = str(
            defaults_cfg['scour_depth_fallback_m']
        )
        # Entrained snow density: use release density
        cp['GENERAL']['rhoEnt'] = str(rho)
        cp['GENERAL']['entEroEnergy'] = str(ent_cfg['ent_ero_energy_jkg'])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        cp.write(f)

    return out_path


def run_com1dfa(ava_dir, local_cfg_path):
    # Run AvaFrame com1DFA on a prepared scenario directory
    # @param ava_dir: Path to the scenario's avaframe/ directory
    # @param local_cfg_path: Path to the per-scenario com1DFA .ini
    # @returns: int - number of simulations completed (typically 1)
    # AvaFrame >= 2.1; config loaded via fileOverride, passed to
    #       com1DFAMain as cfgInfo

    # AVACHAIN_STUB_COM1DFA replaces com1DFA with a stand-in (smoke test)
    if os.environ.get('AVACHAIN_STUB_COM1DFA'):
        return _stub_com1dfa(ava_dir, local_cfg_path)

    # Import inside the function so the module loads without AvaFrame
    # (dry-run validation, post-processing tests)
    from avaframe.in3Utils import cfgUtils
    from avaframe.com1DFA import com1DFA
    
    # Set up the AvaFrame "main" config to point at this scenario's avaDir
    cfg_main = cfgUtils.getGeneralConfig()
    cfg_main['MAIN']['avalancheDir'] = str(ava_dir)
    
    # Load com1DFA config with our per-scenario overrides as a configparser
    # object (not just a side-effect; we need to pass it to com1DFAMain)
    com1dfa_cfg = cfgUtils.getModuleConfig(
        com1DFA, fileOverride=str(local_cfg_path),
        modInfo=False, toPrint=False
    )
    
    # Run the simulation (cfgInfo = our config object)
    dem, plot_dict, report_dict, sim_df = com1DFA.com1DFAMain(
        cfg_main, cfgInfo=com1dfa_cfg
    )
    
    return len(sim_df)


def _stub_com1dfa(ava_dir, local_cfg_path):
    # Stand-in for com1DFA used by the smoke test. Writes a straight band
    # of flow downslope from the release, longer for deeper releases and
    # shorter for higher tau0, as peak rasters where AvaFrame would.
    # @param ava_dir: Path to the scenario's avaframe/ directory
    # @param local_cfg_path: Path to the per-scenario .ini (tau0 read from it)
    # @returns: int - 1
    # Only reached through AVACHAIN_STUB_COM1DFA
    ava_dir = Path(ava_dir)
    with rasterio.open(ava_dir / 'Inputs' / 'DEM.tif') as d:
        prof = d.profile.copy()
        nrow, ncol = d.height, d.width
    with rasterio.open(ava_dir / 'Inputs' / 'RELTH' / 'release_01.asc') as r:
        relth = r.read(1)
    depth = float(relth[relth > 0].mean()) if (relth > 0).any() else 0.5
    rows, cols = np.where(relth > 0)
    r0, c0, c1 = int(rows.max()), max(0, int(cols.min()) - 5), min(ncol, int(cols.max()) + 5)
    tau0 = 125.0
    for line in Path(local_cfg_path).read_text().splitlines():
        if line.strip().lower().startswith('tau0samosatsmall'):
            tau0 = float(line.split('=')[1])
    length = int((120 + 220 * depth) * (125.0 / tau0) ** 0.5)
    r1 = min(nrow - 1, r0 + length)
    pft = np.zeros((nrow, ncol), 'float32')
    pfv = np.zeros_like(pft)
    pft[r0:r1, c0:c1] = np.linspace(1.5, 0.15, r1 - r0)[:, None]
    pfv[r0:r1, c0:c1] = np.linspace(25.0, 1.0, r1 - r0)[:, None]
    ppr = (0.5 * 300 * pfv ** 2 / 1000.0).astype('float32')
    peak = ava_dir / 'Outputs' / 'com1DFA' / 'peakFiles'
    peak.mkdir(parents=True, exist_ok=True)
    prof.update(dtype='float32', count=1, nodata=-9999.0)
    for name, arr in (('pft', pft), ('pfv', pfv), ('ppr', ppr), ('pta', pfv * 0 + 20)):
        with rasterio.open(peak / f'stub_{name}.tif', 'w', **prof) as d:
            d.write(arr.astype('float32'), 1)
    return 1


def find_peak_rasters(ava_dir):
    # Locate com1DFA peak output rasters
    # @param ava_dir: Path to the scenario's avaframe/ directory
    # @returns: dict keyed by result type ('ppr', 'pft', 'pfv', 'pta')
    #          with Path values; missing types are absent
    # @raises FileNotFoundError: if peakFiles/ is missing or empty
    # filename patterns vary across AvaFrame versions, so match on
    #       the _<resType> suffix and pick the newest file per type
    
    peak_dir = Path(ava_dir) / 'Outputs' / 'com1DFA' / 'peakFiles'
    if not peak_dir.exists():
        # throw error if AvaFrame produced no peakFiles dir
        raise FileNotFoundError(f"No peakFiles directory at {peak_dir}")
    
    # Collect all raster files anywhere under peak_dir (recursive)
    all_rasters = list(peak_dir.rglob('*.asc')) + list(peak_dir.rglob('*.tif'))
    
    if not all_rasters:
        # throw error if peakFiles dir exists but is empty
        raise FileNotFoundError(
            f"No raster files found under {peak_dir}. "
            "Check Outputs/com1DFA/ for the actual location of peak fields."
        )
    
    rasters = {}
    for res_type in ('ppr', 'pft', 'pfv', 'pta'):
        # Match final peak files only - filename must END with _<res_type>
        # (e.g. abc_pft.tif). This excludes time-step snapshots in
        # peakFiles/timeSteps/ that look like abc_pft_t0.00.tif.
        suffix = f'_{res_type}'
        matches = [p for p in all_rasters if p.stem.endswith(suffix)]
        if matches:
            # Sort by modification time, newest first
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            rasters[res_type] = matches[0]
            logging.debug(f"  Found {res_type}: {matches[0].name}")
        else:
            logging.warning(f"  No {res_type} peak raster found")
    
    return rasters


def restype_to_human_name(res_type):
    # Map com1DFA result type codes to human-readable names
    # @param res_type: 'ppr', 'pft', 'pfv', or 'pta'
    # @returns: str - readable name (or original code if unknown)
    
    return {
        'ppr': 'pressure',
        'pft': 'flow_depth',
        'pfv': 'velocity',
        'pta': 'travel_angle'
    }.get(res_type, res_type)


def build_reach_mask(pft_path, out_path, depth_threshold_m,
                     pfv_path=None, velocity_threshold_ms=0.0,
                     ppr_path=None, pressure_threshold_kpa=0.0):
    # Build a binary reach mask from peak flow depth, optionally also requiring
    # minimum velocity and/or peak pressure (RAMMS-style hazard thresholds).
    # @param pft_path: Path to peak flow thickness raster
    # @param out_path: Path to write binary reach mask GeoTIFF
    # @param depth_threshold_m: minimum flow depth to count as "reached"
    # @param pfv_path: Optional path to peak velocity (only used if threshold>0)
    # @param velocity_threshold_ms: minimum velocity to count (0 = no constraint)
    # @param ppr_path: Optional path to peak pressure (only used if threshold>0)
    # @param pressure_threshold_kpa: minimum peak pressure to count
    #                                (0 = no constraint). AvaFrame ppr is
    #                                already in kPa; this threshold is kPa too.
    # @returns: Path to written mask
    # output is uint8 (0/1) aligned to the pft grid; all active
    #       thresholds are AND-combined
    
    depth, meta = h.read_raster(pft_path)
    reach = (depth >= depth_threshold_m)
    
    if velocity_threshold_ms > 0 and pfv_path is not None:
        vel, _ = h.read_raster(pfv_path)
        reach = reach & (vel >= velocity_threshold_ms)
    
    if pressure_threshold_kpa > 0 and ppr_path is not None:
        pres, _ = h.read_raster(ppr_path)
        reach = reach & (pres >= pressure_threshold_kpa)
    
    # Write as uint8 GeoTIFF (0/1 binary)
    profile = meta['profile']
    profile['nodata'] = 0
    return h.write_raster(
        out_path, reach.astype('uint8'), profile,
        driver='GTiff', cog=True
    )


def analyze_road_intersection(reach_mask_path, pft_path, pfv_path,
                              road_path, release_polygon_path,
                              road_buffer_m=5.0,
                              chainage_origin='far_from_release'):
    # Did the scenario reach the road? If yes, where, depth, velocity?
    # @param reach_mask_path: Path to binary reach mask
    # @param pft_path: Path to peak flow depth raster (for sampling at impact)
    # @param pfv_path: Path to peak velocity raster (for sampling at impact)
    # @param road_path: Path to road centerline polyline
    # @param release_polygon_path: Path to release polygon (for chainage origin)
    # @param road_buffer_m: buffer width around centerline = "road corridor"
    # @param chainage_origin: 'far_from_release', 'first_vertex', 'last_vertex'
    # @returns: dict with reached, intersection_xy, flow_depth_at_road_m,
    #          velocity_at_road_ms, distance_to_road_m, chainage_m
    # @raises ValueError: if the road file has no usable line geometry
    # distance_to_road_m is positive if the flow fell short, 0 if
    #       it reached; multi-segment polylines are concatenated
    
    # Load and merge road segments
    # Real road shapefiles often have multiple features (one per highway
    # segment) and may use MultiLineString geometry. Flatten everything
    # to a single LineString for chainage and buffering.
    road_gdf = gpd.read_file(road_path)
    road_lines = []
    for geom in road_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        # MultiLineString -> list of LineStrings
        if geom.geom_type == 'MultiLineString':
            road_lines.extend(list(geom.geoms))
        elif geom.geom_type == 'LineString':
            road_lines.append(geom)
        else:
            logging.warning(f"Skipping unexpected road geom type: {geom.geom_type}")
    
    if not road_lines:
        # throw error if road file has no usable geometry
        raise ValueError(f"Road file {road_path} has no usable LineString geometry")
    
    if len(road_lines) == 1:
        road = road_lines[0]
    else:
        # Concatenate all segment coords into a single polyline
        # Order roughly along the road by sorting on first vertex
        road_lines.sort(key=lambda g: (g.coords[0][0], g.coords[0][1]))
        all_coords = []
        for g in road_lines:
            all_coords.extend(list(g.coords))
        road = LineString(all_coords)
    
    # Buffer road -> "road corridor" polygon for the reached test
    corridor = road.buffer(road_buffer_m, cap_style=2)
    
    # Test if reach mask intersects corridor
    reach, meta = h.read_raster(reach_mask_path)
    transform = meta['transform']
    
    # Burn corridor into a mask aligned to reach raster, then intersect
    corridor_mask = h.polygon_to_raster_mask(
        corridor, reach.shape, transform, fill_value=1.0
    )
    overlap = reach.astype(bool) & corridor_mask.astype(bool)
    
    # Compute chainage origin point
    release_gdf = gpd.read_file(release_polygon_path)
    release_centroid = release_gdf.geometry.iloc[0].centroid
    
    if chainage_origin == 'far_from_release':
        # Whichever end of the polyline is farther from the release
        pt_first = Point(road.coords[0])
        pt_last = Point(road.coords[-1])
        if pt_first.distance(release_centroid) < pt_last.distance(release_centroid):
            origin_pt = pt_last
        else:
            origin_pt = pt_first
    elif chainage_origin == 'first_vertex':
        origin_pt = Point(road.coords[0])
    elif chainage_origin == 'last_vertex':
        origin_pt = Point(road.coords[-1])
    else:
        # Default to first vertex on unknown option
        origin_pt = Point(road.coords[0])
    
    # If origin is the last vertex, reverse the line so project() measures
    # from origin going along the road
    if origin_pt.equals(Point(road.coords[-1])):
        road_for_chainage = LineString(list(road.coords)[::-1])
    else:
        road_for_chainage = road
    
    # Case 1: Did NOT reach. Compute shortest distance to road.
    if not overlap.any():
        if reach.any():
            # Flow exists but didn't hit road - measure distance from any
            # reach cell to road centerline
            ys, xs = np.where(reach.astype(bool))
            wx, wy = h.pixels_to_world(ys, xs, transform)
            
            # Subsample for speed if huge
            if len(wx) > 5000:
                idx = np.linspace(0, len(wx) - 1, 5000, dtype=int)
                wx, wy = wx[idx], wy[idx]
            
            min_dist = min(
                road.distance(Point(x, y))
                for x, y in zip(wx, wy)
            )
        else:
            # No reach at all - distance from release centroid to road
            min_dist = float(release_centroid.distance(road))
        
        return {
            'reached': False,
            'intersection_xy': None,
            'flow_depth_at_road_m': None,
            'velocity_at_road_ms': None,
            'distance_to_road_m': float(min_dist),
            'chainage_m': None
        }
    
    # Case 2: Reached. Find first impact point + sample conditions.
    # First impact = overlap cell closest to the release
    ys, xs = np.where(overlap)
    wx, wy = h.pixels_to_world(ys, xs, transform)
    dists = np.hypot(wx - release_centroid.x, wy - release_centroid.y)
    first_idx = int(np.argmin(dists))
    impact_pt = Point(wx[first_idx], wy[first_idx])
    
    # Sample flow depth and velocity at the impact point
    depth_val = h.sample_raster_at_xy(pft_path, [(impact_pt.x, impact_pt.y)])[0]
    vel_val = h.sample_raster_at_xy(pfv_path, [(impact_pt.x, impact_pt.y)])[0]
    
    # Chainage along road
    chainage = road_for_chainage.project(impact_pt)
    
    return {
        'reached': True,
        'intersection_xy': [float(impact_pt.x), float(impact_pt.y)],
        'flow_depth_at_road_m': float(depth_val),
        'velocity_at_road_ms': float(vel_val),
        'distance_to_road_m': 0.0,
        'chainage_m': float(chainage)
    }


def build_scenario_summary(scenario, peak_rasters, reach_mask_path,
                           road_result, entrainment_info, runtime_s):
    # Aggregate per-scenario metrics into a single dict (-> JSON later)
    # @param scenario: scenario dict
    # @param peak_rasters: dict from find_peak_rasters
    # @param reach_mask_path: Path to reach mask raster
    # @param road_result: dict from analyze_road_intersection
    # @param entrainment_info: dict from auto_detect_entrainment (or None)
    # @param runtime_s: float - total scenario runtime in seconds
    # @returns: dict ready to JSON-dump
    
    # Peak max stats
    pft_data, pft_meta = h.read_raster(peak_rasters['pft'])
    pfv_data, _ = h.read_raster(peak_rasters['pfv'])
    ppr_data, _ = h.read_raster(peak_rasters['ppr'])
    
    cell_area = abs(pft_meta['transform'].a * pft_meta['transform'].e)
    
    # Reach mask + max runout
    reach, reach_meta = h.read_raster(reach_mask_path)
    reach_area = float(reach.sum()) * cell_area
    
    # Max runout = farthest reach cell from release centroid
    release_gdf = gpd.read_file(scenario['release_geojson'])
    centroid = release_gdf.geometry.iloc[0].centroid
    
    if reach.any():
        ys, xs = np.where(reach.astype(bool))
        wx, wy = h.pixels_to_world(ys, xs, reach_meta['transform'])
        dists = np.hypot(wx - centroid.x, wy - centroid.y)
        max_runout = float(dists.max())
    else:
        max_runout = 0.0
    
    return {
        'scenario_id': scenario['scenario_id'],
        'params': scenario['params'],
        'entrainment': entrainment_info or {},
        'runout_max_distance_m': max_runout,
        'reach_area_m2': reach_area,
        'peak_velocity_max_ms': float(np.nanmax(pfv_data)),
        'peak_flow_depth_max_m': float(np.nanmax(pft_data)),
        # AvaFrame ppr is already in kPa. Do not divide by 1000.
        'peak_pressure_max_kpa': float(np.nanmax(ppr_data)),
        'road': road_result,
        'runtime_s': runtime_s
    }


def run_one_scenario(scenario, shared_paths, config, skip_sim=False):
    # Run a single scenario through the full pipeline
    # @param scenario: scenario dict (from load_scenarios in main)
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param config: top-level config dict
    # @param skip_sim: If True, skip com1DFA call (use existing peak rasters)
    # @returns: dict - the scenario summary (also written to summary.json)
    # #   - Used both directly (sequential mode) and as multiprocessing worker
    
    sid = scenario['scenario_id']
    logging.info(f"[{sid}] starting")
    start_t = time.time()
    
    # Same work_dir rule as materialize_ava_dir: write to the scenario
    # folder unless a separate work_dir was assigned upstream.
    work_dir = Path(scenario.get('work_dir') or scenario['scenario_dir'])
    scenario_outputs = work_dir / 'outputs'
    scenario_outputs.mkdir(parents=True, exist_ok=True)
    
    # 1. Materialize AvaFrame directory + auto-detect entrainment + save flowpath
    ava_dir, ent_info = materialize_ava_dir(
        scenario, shared_paths, config, preserve_outputs=skip_sim
    )
    
    # 2. Write com1DFA config
    local_cfg = work_dir / 'scenario.ini'
    write_com1dfa_config(scenario, config, local_cfg)
    
    # 3. Run com1DFA (or skip if --post-only)
    if not skip_sim:
        logging.info(f"[{sid}] running com1DFA")
        n_sims = run_com1dfa(ava_dir, local_cfg)
        logging.info(f"[{sid}] com1DFA finished ({n_sims} sim)")
    else:
        logging.info(f"[{sid}] skipping simulation (--post-only)")
    
    # 4. Find peak rasters and copy to scenario outputs as COGs
    peak_rasters = find_peak_rasters(ava_dir)
    for res_type, src_path in peak_rasters.items():
        dst = scenario_outputs / f"peak_{restype_to_human_name(res_type)}.tif"
        h.convert_raster_format(src_path, dst)
    
    # 5. Reach mask (flow depth >= threshold; optionally also velocity, pressure)
    reach_path = scenario_outputs / 'reach_mask.tif'
    build_reach_mask(
        pft_path=peak_rasters['pft'],
        out_path=reach_path,
        depth_threshold_m=config['reach']['flow_depth_threshold_m'],
        pfv_path=peak_rasters.get('pfv'),
        velocity_threshold_ms=config['reach']['min_velocity_ms'],
        ppr_path=peak_rasters.get('ppr'),
        pressure_threshold_kpa=config['reach'].get('pressure_threshold_kpa', 0.0)
    )
    
    # 6. Reach polygon (vector form of reach mask, for the viewer)
    reach_data, reach_meta = h.read_raster(reach_path)
    reach_polys = h.mask_to_polygons(reach_data, reach_meta['transform'])
    if reach_polys:
        reach_gdf = gpd.GeoDataFrame(
            {'scenario': [sid] * len(reach_polys)},
            geometry=reach_polys,
            crs=reach_meta['crs']
        )
        reach_gdf.to_file(
            scenario_outputs / 'reach_polygon.geojson', driver='GeoJSON'
        )
    
    # 7. Road intersection analysis
    road_result = analyze_road_intersection(
        reach_mask_path=reach_path,
        pft_path=peak_rasters['pft'],
        pfv_path=peak_rasters['pfv'],
        road_path=shared_paths['road'],
        release_polygon_path=scenario['release_geojson'],
        road_buffer_m=config['road']['buffer_m'],
        chainage_origin=config['road']['chainage_origin']
    )
    
    # 8. Reference points (tower, bench, road, ...) when the path has them
    ref_results = []
    rp_path = shared_paths.get('reference_points')
    if rp_path and Path(rp_path).exists():
        release_centroid = gpd.read_file(
            scenario['release_geojson']
        ).geometry.iloc[0].centroid
        ref_results = h.analyze_reference_points(
            reach_mask_path=reach_path,
            pft_path=peak_rasters['pft'],
            pfv_path=peak_rasters['pfv'],
            refpoints_path=rp_path,
            release_centroid=release_centroid,
            radius_m=float(config.get('reference_points', {}).get('radius_m', 10.0)),
        )
        # A point named 'road' takes the road-corridor result, so the
        # road number is the same everywhere it appears
        for r in ref_results:
            if r['ref_name'].lower() == 'road':
                r['reached'] = bool(road_result['reached'])
                r['distance_short_m'] = float(road_result['distance_to_road_m'])
                r['depth_m'] = road_result['flow_depth_at_road_m']
                r['velocity_ms'] = road_result['velocity_at_road_ms']

    # 9. Build summary and write to summary.json
    summary = build_scenario_summary(
        scenario=scenario,
        peak_rasters=peak_rasters,
        reach_mask_path=reach_path,
        road_result=road_result,
        entrainment_info=ent_info,
        runtime_s=time.time() - start_t
    )
    summary['reference_points'] = ref_results
    with open(scenario_outputs / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(
        f"[{sid}] done in {summary['runtime_s']:.1f}s | "
        f"reached_road={road_result['reached']} | "
        f"runout={summary['runout_max_distance_m']:.1f} m"
    )
    return summary