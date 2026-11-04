# Common commands

All commands run from the repo root. Dates are YYYYMMDD; YYYY-MM-DD
is accepted too.

## Daily forecast

    # with a snow height survey (entrainment from the raster)
    python -m scripts run --profile forecast --date 20261201 --snow-height latest
    python -m scripts run --profile forecast --date 20261201 --snow-height /home/avachain/data/surveys/snow_height/little_professor/261130_Professor_snowHeight.tif

    # no survey available
    python -m scripts run --profile forecast_nosurvey --date 20261201

    # today's date is the default
    python -m scripts run --profile forecast_nosurvey

    # Bethel (polygon entrainment)
    python -m scripts run --profile forecast_bethel --date 20261201

## Check before running

    # copy the chain's scenarios, validate CRS and inputs, write run configs, no simulations
    python -m scripts run --profile forecast --date 20261201 --snow-height <HS.tif> --dry-run

## One horizon, one path

    python -m scripts run --profile forecast_nosurvey --date 20261201 --horizon T_0
    python -m scripts run --profile forecast_nosurvey --date 20261201 --horizon T_0 --horizon T_1
    python -m scripts run --profile forecast_nosurvey --date 20261201 --path bethel

## Reuse what is already copied

    # scenarios are already under intermediate/runout; skip the copy
    python -m scripts run --profile forecast_nosurvey --date 20261201 --no-copy

    # simulations already ran; redo post-processing (reach masks, stack, products)
    python -m scripts run --profile forecast_nosurvey --date 20261201 --no-copy --post-only

## Rebuild the report and viewer

    # from the outputs on disk, no simulations, then publish
    python -m scripts products --path little_professor --date 20261201

    # report, details and workbook only
    python -m scripts products --path little_professor --date 20261201 --products report

    # do not copy to the web folder
    python -m scripts products --path little_professor --date 20261201 --no-publish

## Try a parameter without touching the operational physics

    # any --set needs a --run-name; the manifest records the override
    python -m scripts run --profile forecast_nosurvey --date 20261201 --set friction.tau0=100 --run-name tau100
    python -m scripts run --profile forecast_nosurvey --date 20261201 --set reach.flow_depth_threshold_m=0.5 --run-name reach05
    python -m scripts run --profile forecast_nosurvey --date 20261201 --set entrainment.mode=polygon --run-name poly

    # a whole physics file
    python -m scripts run --profile forecast_nosurvey --date 20261201 --set run.physics=voellmy_test --run-name voellmy

    # a profile file outside the repo
    python -m scripts run --profile mytest --profile-file ~/mytest.yaml --date 20261201

## Depth sweep (Bethel lookup)

    # full start zone, every friction class, with and without entrainment
    python -m scripts run --profile sweep --path bethel --run-name preseason

    # check inputs and generate the scenarios, no simulations
    python -m scripts run --profile sweep --path bethel --run-name preseason --dry-run

    # two classes only, or a different entrainment depth
    python -m scripts run --profile sweep --path bethel --run-name soft_hard --set run.sweep='{"classes":["soft","hard"]}'
    python -m scripts run --profile sweep --path bethel --run-name ent30 --set run.sweep='{"entrainment_depth_m":0.30}'

    # simulations done; rebuild the table, lookup, figure and page, then publish
    python -m scripts products --path bethel --sweep preseason

## Calibrate against an observed event

    # write the event record first: data/static/<slug>/observed/<YYYYMMDD>/event.yaml (see USER_GUIDE section 10)

    # tau0 grid from the profile, synthetic scenario from the observed release
    python -m scripts run --profile calibrate --path little_professor --event 20260118 --run-name tau_grid

    # use an ensemble member instead of a synthetic scenario
    python -m scripts run --profile calibrate --path little_professor --event 20260118 --run-name tau_grid --scenario scenario_038

    # a different grid, or a different physics file
    python -m scripts run --profile calibrate --path bethel --event 20260227 --run-name wide --set run.grid='{"friction.tau0":[25,50,75,100,150,200,300,500]}'
    python -m scripts run --profile calibrate_voellmy --path little_professor --event 20260118 --run-name voellmy

    # refit every recorded event after a physics change
    python -m scripts run --profile calibrate --path little_professor --event all --run-name refit_v2

    # simulations done; rescore and rewrite the catalog row
    python -m scripts run --profile calibrate --path little_professor --event 20260118 --run-name tau_grid --post-only

    # the catalog
    outputs/calibration/catalog.csv, catalog.html

## Verify a past forecast

    # rerun that day's T+0 with the physics the forecast used, score against the event of the same date
    python -m scripts run --profile verify --path little_professor --date 20260118 --run-name v1

    # event on a different date than the cycle, or extra physics files beside the forecast's
    python -m scripts run --profile verify --path little_professor --date 20260117 --event 20260118 --run-name v1
    python -m scripts run --profile verify --path little_professor --date 20260118 --run-name bracket --set run.bracket=tau0_low,tau0_high

    # the log
    outputs/verify/verification.csv

## Synthetic scenarios (calibration inputs)

    # a specific polygon (observed release, for calibration)
    python -m scripts generate-scenarios --path little_professor --release /home/avachain/data/static/little_professor/observed/20260118/release.geojson --out <dir> --depths 0.9 --densities 315

## Check the install

    python tests/smoke_test.py                       # every mode, no AvaFrame, a few minutes
    python tests/smoke_test.py --keep /tmp/smoke     # keep the synthetic tree to look at

## Debugging

    # every run writes <avachain_root>/logs/runout/<stamp>_<profile>_<run>.log (--no-log to skip)

    python -m scripts run --profile forecast_nosurvey --date 20261201 --horizon T_0 --no-parallel -v

    # the standalone tools run from inside scripts/
    cd scripts
    python rethreshold_reach.py --config /home/avachain/outputs/forecasts/little_professor/20261201/T_0/run_config.yaml --depth-threshold 0.5
    python generate_3d_viewer.py --config /home/avachain/outputs/forecasts/little_professor/20261201/T_0/run_config.yaml
    python fix_input_crs.py --scenarios-dir <dir> --epsg 6342 --dry-run
    python release_volume.py --depth depth.tif --release release.shp

## Where things are

    outputs/forecasts/<slug>/<YYYYMMDD>/manifest.json           what the run was made from
    outputs/forecasts/<slug>/<YYYYMMDD>/report.html             the forecast
    outputs/forecasts/<slug>/<YYYYMMDD>/<T_n>/run_config.yaml   the exact config a horizon ran with
    outputs/forecasts/<slug>/<YYYYMMDD>/<T_n>/scenarios_summary.csv
    intermediate/runout/<slug>/<YYYYMMDD>/<T_n>/scenario_###/outputs/summary.json
    intermediate/runout/<slug>/<YYYYMMDD>/<T_n>/scenario_###/avaframe/   com1DFA working dir (safe to delete)
    outputs/calibration/<slug>/sweep_<name>/sweep.html                   a depth sweep
    outputs/calibration/<slug>/sweep_<name>/lookup.csv                   first depth that reaches each point
    config/physics/friction_classes.yaml                                 tau0 and density per slab hardness class
