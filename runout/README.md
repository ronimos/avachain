# avachain (runout side)

Probabilistic avalanche runout forecasting for CAIC highway paths. The
runout side reads the snowpack scenario ensembles the snowpack chain produces,
runs each scenario through AvaFrame com1DFA, stacks the results into a
probability of reach, and reports whether the slide reaches the road and
the path's reference points.

Documentation:

- docs/USER_GUIDE.md  what the pipeline does, setup, running a forecast,
  reading the products, troubleshooting
- docs/COMMANDS.md    the commands, grouped by task
- docs/CONFIG.md      every key in the config files
- docs/SCENARIO_CONTRACT.md  what the runout side reads from each scenario folder
- docs/HANDOFF.md     how the chain calls the runout side, what to poll, what to delete

## Layout

    config/
      local.example.yaml       copy to local.yaml, set the roots
      paths/<slug>.yaml        static inputs for one path
      physics/<name>.yaml      model parameters, hashed into every run
      profiles.yaml            named jobs (forecast, sweep, calibrate, verify)
    docs/                      documentation
    scripts/                   the package; python -m scripts
    tests/smoke_test.py        end-to-end check without AvaFrame
    requirements.txt

Data tree (roots from local.yaml):

    <avachain_root>/data/static/<slug>/{terrain,canopy,infrastructure,release}/
    <avachain_root>/data/surveys/{ortho,raw,snow_height,snow_on_DSM,snow_off_DSM}/<slug>/<files>
    <scenario_root>/<slug>/<YYYYMMDD>/<T_n>/scenario_###/inputs/           snowpack chain, read-only
    <avachain_root>/intermediate/runout/<slug>/<YYYYMMDD>/<T_n>/scenario_###/{release,avaframe,outputs}/
    <avachain_root>/outputs/forecasts/<slug>/<YYYYMMDD>/{manifest.json, report.html, details.html, viewer.html, scenarios.xlsx, <T_n>/}
    <avachain_root>/outputs/calibration/<slug>/{sweep_<name>/, <event>/<run>/}, catalog.csv
    <avachain_root>/outputs/verify/<slug>/<date>/<run>/, verification.csv
    <avachain_root>/data/static/<slug>/observed/<YYYYMMDD>/event.yaml     observed events
    <publish_root>/forecasts/<slug>/{<YYYYMMDD>/, latest/, index.html}     web copy

## Quick start

    pip install -r requirements.txt        # AvaFrame separately, per its docs
    cp config/local.example.yaml config/local.yaml
    python -m scripts run --profile forecast --date 20261201 --snow-height <HS.tif> --dry-run
    python -m scripts run --profile forecast --date 20261201 --snow-height <HS.tif>

## Status

Version 3.0.0. Forecast, sweep, calibrate and verify modes are complete.
Triggering, notification and cleanup are the chain's job; see
docs/HANDOFF.md.
