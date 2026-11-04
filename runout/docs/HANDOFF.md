# Running the runout side from the chain

This page describes how to call the runout side, how to check whether a
run finished, where its files land, and what can be deleted. USER_GUIDE.md
covers the rest.

## Install check

    pip install -r requirements.txt
    pip install avaframe                 # 2.1 or newer
    cp config/local.example.yaml config/local.yaml   # set the three roots
    python tests/smoke_test.py           # a few minutes, no AvaFrame needed

The smoke test builds a synthetic path and scenario drop under /tmp and
runs every mode with com1DFA replaced by a stand-in. It passes when the
install and the config layout are right. Run it after any change.

## The command

    python -m scripts run --profile forecast --date YYYYMMDD --snow-height latest

- --date is the folder name under <scenario_root>/<slug>/. Default is
  today.
- --snow-height latest takes the newest file in
  data/surveys/snow_height/<slug>/ (by the date at the start of the file
  name). A file path works too. Use --profile forecast_nosurvey when there
  is none; the forecast profile does not run without a raster.
- The profile lists the paths. --path <slug> runs one.
- Every horizon folder present is run. A horizon that fails is logged
  and the others continue.

Exit code 0: every horizon completed and the products were built and
published. Exit code 1: something did not; the log and the manifest say
what. Exit code 2: a config error before anything ran.

## What to poll

<avachain_root>/outputs/forecasts/<slug>/<YYYYMMDD>/manifest.json:

- status: running, ok, partial, failed, dry-run
- finished: timestamp, null while running
- horizons.<T_n>.status: ok, FAILED (exit 1), FAILED (config),
  skipped (...)
- physics.hash, survey.date, scenario_source.modified: what the run was
  made from

The manifest is written when the run starts and rewritten after every
horizon, so a watcher can read it mid-run.

## Where things land

    <avachain_root>/logs/runout/<stamp>_<profile>_<run>.log      full log of every run
    <avachain_root>/outputs/forecasts/<slug>/<YYYYMMDD>/          products and manifest
    <publish_root>/forecasts/<slug>/<YYYYMMDD>/, latest/           web copy
    <avachain_root>/intermediate/runout/<slug>/<YYYYMMDD>/         work tree

## Cleanup

Safe to delete at any time:

- intermediate/runout/**/avaframe/ (com1DFA working files, the bulk of
  the disk use)
- intermediate/runout/<slug>/<YYYYMMDD>/ as a whole, once its products
  exist; the forecast products never read it again. Verify mode reruns
  from scenario_root, not from here.

Keep:

- intermediate/scenarios/ (verify and calibrate reread the scenario
  drops)
- outputs/ (forecasts, calibration, verify, the catalog and the log)
- data/static/ including observed/

Suggested rule: delete intermediate/runout/<slug>/<YYYYMMDD>/ folders
older than two weeks and leave sweep, calibrate and verify work folders
until someone removes them by hand.

## Rerunning

    python -m scripts run --profile forecast --date YYYYMMDD --no-copy --post-only
        simulations finished, redo post-processing and products

    python -m scripts products --path <slug> --date YYYYMMDD
        rebuild and republish report, details, workbook and viewer from disk

Two runs on the same path and date should not overlap. There is no lock;
whatever calls the command should confirm the previous one finished
(poll finished in the manifest).

## Items on the chain side

These are not part of the runout side:

1. Trigger. Deciding when a date folder is complete. A marker file
   written as the chain's last step, or a poll for a date folder that has
   stopped changing. The runout side has no watcher; it runs when called.
2. Notification. The runout side writes nothing but logs and the
   manifest. Mail, a webhook, or nothing.
3. Cleanup. The rule above, scheduled.
4. Path names. scenario_root/<folder>/ is matched by scenario_slug in
   config/paths/<slug>.yaml (default: the slug). If the chain writes a
   different folder name, set scenario_slug to it and nothing else
   changes.

## Where to look in the code

- scripts/cli.py, run_forecast: the cycle loop
- scripts/discover.py: how horizons and scenarios are found
- docs/SCENARIO_CONTRACT.md: what each scenario folder must contain
- config/local.yaml: the only machine-specific file
