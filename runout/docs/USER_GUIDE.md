# User guide

This guide covers the runout side of avachain: what it computes, how to
set it up, how a forecast cycle runs, how to read the products, and what
to do when a run fails. COMMANDS.md lists the commands. CONFIG.md lists
every config key.

## 1. What the pipeline does

The snowpack chain produces, for each forecast date and horizon, an
ensemble of release scenarios. Each scenario is one way the start zone
could release: a release polygon, a release depth raster, a density, and
a weight (scenario_probability) from the snowpack model.

For each scenario the runout side:

1. Builds an AvaFrame input folder (DEM, release area and thickness,
   forest resistance area, entrainment area and thickness).
2. Runs com1DFA.
3. Thresholds peak flow depth into a reach mask (the cells the flow
   reached).
4. Tests the reach mask against the road corridor and against each
   reference point (tower, bench, road, and so on).
5. Writes a summary.

Across the ensemble it then:

6. Stacks the reach masks, weighted by scenario probability, into a
   probability raster. At each cell this is the weighted share of
   scenarios whose flow got there.
7. Draws the envelope (P at or above 0.05) and computes reach to the road
   and to each reference point.
8. Builds the cycle products: report, details, workbook, viewer.

The probability is conditional on release. It says how far the slide
would run given the snowpack scenarios, not how likely the slope is to
slide. Release likelihood is carried by the scenario weights.

Depths are slope-normal everywhere in the runout side. Any conversion
from vertical to slope-normal happens before the snowpack chain. The
runout side does no conversion.

## 2. Setting up a machine

1. Python 3.10 or newer. pip install -r requirements.txt. Install
   AvaFrame 2.1 or newer per its own documentation (it is not in
   requirements.txt).
2. Copy config/local.example.yaml to config/local.yaml and set:
   - avachain_root: the avachain tree, normally /home/avachain
   - scenario_root: where the snowpack chain writes scenarios, normally
     /home/avachain/intermediate/scenarios
   - publish_root: the web-served folder, normally
     /home/www/html/avachain, or leave it out to skip publishing
   local.yaml is gitignored. It is the only file that differs between
   machines.
3. Place the static layers for each path under
   <avachain_root>/data/static/<slug>/ and check that the file names in
   config/paths/<slug>.yaml match. Required: DEM, forest shapefile, road
   line. Optional: reference points (a point layer with a ref_name
   field), entrainment polygon, start zone polygon.
4. Run a dry run (section 4). It copies scenarios, validates CRS across
   every input, writes the run configs and the manifest, and stops before
   simulating. Most setup mistakes show up here.

## 3. The three config layers

Path profile (config/paths/<slug>.yaml). Static inputs for one avalanche
path: terrain, canopy, road, reference points, release and entrainment
polygons, display names, road buffer, reference point radius. One file
per path. Nothing in it changes between cycles.

Physics (config/physics/<name>.yaml). Model parameters: friction model
and values, simulation type, resistance, entrainment parameters, reach
thresholds, probability settings. operational.yaml is the locked set
forecasts run with. A content hash of the resolved physics goes into
every manifest and report footer, so a run with a changed tau0 is never
mistaken for an operational one. Test values go in another file under
config/physics/, never in operational.yaml.

Profiles (config/profiles.yaml). Named jobs. A profile names a mode, a
physics file, the paths and horizons to run, the entrainment mode and
the products to build, and may override individual physics values.
extends inherits another profile. --set section.key=value on the command
line overrides one value for one run and requires --run-name.

Resolution order, lowest to highest: physics file, profile sections,
--set. The result is written as run_config.yaml in each horizon's output
folder, so the exact configuration that ran is always on disk.

## 4. Running a forecast cycle

    python -m scripts run --profile forecast --date 20261201 --snow-height latest

What happens:

1. The profile resolves and the physics and path layers load. Errors
   here are config problems and the run stops before touching data.
2. For each path in the profile, the horizons present under
   <scenario_root>/<slug>/<YYYYMMDD>/ are discovered: T_0, T_1, and so
   on, however many the chain produced. --horizon narrows them and fails
   if the label does not exist for that cycle.
3. manifest.json is written to the cycle output folder with status
   running.
4. For each horizon: the scenario inputs are copied into
   intermediate/runout/<slug>/<YYYYMMDD>/<T_n>/scenario_###/release/,
   run_config.yaml is written, and avalanche_runout.py runs it in a
   subprocess. A horizon that fails is logged and the next one runs. The
   manifest is updated after every horizon.
5. The cycle products are built from the completed horizons and, when
   publish_root is set, copied to the web folder.
6. The manifest ends with status ok, partial (some horizons failed), or
   failed.

Entrainment mode comes from the profile:

- raster: corridor along the flowline; per-cell thickness is the smaller
  of the snow height raster and the scour cap. Needs --snow-height FILE,
  or --snow-height latest for the newest file in
  data/surveys/snow_height/<slug>/ (newest by the date at the start of
  the file name, such as 261130_...). This is the forecast profile.
- auto: corridor along the flowline at the scour cap. No survey needed.
  This is forecast_nosurvey.
- polygon: the path profile's hand-drawn polygon at the scour cap. This
  is forecast_bethel.

The scour cap is each scenario's scour_depth_m from params.json, or
defaults.scour_depth_fallback_m when that is zero.

## 5. Reading the products

All four live in outputs/forecasts/<slug>/<YYYYMMDD>/ and, when
publishing is on, under <publish_root>/forecasts/<slug>/<YYYYMMDD>/ with
a latest/ copy and an index.html listing cycles.

report.html is the forecast.

- Header: path, issue time, forecast date (the T+0 date), last snow
  height survey, and one row per horizon with its valid date.
- Reach to road table, one row per horizon. The weighted P is the share
  of scenario weight whose flow reaches the road corridor. The
  unweighted P and its Wilson 95% interval show how many scenarios
  reached, which matters when a few heavy scenarios carry the weighted
  number. Nearest miss names the scenario that stopped closest, how far
  short, its weight, and its rank by weight in the ensemble. A nearest
  miss ranked 2 of 75 means a likely scenario nearly reached; ranked 70
  of 75 means an unlikely one did.
- Probability maps, one per horizon: hillshade, probability in viridis,
  road in orange, impact points as red dots, reference points as
  triangles.
- Runout exceedance: for each horizon, the weighted probability that
  runout is at least a given distance from the release. Vertical lines
  mark the road and the reference points. Where a horizon's line crosses
  the road line is P(reach road). Lines moving right from T+0 to T+2
  mean the forecast is trending longer.
- Reference points table: weighted P(reach) at each point, per horizon.
  A point named road uses the road corridor test so it agrees with the
  table above.
- Where the road is hit: chainage along the road of the impact points,
  weighted. Shown only when something reached.
- Footer: physics file and hash, friction model and values, simulation
  type, reach threshold, entrainment mode, depth convention, pipeline
  version, git commit, AvaFrame version.

details.html: per horizon, the configuration that ran, a runout
histogram, sensitivity scatter against every ensemble input that varies,
and the full scenario table with a reached column per reference point.

scenarios.xlsx: summary (the reach-to-road table), reference_points, one
sheet per horizon with every scenario, and manifest.

viewer.html: one Leaflet viewer with a button per horizon. Layers:
hillshade, imagery, probability heatmap, envelope, road, impact points,
forest, and a per-scenario accordion (release and runout polygon for
every scenario, filterable). The accordion can be dropped from a profile
with output.viewer_scenario_layers: false if the file gets heavy.

manifest.json is the record of what a run was made from: resolved
physics and path profile with hashes, the scenario folder and its
modification time, the survey file and date, scenario counts and status
per horizon, git commit, AvaFrame version. The verify mode reads it to
reproduce a past forecast.

## 6. Changing physics

Do not edit operational.yaml to try something. Either:

- --set friction.tau0=100 --run-name tau100, or
- copy operational.yaml to config/physics/<name>.yaml, edit, and run
  with --set run.physics=<name> --run-name <name> or a profile that
  names it.

Both leave the operational file alone, and both are recorded in the
manifest. When a calibration run supports a change to the operational
value, edit operational.yaml and put the catalog row id in the comment
next to the value.

friction.sources says where each parameter comes from: fixed (the value
in the file) or params (each scenario's params.json). Only Voellmy-family
models use mu and xi; samosAT models use tau0 and their internal
calibration. A model ignores parameters it has no key for.

## 7. When something fails

"local.yaml not found": copy config/local.example.yaml and set the
roots.

"no path profile for 'x'": the slug has no config/paths/x.yaml, or the
slug: inside the file does not match the file name.

"terrain.dem not found" and similar: the path profile points at a file
that is not under <avachain_root>. Paths in the yaml are relative to
avachain_root.

"source date folder not found": the snowpack chain has not written
<scenario_root>/<slug>/<YYYYMMDD>/ yet, or the date is wrong. The folder
name is YYYYMMDD.

"horizon(s) not found for this cycle": --horizon named a label this cycle
does not have. The message lists what exists.

"entrainment.mode is 'raster' but no snow height raster was given": the
forecast profile needs --snow-height. Use forecast_nosurvey when there
is no survey.

"CRS validation failed": an input is not in the DEM's horizontal CRS
(EPSG:6342). The message names the file. Reproject it, or for a scenario
file with missing metadata use fix_input_crs.py.

"--run-name is required when --set overrides are used": a tweaked run
has to be named so it is not mistaken for the profile.

"--set friction.tau00: not a key in the physics file": typo guard.
--set only accepts keys the physics file already has, plus
entrainment.mode, execution.* and output.*.

A horizon shows FAILED (exit 1): the run log holds that horizon's
traceback. Common causes: a scenario folder missing a required file
(release.geojson, depth.asc, density.json, params.json with mu, xi,
scour_depth_m and scenario_probability), the forest shapefile not
overlapping the DEM, an AvaFrame error. Fix and rerun that horizon alone
with --horizon T_n --no-copy, or --no-copy --post-only if the
simulations finished and only post-processing failed.

"nothing to publish" or a permission error on the web folder: the
products are still in outputs/. Check write access to publish_root, then
python -m scripts products --path <slug> --date <date>.

The report says an older version after a rebuild: the footer reads the
version from the manifest the run wrote. A rebuilt report on an old
cycle keeps that cycle's version.

## 8. Disk use

intermediate/runout/.../avaframe/ holds com1DFA working files for every
scenario and is the bulk of the disk use. It is safe to delete at any
time; everything downstream rebuilds from outputs/ and from
scenario_###/outputs/. Scenarios under intermediate/scenarios/ are slow
to rebuild and should stay.

## 9. Depth sweeps

A sweep answers a different question from a forecast: not whether
today's snowpack reaches the road, but how much snow in the start zone
it takes. It exists for paths where surveys are rare, so start zone
depth can be tracked and read against a lookup.

    python -m scripts run --profile sweep --path bethel --run-name preseason

The sweep takes the path's start zone polygon (release.start_zone in the
path profile) and generates one scenario per depth in sweep.depths, with
the whole polygon at that uniform depth. It runs that set once per
friction class, and for each class once with entrainment (the path's
entrainment polygon at sweep.entrainment_depth_m) and once without.
Weights are equal and the probability stack is not used.

Friction classes live in config/physics/friction_classes.yaml, one row
per slab hand hardness (F, 4F, 1F, P, K) with a tau0 and a density. The
tau0 values are placeholders inside the dry-snow cohesion range of
Bartelt et al. (2015) and around the one existing fit; calibration runs
replace them one row at a time. A wet row is present and off by default,
since wet snow is a different regime.

Products, in outputs/calibration/<slug>/sweep_<name>/ and published
under <publish_root>/calibration/<slug>/:

- sweep.html: the class table, the lookup (smallest depth that reaches
  each reference point and the road, per class, with and without
  entrainment), and the figure.
- sweep.png: runout against depth, one line per class, solid with
  entrainment and dashed without, reference points as horizontal lines.
- lookup.csv and sweep_table.csv: the same numbers as tables, the second
  with one row per simulation.

Read the lookup with the class that matches the expected slab. A road
value of 1.25 m in the medium row means a 1F slab needs about 1.25 m of
uniform start zone depth to reach the road under the placeholder tau0.
The gap between the with and without entrainment rows is the track snow
effect. Partial start zones are not in the sweep; they would be one more
polygon and one more profile.

## 10. Recording an observed event

Every calibration and verification starts from an event record. Make one
folder per event:

    data/static/<slug>/observed/<YYYYMMDD>/
        event.yaml
        debris.geojson      optional
        release.geojson     optional
        toe.geojson         optional, one point

event.yaml:

    date: 2026-01-18
    path: little_professor
    debris_polygon: debris.geojson
    release_polygon: release.geojson
    toe_point: toe.geojson          # or [easting, northing] in the DEM CRS
    points:                         # per reference point, reached or not_reached
      tower: reached
      bench: reached
      road: not_reached
    slab:
      hand_hardness: -3             # SNOWPACK index, -1 F to -5 K
      density_kgm3: 315
      depth_m: 0.9                  # slope-normal
    source: resurvey flight
    notes: free text

Any of the three observations may be missing. A photo and a note become
a points: block, which is enough to score. The slab block ties the fit
to the friction class table, so hardness should be filled in whenever it
is known. Geometries can be in any CRS; they are reprojected to the
DEM's.

## 11. Calibrating

    python -m scripts run --profile calibrate --path little_professor --event 20260118 --run-name tau_grid

Calibrate runs one scenario across the parameter grid in the profile
(grid: {friction.tau0: [50, 75, ...]}) and scores every cell against the
event. The grid can name any physics key, so the same mode fits tau0
under samosAT, mu and xi under Voellmy, or a forest parameter.

The scenario comes from calibrate.scenario: synthetic builds one from
the event's release polygon at the slab depth and density in the record;
an id such as scenario_038 copies that member from the ensemble for the
event date. --scenario and --scenario-date override the profile for one
run.

Every cell is scored at each threshold in calibrate.thresholds from the
same simulation, and judged at judge_threshold. The scores:

- IoU, recovery and over-prediction against the debris polygon.
- Toe error, modeled farthest reach minus observed toe, both as distance
  from the release centroid. Positive runs long.
- Points agree, the share of reference points whose modeled reach
  matches the observation, with the mismatches named.

The recommended fit follows what was observed: highest IoU when there is
a polygon, smallest toe error when there is only a toe, and the set of
cells that agree with every point when there are only points. The range
is every cell within calibrate.tolerance of the best. With one event the
score surface is usually flat near the minimum, so the range is the
number to carry, not the best.

Each run writes calibrate.html, scores.csv and a figure to
outputs/calibration/<slug>/<event>/<run_name>/, and one row to
outputs/calibration/catalog.csv (replacing an earlier row with the same
path, event and run name). catalog.html plots fitted tau0 against slab
hardness with the range as error bars, grouped by physics hash, since
rows are comparable only within one physics family. After a physics
change, --event all refits every recorded event in one go.

Nothing here edits operational.yaml or friction_classes.yaml. The
catalog is read, the file is changed by hand, and the catalog row is
noted in the comment.

## 12. Verifying a forecast

    python -m scripts run --profile verify --path little_professor --date 20260118 --run-name v1

Verify answers a different question from calibrate: not which parameters
reproduce the event, but whether the forecast as issued would have
warned. It reads the forecast's manifest.json, reruns that day's T+0
ensemble with the physics recorded there (and with each physics file in
the profile's bracket), and scores each member against the event record
for that date (--event when the event date differs):

- P(reach road) against the observed outcome at the road, with the Brier
  term (p minus o) squared. Near 0 is good; 1 is a confident miss.
- The same per reference point.
- With a debris polygon, the share of the debris inside the P at or
  above 0.05 envelope and the mean P over the debris.

A day with no event record cannot be verified. The runout side says
nothing about a day on which nothing released; that is the release
model's score, not this one's.

Each run writes verify.html and scores.csv to
outputs/verify/<slug>/<date>/<run_name>/ and one row per member to
outputs/verify/verification.csv, the season's track record.

## 13. Logs and the smoke test

Every python -m scripts run writes its full log, including the
per-horizon subprocess output, to
<avachain_root>/logs/runout/<stamp>_<profile>_<run>.log. --no-log turns
that off.

python tests/smoke_test.py builds a synthetic path and scenario drop
under /tmp and runs forecast, sweep, calibrate and verify with com1DFA
replaced by a stand-in. It is the install check and the regression test.

## 14. Not in the runout side

Triggering a run when the chain finishes, notification, and cleanup of
intermediate/runout/ belong to the chain. HANDOFF.md states the command,
the exit codes, the manifest fields to poll, and the deletion rule.
Per-scenario tau0 from the friction class table by slab hardness is a
future change, once the catalog has enough rows to support it.
