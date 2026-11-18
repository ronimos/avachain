"""runout_step.py — run com1DFA runout on a cycle's published release sets.

    python src/avachain/runout_step.py --toml slopes/Professor_PTC/slope_config.toml \
           [--snapshot-date YYYY-MM-DD] [--dry-run]

The last step of run_<slope>.sh and run_daily_<slope>.sh (Ron, 2026-11-04:
chained inline after `scenarios`). It reads the set `scenarios` just published
at <scenarios_dir>/<cycle date>/ (T_0, T_1, ... from `analysis_pipeline forecast`,
or one set at the date folder itself) and runs the runout part (runout/, its own
venv: AvaFrame 2.1 needs numpy < 2) on it:

  1. Each set must have READY and match its manifest (handoff.verify): every
     file present with its recorded size and sha256. A set that fails is left
     out with an error; no readable set at all is exit 2.
  2. The runout tree for the slope, all under [runout] work_dir:
       config/local.yaml          avachain_root = work_dir, scenario_root,
                                  publish_root (generated each run)
       config/{paths,physics,profiles.yaml} -> the repo's runout/config/
       scen/<path_profile>/<YYYYMMDD>/T_n -> each verified release set, or
                                  a merged view of it when scenarios repeat
                                  the same runout input (merge_view: one run
                                  per group, with the summed probability)
     Runout then writes its work tree, logs and products under work_dir and
     publishes report.html etc. to <publish_root>/forecasts/<path_profile>/.
  3. One global lock ([runout] lock_file) for every runout on the host, taken
     blocking: AvaFrame sims can reach >10 GB each and the 4-sim cap
     (execution.max_workers_cap) must hold across slopes, not only per run.
  4. runout's forecast profile with --snow-height set: each day entrains its
     own set's scour_depth.tif (entrainment rule B). runout publishes the
     report after each day, with the days still running shown as pending. An
     empty set (status no_scenarios) needs no sims; runout reports why there
     is no forecast for that day.

Forced explosive shots ([scenarios] force_explosive_triggers): when the
cycle also has shot sets at <date>/shots[/T_n], a second runout runs on them
under the same lock, right after the model run. It has its own tree,
<work_dir>/shots/, and its own web folder, <publish_root>/shots/forecasts/
<path_profile>/, so neither run's report, index or latest/ touches the
other's. Its path profile is a generated copy whose display names it as the
explosive-shot forecast and says what the probabilities are conditional on.

Exit status is runout's (the first failure of the two runs). The run scripts treat a failure as a warning: the
survey stays DONE, because snowpack and release succeeded and are published
(Ron, 2026-11-04), and runout can be rerun alone with this command.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ProjectConfig  # noqa: E402
import handoff  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNOUT = REPO / "runout"


def runout_settings(toml_path: Path) -> dict | None:
    """The TOML's [runout] table with defaults, or None when it has none."""
    rc = tomllib.loads(Path(toml_path).read_text()).get("runout")
    if not rc:
        return None
    for key in ("path_profile", "work_dir"):
        if not rc.get(key):
            raise ValueError(f"{toml_path}: [runout] {key} is required")
    out = dict(rc)
    out["work_dir"] = Path(rc["work_dir"])
    out["lock_file"] = Path(rc.get("lock_file") or out["work_dir"].parent / ".runout.lock")
    out["profile"] = rc.get("profile", "forecast")
    out["publish_root"] = rc.get("publish_root") or None
    return out


SHOTS = "shots"     # analysis_pipeline.SHOT_SET_NAME
SHOT_DISPLAY_SUFFIX = " — explosive shots"
SHOT_CAVEAT = (
    "Explosive-shot forecast: every scenario starts at one of the slope's "
    "explosive-control shot locations, assumed to initiate a failure. Only "
    "propagation is modelled, and a shot whose crack does not propagate "
    "releases nothing. Each shot carries 1/N of the probability, and a shot "
    "that does not release counts as no reach, so P(reach) here is per shot "
    "fired, not conditional on release. The model forecast is published "
    "separately.")


def shot_settings(rc: dict) -> dict:
    """[runout] settings for the shot run: its own tree and web folder."""
    out = dict(rc)
    out["work_dir"] = rc["work_dir"] / SHOTS
    out["publish_root"] = (str(Path(rc["publish_root"]) / SHOTS)
                           if rc["publish_root"] else None)
    out["shots"] = True
    return out


def release_sets(cfg: ProjectConfig, cycle_date: str, shots: bool = False) -> dict:
    """The cycle's published release sets, {label: dir}, T_0 first.

    A forecast cycle publishes <scenarios_dir>/<date>/T_0, T_1, ...
    (analysis_pipeline forecast); a single-day run publishes the set at
    <scenarios_dir>/<date>/ itself, which runs as T_0. A folder counts once
    it has READY (verification comes after)."""
    base = cfg.scenarios_dir / cycle_date
    if shots:
        base = base / SHOTS
    sets = {d.name: d for d in sorted(base.glob("T_*"), key=lambda d: d.name)
            if d.is_dir() and d.name[2:].isdigit() and handoff.is_ready(d)} \
        if base.is_dir() else {}
    if not sets and handoff.is_ready(base):
        sets = {"T_0": base}
    return dict(sorted(sets.items(), key=lambda kv: int(kv[0][2:])))


def prepare_tree(rc: dict, sets: dict, cycle_date: str) -> Path:
    """Write runout's config dir and one T_n link per set. Returns the config dir."""
    work = rc["work_dir"]
    cfg_dir = work / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by src/avachain/runout_step.py from the slope TOML [runout];",
        "# rewritten every run, do not edit.",
        f"avachain_root: {work}",
        f"scenario_root: {work / 'scen'}",
    ]
    if rc["publish_root"]:
        lines.append(f"publish_root: {rc['publish_root']}")
    (cfg_dir / "local.yaml").write_text("\n".join(lines) + "\n")
    for name in ("paths", "physics", "profiles.yaml"):
        link = cfg_dir / name
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        elif link.exists():
            link.unlink()
        if name == "paths" and rc.get("shots"):
            _write_shot_path_profile(rc["path_profile"], link)
        else:
            link.symlink_to(RUNOUT / "config" / name)
    day = work / "scen" / rc["path_profile"] / cycle_date.replace("-", "")
    day.mkdir(parents=True, exist_ok=True)
    for old in day.glob("T_*"):                 # an earlier run of this date
        if old.is_symlink():
            old.unlink()
        elif old.is_dir():                      # a merged view (merge_view)
            shutil.rmtree(old)
    for label, set_dir in sets.items():
        merge_view(Path(set_dir).resolve(), day / label)
    return cfg_dir


# Scenarios whose runout inputs are identical run once (Ron, 2026-11-18).
# With the distance caps off (calibration 2026-11-17.1) the size factors scale
# nothing in the release BFS, so the five size-factor variants of a trigger
# and depth percentile are the same polygon, depth raster, density and
# parameters: Jan 18 holds 15 distinct model runouts in 75 scenarios and 9 in
# 45 shots. com1DFA is deterministic on identical input (checked on Jan 18:
# every duplicate's rasters byte-identical), so one run carries the group,
# with the summed scenario_probability. Every weighted product (P(reach) stack,
# envelope, contours, weighted road and reference-point fractions) sums
# w_i * mask_i, which is unchanged; absolute shot weights keep
# ensemble_weight_total, so P(no release) is unchanged too. What does change
# is anything that counts runs: n_scenarios, n_reached, the unweighted
# fraction and its Wilson interval, which now count distinct runouts.
#
# These are the only fields that may differ inside a group: labels the runout
# never reads. Any other difference, in any file, keeps the scenarios apart.
MERGE_LABEL_PARAMS = ("scenario_probability", "size_factor")
MERGE_LABEL_RELEASE_PROPS = ("scenario_id", "size_factor", "weight")


def _input_key(scen_dir: Path) -> str:
    """A digest of everything a scenario gives the runout, labels left out."""
    h = hashlib.sha256()
    for f in sorted(p for p in scen_dir.rglob("*") if p.is_file()):
        if f.name == "params.json":
            d = json.loads(f.read_text())
            blob = json.dumps({k: v for k, v in d.items()
                               if k not in MERGE_LABEL_PARAMS}, sort_keys=True)
        elif f.name == "release.geojson":
            d = json.loads(f.read_text())
            for feat in d.get("features") or []:
                feat["properties"] = {k: v for k, v in (feat.get("properties") or {}).items()
                                      if k not in MERGE_LABEL_RELEASE_PROPS}
            blob = json.dumps(d, sort_keys=True)
        else:
            blob = f.read_bytes()
        if isinstance(blob, str):
            blob = blob.encode()
        h.update(f.relative_to(scen_dir).as_posix().encode() + b"\0"
                 + hashlib.sha256(blob).digest())
    return h.hexdigest()


def merge_groups(set_dir: Path) -> list[list[Path]]:
    """The set's scenario dirs grouped by identical runout input, in id order;
    the first of each group is the one that runs."""
    groups: dict[str, list[Path]] = {}
    scen_root = set_dir / "scenarios"
    if not scen_root.is_dir():
        return []
    for d in sorted(scen_root.iterdir()):
        if d.is_dir() and d.name.startswith("scenario_"):
            groups.setdefault(_input_key(d), []).append(d)
    return list(groups.values())


def merge_view(set_dir: Path, view: Path) -> None:
    """Link `view` to the set; where scenarios duplicate, a merged view instead.

    No duplicates: `view` is a symlink to the published set, as before.
    Otherwise `view` is a directory of links to the set's files, with one
    scenario per group: its files linked, except params.json, which gets the
    group's summed scenario_probability and `merged_scenarios` (the ids it
    stands for, itself first). The published set is never written."""
    groups = merge_groups(set_dir)
    if all(len(g) == 1 for g in groups):
        view.symlink_to(set_dir)
        return
    view.mkdir()
    for f in set_dir.iterdir():
        if f.name != "scenarios":
            (view / f.name).symlink_to(f)
    scen = view / "scenarios"
    scen.mkdir()
    for group in groups:
        rep = group[0]
        out = scen / rep.name
        out.mkdir()
        for f in rep.iterdir():
            if f.name != "params.json":
                (out / f.name).symlink_to(f)
        params = json.loads((rep / "params.json").read_text())
        if len(group) > 1:
            params["scenario_probability"] = math.fsum(
                json.loads((d / "params.json").read_text())["scenario_probability"]
                for d in group)
            params["merged_scenarios"] = [d.name for d in group]
        (out / "params.json").write_text(json.dumps(params, indent=2))
    n = sum(len(g) for g in groups)
    print(f"  {set_dir.name}: {n} scenarios, {len(groups)} distinct runout inputs "
          f"(identical inputs run once with their summed probability)")
    (view / "merged_scenarios.json").write_text(json.dumps(
        {g[0].name: [d.name for d in g] for g in groups}, indent=2))


def _write_shot_path_profile(path_profile: str, paths_dir: Path) -> None:
    """The path profile, relabelled for the shot run, as the run's only one."""
    import yaml
    data = yaml.safe_load((RUNOUT / "config" / "paths" / f"{path_profile}.yaml").read_text())
    display = dict(data.get("display") or {})
    display["project_name"] = (display.get("project_name") or path_profile) + SHOT_DISPLAY_SUFFIX
    display["caveats"] = [SHOT_CAVEAT] + list(display.get("caveats") or [])
    data["display"] = display
    paths_dir.mkdir()
    (paths_dir / f"{path_profile}.yaml").write_text(
        "# Generated by src/avachain/runout_step.py for the explosive-shot run\n"
        f"# from runout/config/paths/{path_profile}.yaml; rewritten every run.\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def runout_command(rc: dict, cfg_dir: Path, cycle_date: str, slope: str) -> list[str]:
    # --snow-height set: each day entrains its own set's scour_depth.tif
    # (T_0..T_n come from different snowpack states).
    return [str(RUNOUT / ".venv" / "bin" / "python"), "-m", "scripts", "run",
            "--profile", rc["profile"], "--path", rc["path_profile"],
            "--date", cycle_date.replace("-", ""),
            "--config-dir", str(cfg_dir),
            "--snow-height", "set",
            "--run-name", f"{slope}_{'shots_' if rc.get('shots') else ''}"
                          f"{cycle_date.replace('-', '')}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--toml", required=True, type=Path)
    ap.add_argument("--snapshot-date", default=None,
                    help="release set date YYYY-MM-DD (default: today, or the replay clock)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the set and prepare the tree; print the command, run nothing")
    args = ap.parse_args(argv)

    rc = runout_settings(args.toml)
    if rc is None:
        print(f"No [runout] in {args.toml}: runout not configured for this slope — skipped")
        return 0
    cfg = ProjectConfig.from_toml(args.toml)
    if args.snapshot_date is None:
        from analysis_pipeline import default_snapshot_date
        args.snapshot_date = default_snapshot_date()

    sets = _readable(release_sets(cfg, args.snapshot_date))
    shot_sets = _readable(release_sets(cfg, args.snapshot_date, shots=True))
    if not sets and not shot_sets:
        print(f"ERROR: no readable release set for {args.snapshot_date} under "
              f"{cfg.scenarios_dir / args.snapshot_date} (READY + verified manifest)",
              file=sys.stderr)
        return 2
    py = RUNOUT / ".venv" / "bin" / "python"
    if not py.exists():
        print(f"ERROR: no runout venv at {py.parent.parent}; run install.py "
              f"(or: cd {RUNOUT} && uv sync --frozen)", file=sys.stderr)
        return 2

    runs = []           # (name, settings, command)
    for name, r, run_sets in (("model", rc, sets), ("shots", shot_settings(rc), shot_sets)):
        if not run_sets:
            continue
        cfg_dir = prepare_tree(r, run_sets, args.snapshot_date)
        cmd = runout_command(r, cfg_dir, args.snapshot_date, cfg.slope_name)
        print(f"Runout ({name}) on {', '.join(f'{k}={v}' for k, v in run_sets.items())}")
        print("  " + " ".join(cmd))
        runs.append((name, cmd))
    if args.dry_run:
        return 0 if sets else 2

    rc["lock_file"].parent.mkdir(parents=True, exist_ok=True)
    codes = []
    with open(rc["lock_file"], "a") as lock:
        t0 = time.monotonic()
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"  waiting for {rc['lock_file']} (another runout is running)", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(f"  lock taken after {time.monotonic() - t0:.0f} s", flush=True)
        # A rerun republishes the date's reports: keep the ones forecasters
        # already had (season record, Ron 2026-11-08). Never fails the run.
        import metrics
        metrics.keep_published_reports(cfg, args.snapshot_date)
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)          # the run scripts activate avachain's venv
        # Model first: its report is the one due at 03:00.
        for name, cmd in runs:
            r = subprocess.run(cmd, cwd=RUNOUT, env=env)
            print(f"Runout ({name}) exit {r.returncode}")
            codes.append(r.returncode)
    if not sets:
        print("ERROR: no readable model release set; only the shot sets ran",
              file=sys.stderr)
        return 2
    return next((c for c in codes if c), 0)


def _readable(found: dict) -> dict:
    """The sets that pass handoff.verify; the others are reported and left out."""
    sets = {}
    for label, set_dir in found.items():
        problems = handoff.verify(set_dir)
        if problems:
            print(f"ERROR: release set {set_dir} cannot be read; {label} left out:",
                  file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
        else:
            sets[label] = set_dir
    return sets


if __name__ == "__main__":
    sys.exit(main())
