"""
tier1_restart_check.py — Is SNOWPACK's incremental restart exact?

Tier 1 of the operational replay harness (docs/operational_replay_testing.md
section 7). The replay advances SNOWPACK one tick at a time from restart
files; the season baseline was produced by one continuous run. If those two
do not agree, every number the replay produces is uninterpretable, so this
gates the rest of the module.

The experiment holds everything constant except the restart pattern:

    warmup   template.sno -> run SMET start .. T_start   (one shared .sno)
    A        T_start .. T_end                            in ONE invocation
    B        T_start -> +step -> +step -> .. -> T_end     restarting each time

Both arms start from a byte-identical warmup .sno, read the same SMET files
with the same binary and the same master_config.ini. The only difference is
how many times SNOWPACK stopped and restarted.

The .sno file is the complete state vector SNOWPACK round-trips through, so
comparing it layer by layer is the sharpest available test -- sharper than the
.pro, which is written at PROF_DAYS_BETWEEN resolution and may be appended
rather than rewritten.

Usage
-----
    python src/avachain/tier1_restart_check.py \
        --toml slopes/little_prof/slope_config.toml \
        --clusters 12 --start 2026-02-05 --end 2026-02-12

Exit code is 0 when the arms agree within --tol, 1 otherwise.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig  # noqa: E402

# Columns of the .sno [DATA] layer table, after the timestamp.
SNO_COLS = ["Layer_Thick", "T", "Vol_Frac_I", "Vol_Frac_W", "Vol_Frac_V",
            "Vol_Frac_S", "Rho_S", "Conduc_S", "HeatCapac_S", "rg", "rb",
            "dd", "sp", "mk", "mass_hoar", "ne", "CDot", "metamo"]

# Scalar header fields that are part of the restart state.
SNO_HEADER_NUMERIC = ["HS_Last", "nSnowLayerData", "ErosionLevel",
                      "TimeCountDeltaHS"]

# Three kinds of check, in order of how much they mean.
#
# 1. STRUCTURAL (must be exact). Layer count and the integer-valued markers.
#    If these move, layer identity changed and no tolerance on the continuous
#    fields makes the comparison meaningful.
# 2. CONSERVATION (hard tolerance). Total snow depth. SNOWPACK is free to
#    place internal element boundaries differently after a restart --
#    COMBINE_ELEMENTS merges thin elements and that decision can land
#    differently -- but it is NOT free to lose or gain snow.
# 3. CHARACTERISATION (measured bounds). Per-layer quantities. These are not
#    a correctness claim; they bound the discretisation noise so a CHANGE in
#    behaviour trips the check.
#
# A per-element *relative* metric is useless here: sphericity and dendricity
# live near 0, so a 1e-4 wobble reads as a 26% error.
#
# The characterisation bounds below were MEASURED on 2026-02-05 -> 2026-02-13,
# 8 clusters spread across the domain, 1/3/7 restarts, Snowpack 3.7.1, and set
# ~1.3x above the worst observed value:
#
#   Layer_Thick  worst 1.55e-2 m  -- in 1 of 8 clusters, ~7 adjacent layers
#                                    redistributed with total depth conserved
#                                    to 1.9e-4 m. 7 of 8 clusters: < 3e-4 m.
#   Vol_Frac_I/V worst 3.66e-2    -- ice vs air fraction, which sum to 1, so
#                                    this is one quantity: ~1% of a layer's
#                                    density in the worst layer.
#   T            worst 3.00e-2 K
#   dd           worst 5.86e-2
#
# Do NOT widen these to make a run pass. Re-measure, and if the numbers moved,
# find out why first -- that is the whole point of the check.
TOL_ABS = {
    "Layer_Thick": 2.0e-2,   # m
    "T":           5.0e-2,   # K
    "Vol_Frac_I":  5.0e-2,
    "Vol_Frac_W":  5.0e-2,
    "Vol_Frac_V":  5.0e-2,
    "rg":          1.0e-2,   # mm
    "rb":          1.0e-2,   # mm
    "dd":          1.0e-1,
    "sp":          1.0e-1,
    "metamo":      1.0,
    "CDot":        50.0,
}
TOL_HS = 1e-3            # m, on HS_Last
TOL_DEPTH_SUM = 1e-3     # m, on sum(Layer_Thick) -- the conservation check

EXACT_COLS = ["mk", "ne", "Rho_S", "Conduc_S", "HeatCapac_S", "Vol_Frac_S"]


# ---------------------------------------------------------------------------
# .sno parsing
# ---------------------------------------------------------------------------

def parse_sno(path: Path) -> dict:
    """Parse a SNOWPACK .sno into {header: dict, layers: ndarray, dates: list}."""
    header: dict = {}
    rows: list[list[str]] = []
    dates: list[str] = []
    in_data = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s == "[DATA]":
            in_data = True
            continue
        if not in_data:
            if "=" in s and not s.startswith("["):
                k, v = s.split("=", 1)
                header[k.strip()] = v.strip()
            continue
        if not s:
            continue
        parts = s.split()
        dates.append(parts[0])
        rows.append(parts[1:])

    layers = (np.array(rows, dtype=float) if rows
              else np.zeros((0, len(SNO_COLS)), dtype=float))
    return {"header": header, "layers": layers, "dates": dates}


def compare_sno(a: Path, b: Path) -> dict:
    """Compare two .sno files on structure and physical state.

    Returns a dict of measurements plus `ok`. No relative-difference metric:
    see TOL_ABS for why.
    """
    A, B = parse_sno(a), parse_sno(b)
    out: dict = {"cluster": a.name.split("_cluster_")[0], "ok": True,
                 "notes": [], "per_col_abs": {}}

    na, nb = A["layers"].shape[0], B["layers"].shape[0]
    out["n_layers_a"], out["n_layers_b"] = na, nb
    out["layers_match"] = (na == nb)

    pa = A["header"].get("ProfileDate")
    pb = B["header"].get("ProfileDate")
    if pa != pb:
        out["ok"] = False
        out["notes"].append(f"ProfileDate differs: {pa} vs {pb}")

    try:
        hs_a = float(A["header"].get("HS_Last", "nan"))
        hs_b = float(B["header"].get("HS_Last", "nan"))
    except ValueError:
        hs_a = hs_b = float("nan")
    out["hs_a"], out["hs_b"] = hs_a, hs_b
    out["hs_diff"] = abs(hs_a - hs_b)
    if np.isfinite(out["hs_diff"]) and out["hs_diff"] > TOL_HS:
        out["ok"] = False
        out["notes"].append(
            f"HS_Last differs by {out['hs_diff']:.3e} m > {TOL_HS:.0e}")

    if na != nb:
        out["ok"] = False
        out["notes"].append(f"LAYER COUNT differs: {na} vs {nb} "
                            f"(structural — tolerances are meaningless)")
        return out
    if na == 0:
        out["notes"].append("both have zero snow layers")
        return out

    # Conservation: internal boundaries may move, total depth may not.
    ti = SNO_COLS.index("Layer_Thick")
    sa, sb = float(A["layers"][:, ti].sum()), float(B["layers"][:, ti].sum())
    out["depth_sum_a"], out["depth_sum_b"] = sa, sb
    out["depth_sum_diff"] = abs(sa - sb)
    if out["depth_sum_diff"] > TOL_DEPTH_SUM:
        out["ok"] = False
        out["notes"].append(
            f"total depth not conserved: {out['depth_sum_diff']:.3e} m "
            f"> {TOL_DEPTH_SUM:.0e} ({sa:.5f} vs {sb:.5f})")

    diff = np.abs(A["layers"] - B["layers"])
    out["n_layers_differing"] = int((diff > 0).any(axis=1).sum())
    out["n_layers_thick_1mm"] = int((diff[:, ti] > 1e-3).sum())

    for i, col in enumerate(SNO_COLS):
        if i >= diff.shape[1]:
            break
        m = float(diff[:, i].max())
        out["per_col_abs"][col] = m
        if col in EXACT_COLS:
            if m != 0.0:
                out["ok"] = False
                out["notes"].append(
                    f"{col} must be exact but differs by {m:.3e} "
                    f"(layer identity changed)")
        elif col in TOL_ABS and m > TOL_ABS[col]:
            out["ok"] = False
            out["notes"].append(
                f"{col} differs by {m:.3e} > tol {TOL_ABS[col]:.0e}")

    out["max_thick"] = out["per_col_abs"].get("Layer_Thick", float("nan"))
    out["max_temp"] = out["per_col_abs"].get("T", float("nan"))
    return out


# ---------------------------------------------------------------------------
# SNOWPACK invocation (mirrors slopes/<slope>/run_snowpack.sh)
# ---------------------------------------------------------------------------

def write_cluster_ini(ini_path: Path, master_cfg: Path, smet_dir: Path,
                      snow_in: Path, out_dir: Path, cid: str) -> None:
    ini_path.write_text(
        f"IMPORT_BEFORE = {master_cfg}\n\n"
        f"[Input]\n"
        f"METEOPATH  = {smet_dir}\n"
        f"SNOWPATH   = {snow_in}\n"
        f"METEOFILE1 = {cid}\n"
        f"SNOWFILE1  = {cid}\n\n"
        f"[Output]\n"
        f"METEOPATH  = {out_dir}\n"
        f"SNOWPATH   = {out_dir}\n"
        f"EXPERIMENT = {cid}\n"
    )


def run_snowpack(binary: Path, lib_dir: str, ini: Path, bdate: str,
                 edate: str, log: Path) -> tuple[int, str]:
    env_lib = f"{lib_dir}:{Path(binary).parent}" if lib_dir else ""
    cmd = [str(binary), "-c", str(ini), "-b", bdate, "-e", edate]
    import os
    env = dict(os.environ)
    if env_lib:
        env["LD_LIBRARY_PATH"] = env_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log.write_text(proc.stdout + "\n" + proc.stderr)
    ok = "done!" in proc.stdout or proc.returncode == 0
    return (0 if ok else 1), proc.stdout


class Arm:
    """One experiment arm: its own input/snow and output trees."""

    def __init__(self, root: Path, name: str):
        self.name = name
        self.root = root / name
        self.snow_in = self.root / "input" / "snow"
        self.out = self.root / "output"
        for d in (self.snow_in, self.out):
            d.mkdir(parents=True, exist_ok=True)

    def res_sno(self, cid: str) -> Path:
        return self.out / f"{cid}_{cid}.sno"

    def seed_from(self, sno: Path, cid: str) -> None:
        shutil.copy(sno, self.snow_in / f"{cid}.sno")

    def restore_restart(self, cid: str) -> bool:
        """Copy the output .sno back to input/snow, as run_snowpack.sh does."""
        r = self.res_sno(cid)
        if r.exists():
            shutil.copy(r, self.snow_in / f"{cid}.sno")
            return True
        return False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--toml", required=True)
    p.add_argument("--clusters", type=int, default=12,
                   help="How many clusters to test (default 12)")
    p.add_argument("--cluster-ids", default=None,
                   help="Comma-separated cluster ids, overrides --clusters")
    p.add_argument("--start", default="2026-02-05",
                   help="T_start, the shared restart point (default 2026-02-05)")
    p.add_argument("--end", default="2026-02-12", help="T_end (default 2026-02-12)")
    p.add_argument("--hour", default="12:00", help="Time of day for both bounds")
    p.add_argument("--steps", default="1",
                   help="Comma-separated incremental step sizes in days. One "
                        "arm per value, each compared to the continuous run, "
                        "so drift can be read against restart count "
                        "(e.g. --steps 1,2,4). Default 1.")
    p.add_argument("--workdir", default=None,
                   help="Scratch dir (default <output_dir>/tier1_restart)")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--keep-warmup", action="store_true",
                   help="Reuse an existing warmup .sno set instead of rerunning")
    args = p.parse_args()

    cfg = ProjectConfig.from_toml(Path(args.toml))
    # ProjectConfig does not carry the binary or lib dir -- they live only in
    # the TOML's [snowpack] section, where install.py reads them from.
    import tomllib
    with open(args.toml, "rb") as fh:
        sp_cfg = tomllib.load(fh).get("snowpack", {})

    work = Path(args.workdir) if args.workdir else cfg.output_dir / "tier1_restart"
    master_cfg = cfg.project_dir / "slopes" / cfg.slope_name / "config" / "master_config.ini"
    template = cfg.slope_dir / "input" / "snow" / "template.sno"
    binary = Path(sp_cfg.get("binary", ""))
    lib_dir = sp_cfg.get("lib_dir", "") or ""

    for f in (master_cfg, template, binary):
        if not Path(f).exists():
            sys.exit(f"ERROR: required file not found: {f}")

    # --- select clusters -------------------------------------------------
    all_smets = sorted(cfg.smet_dir.glob("cluster_*.smet"))
    if not all_smets:
        sys.exit(f"ERROR: no SMET files in {cfg.smet_dir}")
    if args.cluster_ids:
        want = {f"cluster_{int(x):04d}" for x in args.cluster_ids.split(",")}
        smets = [s for s in all_smets if s.stem in want]
    else:
        # Spread across the domain rather than taking the first N, so the
        # sample is not all one corner of the slope.
        idx = np.linspace(0, len(all_smets) - 1, args.clusters).astype(int)
        smets = [all_smets[i] for i in sorted(set(idx))]
    cids = [s.stem for s in smets]

    smet_start = _first_smet_ts(smets[0])
    t_start = f"{args.start}T{args.hour}"
    t_end = f"{args.end}T{args.hour}"

    print("=" * 70)
    print("  Tier 1 — SNOWPACK incremental restart exactness")
    print("=" * 70)
    print(f"  binary     : {binary}  ({_binary_version(binary, lib_dir)})")
    print(f"  clusters   : {len(cids)}  ({cids[0]} .. {cids[-1]})")
    print(f"  SMET start : {smet_start}")
    print(f"  window     : {t_start}  ->  {t_end}")
    print(f"  steps      : {args.steps} d (one arm each)")
    print(f"  tolerances : HS {TOL_HS:.0e} m, thick "
          f"{TOL_ABS['Layer_Thick']:.0e} m, T {TOL_ABS['T']:.0e} K; "
          f"exact: {','.join(EXACT_COLS)}")
    print(f"  workdir    : {work}")
    print()

    # --- shared SMET dir -------------------------------------------------
    smet_dir = work / "smet"
    smet_dir.mkdir(parents=True, exist_ok=True)
    for s in smets:
        tgt = smet_dir / s.name
        if not tgt.exists():
            shutil.copy(s, tgt)

    # --- warmup: one shared restart state at T_start ---------------------
    warm = Arm(work, "warmup")
    need_warmup = not args.keep_warmup or not all(
        warm.res_sno(c).exists() for c in cids)
    if need_warmup:
        print(f">>> Warmup: template -> {t_start}  (shared restart state)")
        for c in cids:
            _seed_template(template, warm.snow_in / f"{c}.sno", c,
                           smet_dir / f"{c}.smet")
        rc = _run_all(cids, warm, binary, lib_dir, master_cfg, smet_dir,
                      smet_start, t_start, args.jobs)
        if rc:
            sys.exit("ERROR: warmup failed; see logs in "
                     f"{warm.out}")
        print(f"    warmup .sno written for {len(cids)} clusters\n")
    else:
        print(">>> Warmup: reusing existing .sno set\n")

    # --- arm A: continuous ----------------------------------------------
    A = Arm(work, "continuous")
    print(f">>> Arm A (continuous): {t_start} -> {t_end} in one invocation")
    for c in cids:
        A.seed_from(warm.res_sno(c), c)
    if _run_all(cids, A, binary, lib_dir, master_cfg, smet_dir,
                t_start, t_end, args.jobs):
        sys.exit(f"ERROR: continuous arm failed; see logs in {A.out}")
    print("    done\n")

    # --- arms B_k: one per step size -------------------------------------
    step_sizes = [int(x) for x in args.steps.split(",")]
    arms: list[tuple[int, int, Arm]] = []   # (step_days, n_restarts, arm)
    for step in step_sizes:
        B = Arm(work, f"incremental_s{step}")
        bounds = _step_bounds(args.start, args.end, step, args.hour)
        print(f">>> Arm B (step {step} d): {len(bounds)} SNOWPACK "
              f"invocation(s), {len(bounds) - 1} restart(s)")
        for c in cids:
            B.seed_from(warm.res_sno(c), c)
        for i, (b, e) in enumerate(bounds, 1):
            if i > 1:
                for c in cids:
                    if not B.restore_restart(c):
                        sys.exit(f"ERROR: no restart .sno for {c} at step {i}")
            if _run_all(cids, B, binary, lib_dir, master_cfg, smet_dir,
                        b, e, args.jobs):
                sys.exit(f"ERROR: step {i} ({b} -> {e}) failed; logs in {B.out}")
        print(f"    done ({bounds[0][0]} -> {bounds[-1][1]})")
        arms.append((step, len(bounds) - 1, B))
    print()

    # --- compare ---------------------------------------------------------
    print(">>> Comparing final restart state (.sno) against the continuous run")
    all_results = []
    for step, n_restarts, B in arms:
        res = [compare_sno(A.res_sno(c), B.res_sno(c)) for c in cids]
        all_results.append((step, n_restarts, res))
    return _report(all_results, work)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _first_smet_ts(smet: Path) -> str:
    in_data = False
    for line in smet.read_text().splitlines():
        s = line.strip()
        if s == "[DATA]":
            in_data = True
            continue
        if in_data and s:
            return s.split()[0][:16]
    raise ValueError(f"no data rows in {smet}")


def _binary_version(binary: Path, lib_dir: str) -> str:
    import os
    env = dict(os.environ)
    if lib_dir:
        env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True,
                             text=True, env=env).stdout
        return out.strip().splitlines()[0]
    except Exception:
        return "version unknown"


def _seed_template(template: Path, dest: Path, cid: str, smet: Path) -> None:
    """Instantiate template.sno for one cluster, as run_snowpack.sh does."""
    hdr = {}
    for line in smet.read_text().splitlines():
        if line.strip() == "[DATA]":
            break
        if "=" in line:
            k, v = line.split("=", 1)
            hdr[k.strip()] = v.strip()
    txt = (template.read_text()
           .replace("WRFPT", cid).replace("WRFNAME", cid)
           .replace("WRFLAT", hdr.get("latitude", "0"))
           .replace("WRFLON", hdr.get("longitude", "0"))
           .replace("WRFELEV", hdr.get("altitude", "0")))
    dest.write_text(txt)


def _step_bounds(start: str, end: str, step_days: int, hour: str):
    """[(b,e), ...] covering start..end in step_days increments."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    cur = s
    while cur < e:
        nxt = min(cur + pd.Timedelta(days=step_days), e)
        out.append((f"{cur.date()}T{hour}", f"{nxt.date()}T{hour}"))
        cur = nxt
    return out


def _run_all(cids, arm: Arm, binary, lib_dir, master_cfg, smet_dir,
             bdate, edate, jobs) -> int:
    def one(cid):
        ini = arm.out / f"{cid}.ini"
        write_cluster_ini(ini, master_cfg, smet_dir, arm.snow_in, arm.out, cid)
        rc, _ = run_snowpack(binary, lib_dir, ini, bdate, edate,
                             arm.out / f"{cid}.log")
        return cid, rc

    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for cid, rc in ex.map(one, cids):
            if rc:
                failures.append(cid)
    if failures:
        print(f"    FAILED: {', '.join(failures[:8])}"
              f"{' ...' if len(failures) > 8 else ''}")
    return len(failures)


def _report(all_results, work: Path) -> int:
    overall_ok = True

    for step, n_restarts, results in all_results:
        print()
        print(f"  --- step {step} d, {n_restarts} restart(s) "
              f"{'-' * 34}")
        print(f"  {'cluster':14s} {'layers':>10s} {'differ':>7s} "
              f"{'ΔHS (m)':>11s} {'Δthick (m)':>11s} {'ΔT (K)':>10s}  ok")
        for r in results:
            layers = (f"{r['n_layers_a']}" if r["layers_match"]
                      else f"{r['n_layers_a']}/{r['n_layers_b']}!")
            print(f"  {r['cluster'][:14]:14s} {layers:>10s} "
                  f"{r.get('n_layers_differing', 0):>7d} "
                  f"{r.get('hs_diff', float('nan')):11.3e} "
                  f"{r.get('max_thick', float('nan')):11.3e} "
                  f"{r.get('max_temp', float('nan')):10.3e}  "
                  f"{'yes' if r['ok'] else 'NO'}")
            for n in r["notes"]:
                print(f"      ! {n}")
        if not all(r["ok"] for r in results):
            overall_ok = False

    # Drift against restart count: the question the replay actually depends on.
    if len(all_results) > 1:
        print()
        print("  --- drift vs restart count " + "-" * 36)
        print(f"  {'restarts':>9s} {'max ΔHS (m)':>13s} "
              f"{'max Δthick (m)':>15s} {'max ΔT (K)':>12s}")
        for step, n_restarts, results in sorted(all_results, key=lambda x: x[1]):
            f = lambda k: max((r.get(k, 0.0) or 0.0) for r in results)
            print(f"  {n_restarts:>9d} {f('hs_diff'):13.3e} "
                  f"{f('max_thick'):15.3e} {f('max_temp'):12.3e}")
        # The assertion the replay actually rests on: drift must grow more
        # slowly than the restart count. Quantisation accumulates like a
        # random walk (~sqrt N); a diverging trajectory would grow at least
        # linearly. A 32-tick replay is only interpretable if this holds.
        pts = sorted(((n, max((x.get("hs_diff", 0.0) or 0.0) for x in r))
                      for _, n, r in all_results if n > 0), key=lambda t: t[0])
        if len(pts) >= 2:
            (n0, d0), (n1, d1) = pts[0], pts[-1]
            growth = (d1 / d0) if d0 > 0 else float("inf")
            ratio = n1 / n0
            print()
            print(f"  {n0} -> {n1} restarts ({ratio:.1f}x): ΔHS grew "
                  f"{growth:.2f}x")
            if growth <= ratio:
                print(f"  SUB-LINEAR — consistent with .sno quantisation "
                      f"accumulating, not divergence.")
            else:
                print(f"  SUPER-LINEAR ({growth:.2f}x > {ratio:.1f}x) — the "
                      f"trajectory is diverging.")
                print("  A per-tick replay would inject drift that looks "
                      "like physics.")
                overall_ok = False

    n_total = sum(len(r) for _, _, r in all_results)
    n_ok = sum(1 for _, _, r in all_results for x in r if x["ok"])
    structural = sum(1 for _, _, r in all_results
                     for x in r if not x["layers_match"])
    print()
    print("=" * 70)
    print(f"  comparisons        : {n_total}")
    print(f"  within tolerance   : {n_ok}/{n_total}")
    print(f"  layer-count breaks : {structural}")
    print(f"  artifacts          : {work}")
    print("=" * 70)

    if overall_ok:
        print("\n  TIER 1 PASS — incremental restart reproduces the continuous")
        print("  run to within .sno storage precision, with identical layer")
        print("  structure. The replay's day-by-day advance is interpretable.")
        return 0

    print("\n  TIER 1 FAIL — incremental restart does not reproduce the")
    print("  continuous run within tolerance. Do not interpret replay")
    print("  comparisons until this is understood: per-tick restart would")
    print("  inject drift that looks like physics.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
