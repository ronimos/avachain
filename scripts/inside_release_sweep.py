#!/usr/bin/env python3
"""Inside-release trigger sweep: propagation scored given a trigger in the slab
that released (Ron, 2026-11-16).

E[IoU] over the five model triggers mixes two questions: did the model pick
good trigger spots, and does a crack started there spread to the right
extent. Release-model switches such as SLAB_CRACK_MODE, BFS_ADJACENCY and
POLYGON_PIECE only touch the second one. This script scores it directly:

  positives  every cluster whose pixels lie mostly inside the observed release
             polygon (and the start zone). The slab did fracture there, so a
             crack started there should spread, and once propagation is
             supercritical the release should not depend much on where it
             started.
  negatives  the observed non-triggers (data/validation/ski_nontriggers_*.json):
             people skied there without triggering, so nothing should spread.

Every point goes through make_release_polygon_2d as a forced shot (no
initiation gate, rectangle_fallback=False), at size factor 1 (with distance
caps off the size factors scale nothing in the BFS; checked with
--check-size). For a point the trigger's own tensile/toughness gate stops,
the polygon is also computed with only that gate removed ("ungated"), so
propagation from there is still scored; the BFS's per-cluster gate stays on.

Scores, per mode and per class (interior / edge = centroid within --edge-m of
the observed boundary / ski):
  released      fraction that gives a polygon with the gate (positives should
                be ~1, ski 0), and which gate stopped the rest
  IoU           median and IQR of each released polygon against the observed
  consensus     IoU of the pixels covered by >= half of the released polygons
                against the observed, and the mean pairwise IoU between the
                polygons (1 = the release does not depend on the trigger)

Still one event. The points are spatially correlated, so they reduce variance
within Jan 18 but add no independent evidence, and Jan 18 is a calibration
target: report this as calibration data, never as validation.

    python scripts/inside_release_sweep.py --toml slopes/little_prof/slope_config.toml \\
        --date 2026-01-18 --features-dir <dir> --modes shear opening --jobs 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mproc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import model_params as mp            # noqa: E402
import release_geometry as _rg       # noqa: E402

G = {}   # inputs, set in the parent before the pool forks


def load_inputs(toml, date, features_dir, ski_json):
    import rasterio.features
    import analysis_pipeline as ap
    import calibration
    from config import ProjectConfig
    from snowpack_io import kml_to_mask

    cfg = ProjectConfig.from_toml(toml)
    applied = calibration.apply(cfg)
    dem, transform, _, _ = ap._load_dem(cfg)
    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))
    fdir = Path(features_dir) if features_dir else cfg.analysis_dir
    feats = pd.read_csv(fdir / f"all_start_zone_features_{date}.csv", index_col=0)
    mel = pd.read_csv(fdir / f"meloche_features_all_{date}.csv", index_col=0)
    observed = ap._load_observed_release_polygon(cfg, date)
    sz = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)
    obs_mask = rasterio.features.rasterize([(observed, 1)], out_shape=dem.shape,
                                           transform=transform, dtype=np.uint8).astype(bool)
    ski = json.loads(Path(ski_json).read_text()) if ski_json else None
    G.update(cfg=cfg, calibration=applied, dem=dem, transform=transform,
             cluster_map=cluster_map, snap_features=feats[~feats.index.duplicated()],
             meloche_df=mel[~mel.index.duplicated()], observed=observed,
             start_zone_mask=sz, obs_mask=obs_mask, ski=ski)


def inside_clusters(min_frac=0.5):
    """Clusters with >= min_frac of their pixels inside the observed polygon
    and inside the start zone, with the centroid's distance to the observed
    boundary (m)."""
    from shapely.geometry import Point
    from release_geometry import pixel_to_utm
    cm, obs, sz = G["cluster_map"], G["obs_mask"], G["start_zone_mask"]
    ids = np.unique(cm[obs & (cm > 0)])
    rows = []
    for cid in ids:
        px = cm == cid
        n = px.sum()
        f_obs, f_sz = (px & obs).sum() / n, (px & sz).sum() / n
        if f_obs < min_frac or f_sz < min_frac:
            continue
        r, c = np.argwhere(px).mean(axis=0)
        x, y = pixel_to_utm(int(r), int(c), G["transform"])
        rows.append({"cid": int(cid), "frac_inside": round(float(f_obs), 3),
                     "edge_dist_m": round(float(G["observed"].exterior.distance(Point(x, y))), 1)})
    return rows


def ski_clusters():
    import analysis_pipeline as ap
    pts = [tuple(p) for p in G["ski"]["points_lat_lon"]]
    return ap._clusters_at_points(pts, G["cluster_map"], G["transform"])


def _a_ca(cid):
    A_ca = mp.A_CA_FALLBACK_M
    mel = G["meloche_df"]
    if "A_ca_brittle" in mel.columns and cid in mel.index:
        v = float(mel.loc[cid, "A_ca_brittle"])
        if np.isfinite(v):
            A_ca = min(v, mp.A_CA_MAX_M)
    return A_ca


def _polygon(cid, size_factor=1.0):
    state = {}
    poly = _rg.make_release_polygon_2d(
        trigger_cluster_id=cid, A_ca=_a_ca(cid), meloche_df=G["meloche_df"],
        cluster_map=G["cluster_map"], dem=G["dem"], transform=G["transform"],
        start_zone_mask=G["start_zone_mask"], snap_features=G["snap_features"],
        size_factor=size_factor, stauchwall_deg=G["cfg"].stauchwall_deg,
        max_clusters=G["cfg"].max_bfs_clusters, state_out=state,
        rectangle_fallback=False)
    return poly, state


def run_one(job):
    """One (mode, cid): gated polygon and, if the trigger gate stopped it, the
    ungated one. Polygons travel back as WKB."""
    import contextlib, io
    mode, cid, cls = job
    mp.SLAB_CRACK_MODE = mode
    out = {"mode": mode, "cid": cid, "class": cls}
    s_t = _rg.cluster_tensile_support(cid, G["meloche_df"], G["snap_features"])
    out["trigger_s_t"] = float(s_t) if np.isfinite(s_t) else None
    with contextlib.redirect_stdout(io.StringIO()):
        poly, state = _polygon(cid)
        out["outcome"] = ("released" if poly is not None else
                          "trigger_gate" if state.get("trigger_tensile_failure") else
                          "no_propagation")
        out["wkb"] = poly.wkb if poly is not None else None
        out["wkb_ungated"] = out["wkb"]
        if out["outcome"] == "trigger_gate":
            orig = _rg.cluster_tensile_support
            _rg.cluster_tensile_support = lambda *a, **k: np.inf
            try:
                p2, _ = _polygon(cid)
            finally:
                _rg.cluster_tensile_support = orig
            out["wkb_ungated"] = p2.wkb if p2 is not None else None
    return out


def _iou(a, b):
    if a is None or a.is_empty:
        return 0.0
    return a.intersection(b).area / a.union(b).area


def _masks(polys):
    import rasterio.features
    return [rasterio.features.rasterize([(p, 1)], out_shape=G["dem"].shape,
                                        transform=G["transform"], dtype=np.uint8).astype(bool)
            for p in polys]


def _consensus(polys):
    """(IoU of the >= half-coverage pixels vs observed, mean pairwise IoU, frequency map)."""
    if not polys:
        return None, None, None
    m = _masks(polys)
    stack = np.stack(m).reshape(len(m), -1)
    freq = stack.mean(axis=0).reshape(G["dem"].shape)
    cons = freq >= 0.5
    obs = G["obs_mask"]
    cons_iou = float((cons & obs).sum() / max((cons | obs).sum(), 1))
    if len(m) < 2:
        return cons_iou, None, freq
    s = stack.astype(np.float32)
    inter = s @ s.T
    area = np.diag(inter)
    union = area[:, None] + area[None, :] - inter
    iu = np.triu_indices(len(m), k=1)
    pair = float(np.mean(inter[iu] / np.maximum(union[iu], 1)))
    return cons_iou, pair, freq


def summarise(rows, edge_m):
    from shapely import wkb as _wkb
    obs = G["observed"]
    for r in rows:
        r["poly"] = _wkb.loads(r["wkb"]) if r["wkb"] else None
        r["poly_ungated"] = _wkb.loads(r["wkb_ungated"]) if r["wkb_ungated"] else None
        r["iou"] = _iou(r["poly"], obs)
        r["iou_ungated"] = _iou(r["poly_ungated"], obs)
        r["area_m2"] = r["poly"].area if r["poly"] is not None else 0.0
        r["area_ungated_m2"] = r["poly_ungated"].area if r["poly_ungated"] is not None else 0.0
    out, freqs = [], {}
    for mode in dict.fromkeys(r["mode"] for r in rows):
        for cls in ("interior", "edge", "inside", "ski", "model"):
            sel = [r for r in rows if r["mode"] == mode and
                   (r["class"] == cls or (cls == "inside" and r["class"] in ("interior", "edge")))]
            if not sel:
                continue
            rec = {"mode": mode, "class": cls, "n": len(sel)}
            for k in ("released", "trigger_gate", "no_propagation"):
                rec[f"n_{k}"] = sum(r["outcome"] == k for r in sel)
            for tag, pk, ik in (("", "poly", "iou"), ("_ungated", "poly_ungated", "iou_ungated")):
                rel = [r for r in sel if r[pk] is not None]
                rec[f"n_poly{tag}"] = len(rel)
                ious = np.array([r[ik] for r in rel]) if rel else np.array([])
                if len(ious):
                    rec[f"iou_median{tag}"] = float(np.median(ious))
                    rec[f"iou_p25{tag}"] = float(np.percentile(ious, 25))
                    rec[f"iou_p75{tag}"] = float(np.percentile(ious, 75))
                    rec[f"area_median{tag}_m2"] = float(np.median([r[pk].area for r in rel]))
                if cls in ("inside", "ski"):
                    c, p, f = _consensus([r[pk] for r in rel])
                    rec[f"consensus_iou{tag}"] = c
                    rec[f"pairwise_iou{tag}"] = p
                    if cls == "inside":
                        freqs[(mode, tag or "_gated")] = f
            out.append(rec)
    return out, freqs


def plot(freqs, rows, path, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.plot import plotting_extent
    keys = sorted(freqs)
    if not keys:
        return
    obs_b = G["observed"].exterior.xy
    sz = G["start_zone_mask"]
    rr, cc = np.where(sz)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    t = G["transform"]
    ext = (t.c + c0 * t.a, t.c + c1 * t.a, t.f + r1 * t.e, t.f + r0 * t.e)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 5.6), squeeze=False)
    from release_geometry import pixel_to_utm
    for ax, k in zip(axes[0], keys):
        f = freqs[k]
        if f is None:
            ax.set_axis_off(); continue
        im = ax.imshow(np.where(sz, f, np.nan)[r0:r1, c0:c1], extent=ext,
                       vmin=0, vmax=1, cmap="viridis", origin="upper")
        mode, tag = k
        for r in rows:
            if r["mode"] != mode or r["class"] not in ("interior", "edge", "ski"):
                continue
            px = np.argwhere(G["cluster_map"] == r["cid"]).mean(axis=0)
            x, y = pixel_to_utm(int(px[0]), int(px[1]), t)
            col = {"released": "#d62728" if r["class"] == "ski" else "#ffffff",
                   "trigger_gate": "#ff7f0e", "no_propagation": "#7f7f7f"}[r["outcome"]]
            ax.plot(x, y, "o" if r["class"] != "ski" else "^", ms=3, color=col,
                    mec="k", mew=0.3)
        import matplotlib.patheffects as pe
        ax.plot(*obs_b, color="w", lw=1.8, zorder=5,
                path_effects=[pe.Stroke(linewidth=3.2, foreground="k"), pe.Normal()])
        ax.set_title(f"{mode} — {'gate on' if tag == '_gated' else 'trigger gate off'}"
                     f"{title_extra}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8,
                 label="fraction of inside-release triggers covering the pixel")
    fig.text(0.01, 0.01, "white line: observed release · dots: inside triggers "
             "(white released, orange trigger gate, grey no propagation) · "
             "triangles: ski non-triggers (red = released)", fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--toml", required=True)
    ap_.add_argument("--date", default="2026-01-18")
    ap_.add_argument("--features-dir")
    ap_.add_argument("--ski", default=str(REPO / "data" / "validation" / "ski_nontriggers_20260118.json"))
    ap_.add_argument("--modes", nargs="+", default=list(mp.SLAB_CRACK_MODES),
                     choices=mp.SLAB_CRACK_MODES)
    ap_.add_argument("--model-triggers", nargs="*", type=int, default=[],
                     help="the day's model triggers, scored alongside for reference")
    ap_.add_argument("--downslope-slide", action=argparse.BooleanOptionalAction, default=None,
                     help="override model_params.USE_DOWNSLOPE_SLIDE")
    ap_.add_argument("--edge-m", type=float, default=10.0)
    ap_.add_argument("--subsample", type=int, default=1,
                     help="use every n-th inside cluster (the release is nearly trigger-"
                          "invariant, so a knob sweep can use fewer); ski and model all kept")
    ap_.add_argument("--jobs", type=int, default=8)
    ap_.add_argument("--out", type=Path, required=True)
    ap_.add_argument("--check-size", nargs="*", type=float, default=[],
                     help="also check, on 5 inside triggers, that these size factors "
                          "give the size-1 polygon")
    a = ap_.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load_inputs(a.toml, a.date, a.features_dir, a.ski)
    if a.downslope_slide is not None:
        mp.USE_DOWNSLOPE_SLIDE = a.downslope_slide
    print(f"calibration {G['cfg'].calibration.get('version') if G['cfg'].calibration else None}: "
          f"{G['calibration']}; observed {G['observed'].area:.0f} m2", flush=True)

    inside = inside_clusters()
    for r in inside:
        r["class"] = "interior" if r["edge_dist_m"] >= a.edge_m else "edge"
    ski = ski_clusters() if G["ski"] else []
    print(f"inside-release triggers: {len(inside)} "
          f"({sum(r['class'] == 'interior' for r in inside)} interior, "
          f"{sum(r['class'] == 'edge' for r in inside)} edge < {a.edge_m:g} m); "
          f"ski clusters: {len(ski)}", flush=True)
    pd.DataFrame(inside).to_csv(a.out / "inside_clusters.csv", index=False)
    if a.subsample > 1:
        inside = inside[::a.subsample]
        print(f"  subsample 1/{a.subsample}: {len(inside)} inside triggers", flush=True)

    if a.check_size:
        import contextlib, io
        mp.SLAB_CRACK_MODE = a.modes[0]
        for cid in [r["cid"] for r in inside][::max(1, len(inside) // 5)][:5]:
            with contextlib.redirect_stdout(io.StringIO()):
                base, _ = _polygon(cid)
                for sf in a.check_size:
                    p, _ = _polygon(cid, sf)
                    same = (base is None and p is None) or (
                        base is not None and p is not None and base.equals(p))
                    print(f"  size check cid {cid} x{sf:g}: "
                          f"{'same' if same else 'DIFFERENT'}", file=sys.stderr)

    jobs = [(m, r["cid"], r["class"]) for m in a.modes for r in inside]
    jobs += [(m, c, "ski") for m in a.modes for c in ski]
    jobs += [(m, c, "model") for m in a.modes for c in a.model_triggers]
    with mproc.get_context("fork").Pool(a.jobs) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=2), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["mode"], r["class"], r["cid"]))
    summary, freqs = summarise(rows, a.edge_m)

    edge = {r["cid"]: r["edge_dist_m"] for r in inside}
    pd.DataFrame([{k: r.get(k) for k in ("mode", "class", "cid", "outcome", "trigger_s_t",
                                         "area_m2", "iou", "area_ungated_m2", "iou_ungated")}
                  | {"edge_dist_m": edge.get(r["cid"])} for r in rows]
                 ).to_csv(a.out / "triggers.csv", index=False)
    meta = {"toml": str(a.toml), "date": a.date, "features_dir": a.features_dir,
            "calibration": G["calibration"],
            "calibration_version": (G["cfg"].calibration or {}).get("version"),
            "observed_m2": G["observed"].area, "edge_m": a.edge_m,
            "downslope_slide": mp.USE_DOWNSLOPE_SLIDE,
            "seconds": round(time.time() - t0)}
    (a.out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=1))

    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
    lines = ["| mode | class | n | released (gate) | stopped: trigger gate / no prop. | "
             "IoU median [IQR] | area median m² | consensus IoU | pairwise IoU | "
             "gate off: polygons | gate off: IoU median | gate off: consensus / pairwise |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(
            f"| {s['mode']} | {s['class']} | {s['n']} | {s['n_released']} | "
            f"{s['n_trigger_gate']} / {s['n_no_propagation']} | "
            f"{f(s.get('iou_median'))} [{f(s.get('iou_p25'))}–{f(s.get('iou_p75'))}] | "
            f"{f(s.get('area_median_m2'), 0)} | {f(s.get('consensus_iou'))} | "
            f"{f(s.get('pairwise_iou'))} | {s['n_poly_ungated']} | "
            f"{f(s.get('iou_median_ungated'))} | "
            f"{f(s.get('consensus_iou_ungated'))} / {f(s.get('pairwise_iou_ungated'))} |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(freqs, rows, a.out / "frequency_maps.png",
         title_extra=", slide on" if mp.USE_DOWNSLOPE_SLIDE else "")
    print(f"done in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
