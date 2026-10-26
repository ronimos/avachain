"""
Release polygon for a known event, from the surveys alone.

Runs avalanche.delineate_crown on the survey pair that brackets the event,
with step_avalanche's own call (defaults, station dHS from the transport
metadata), and keeps the most-eroding region that touches the start zone.
The mapped crown is never used to choose the region.

This is the Jan 18 validation target (Ron, 2026-10-26): it is what the
surveys show slid, where the hand-mapped crowns (6934 m2, and 4550 m2 before
2026-10-25) disagree with each other and with it (IoU 0.65 each). It is
written to cfg.validation_dir, which release_geojsons_for_date prefers over
boundaries_dir; reinit (event_mask_for) never reads it.

    python src/avachain/survey_release_polygon.py \
        --toml slopes/little_prof/slope_config.toml \
        --pair 2026-01-14__2026-01-20 --event-date 2026-01-18
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig, RASTER_EPSG
from pipeline_io import load_dem, load_transport_meta
from avalanche import delineate_crown
from snowpack_io import kml_to_mask


def survey_release_mask(cfg: ProjectConfig, pair_id: str):
    """(mask, regions table, transform) for the release in one survey pair."""
    dem, transform, _ = load_dem(cfg)
    meta = next((m for m in load_transport_meta(cfg) if m['pair_id'] == pair_id), None)
    if meta is None:
        raise SystemExit(f"{pair_id} is not in transport_metadata.json")
    d_a, d_b = pair_id.split('__')
    hs_a = np.clip(np.load(cfg.resampled_dir / f"hs_{d_a}.npy"), 0, None)
    hs_b = np.clip(np.load(cfg.resampled_dir / f"hs_{d_b}.npy"), 0, None)
    regions = delineate_crown(hs_a, hs_b, dem, meta.get('stn_dhs') or 0)

    sz = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)
    in_sz = [(r, int((r['mask'] & sz).sum())) for r in regions]
    in_sz = [(r, n) for r, n in in_sz if n > 0]
    if not in_sz:
        raise SystemExit(f"{pair_id}: no erosion region touches the start zone")
    for r, n in in_sz:
        print(f"  region {r['n_cells']:6d} cells ({n} in start zone)  "
              f"crown={r['n_crown_cells']} flank={r['n_flank_cells']}  "
              f"vol={r['volume_m3']:.0f} m3  mean dHS={r['mean_dhs']:.2f} m")
    best = min(in_sz, key=lambda rn: rn[0]['volume_m3'])[0]   # most erosion
    return best['mask'], transform


def write_geojson(mask: np.ndarray, transform, out_path: Path, props: dict):
    """Vectorise in RASTER_EPSG, with the crs member the mapped crowns carry."""
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union

    geom = unary_union([shape(g) for g, v in
                        shapes(mask.astype(np.uint8), mask=mask, transform=transform)
                        if v == 1])
    gj = {
        "type": "FeatureCollection",
        "name": out_path.stem,
        "crs": {"type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{RASTER_EPSG}"}},
        "features": [{"type": "Feature", "properties": props,
                      "geometry": mapping(geom)}],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gj, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--toml', required=True)
    ap.add_argument('--pair', required=True, help='survey pair, e.g. 2026-01-14__2026-01-20')
    ap.add_argument('--event-date', required=True, help='YYYY-MM-DD, names the output file')
    ap.add_argument('--out-dir', help='default: the slope\'s [paths] validation_dir')
    args = ap.parse_args()

    cfg = ProjectConfig.from_toml(args.toml)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.validation_dir
    if out_dir is None:
        raise SystemExit("no --out-dir and no [paths] validation_dir in the TOML")

    mask, transform = survey_release_mask(cfg, args.pair)
    area = int(mask.sum() * abs(transform.a * transform.e))
    out = out_dir / f"avalanche_release_area_{args.event_date.replace('-', '')}.geojson"
    write_geojson(mask, transform, out, {
        "date": args.event_date,
        "source": f"avalanche.delineate_crown on survey pair {args.pair}",
        "generator": "src/avachain/survey_release_polygon.py",
        "area_m2": area,
    })
    print(f"Wrote {out} ({area} m2)")


if __name__ == '__main__':
    main()
