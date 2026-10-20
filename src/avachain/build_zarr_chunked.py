"""
build_zarr_chunked.py — CLI wrapper around snowpack_io.build_zarr_cache.

All logic lives in snowpack_io. This file is a thin entry point so the
existing nohup command still works unchanged.

Usage:
    python build_zarr_chunked.py
    python build_zarr_chunked.py --batch-size 200 --workers 16

    # After a re-run from a restart point (daily or survey anchor): replace
    # everything after the restart time with the re-run's profiles.
    python build_zarr_chunked.py --pro-dir D --zarr-out S --update-since 2026-02-10T12:00

    # NWP forecast store: only the profiles after T_stable, rebuilt from scratch.
    python build_zarr_chunked.py --pro-dir D --zarr-out F --since 2026-02-10T12:00
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snowpack_io import build_zarr_cache, update_zarr_since


def main():
    parser = argparse.ArgumentParser(
        description="Build Zarr cache from .pro files")
    parser.add_argument('--pro-dir',   default=None)
    parser.add_argument('--zarr-out',  default=None)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers',    type=int, default=31)
    parser.add_argument('--max-layers', type=int, default=338)
    parser.add_argument('--rebuild', action='store_true',
                        help='Overwrite an existing store instead of resuming '
                             'into it.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--update-since', default=None, metavar='TS',
                      help='Replace everything in the store after TS with the '
                           '.pro blocks after TS (incremental re-run update).')
    mode.add_argument('--since', default=None, metavar='TS',
                      help='Build a fresh store holding only profiles after TS '
                           '(the NWP forecast store). Implies --rebuild.')
    args = parser.parse_args()

    if args.update_since:
        update_zarr_since(Path(args.pro_dir), Path(args.zarr_out),
                          args.update_since, n_cpus=args.workers,
                          max_layers=args.max_layers)
        return

    build_zarr_cache(
        pro_dir    = Path(args.pro_dir),
        zarr_out   = Path(args.zarr_out),
        batch_size = args.batch_size,
        n_cpus     = args.workers,
        max_layers = args.max_layers,
        rebuild    = args.rebuild or bool(args.since),
        since      = args.since,
    )


if __name__ == '__main__':
    main()

