"""
handoff.py — publish a file set for the next part of the chain.

Parts exchange files only through handoff/<path>/<producer>/<date>/<horizon>/
(release → runout today). Three rules make that safe without locks:

1. Only the producer writes a set; only the consumer reads it.
2. A set is built in a hidden staging directory beside its final place,
   renamed into place, and READY is written last. A consumer that waits for
   READY never sees a partial set.
3. Every set carries manifest.json: what it is, which code and TOML made it,
   and every file with its size and sha256, so the consumer can verify it.

Republishing (a manual rerun) removes the old set's READY first, then swaps
the directories, so a consumer sees either the old set, no set, or the new
one, never a mix.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

MANIFEST = "manifest.json"
READY = "READY"

_REPO = Path(__file__).resolve().parent.parent.parent


def staging_dir(final: Path) -> Path:
    """A fresh hidden directory beside `final`, on the same filesystem so the
    publish is a rename."""
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = final.parent / f".{final.name}.staging-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    return stage


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dirty() -> Optional[bool]:
    """True when tracked files differ from the commit: the manifest's commit
    alone would then not say what code made the set."""
    try:
        out = subprocess.run(["git", "-C", str(_REPO), "status", "--porcelain",
                              "--untracked-files=no"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.strip())


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def publish(stage: Path, final: Path, producer: str, *,
            toml: Optional[Path] = None, **fields) -> Path:
    """Write the manifest into `stage`, move it to `final`, then write READY.

    `fields` go into the manifest as is (snapshot_date, horizon, status, ...).
    """
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    manifest = {
        "producer": producer,
        **fields,
        "created_utc": _now(),
        "commit": _commit(),
        "commit_dirty": _dirty(),
        "toml": str(toml) if toml else None,
        "toml_sha256": _sha256(toml) if toml and Path(toml).exists() else None,
        "files": [{"path": str(p.relative_to(stage)), "bytes": p.stat().st_size,
                   "sha256": _sha256(p)} for p in files],
    }
    _write_atomic(stage / MANIFEST, json.dumps(manifest, indent=2) + "\n")

    old = None
    if final.exists():
        (final / READY).unlink(missing_ok=True)       # unpublish before the swap
        # Subdirectories this build did not produce are not part of the set:
        # published <date>/<horizon>/ sets, or experiment runs someone put
        # under <date>/. They move across instead of being dropped.
        for child in final.iterdir():
            if child.is_dir() and not (stage / child.name).exists():
                os.rename(child, stage / child.name)
        old = final.parent / f".{final.name}.old-{os.getpid()}"
        os.rename(final, old)
    os.rename(stage, final)
    _write_atomic(final / READY, json.dumps(
        {"published_utc": _now(), "manifest": MANIFEST}) + "\n")
    if old is not None:
        shutil.rmtree(old)
    return final


def discard(stage: Path) -> None:
    """Drop a staging directory after a failed build; the published set stays."""
    shutil.rmtree(stage, ignore_errors=True)


def is_ready(final: Path) -> bool:
    return (final / READY).exists()
