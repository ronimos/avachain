"""Per-slope release-model calibration: the slope TOML's [calibration].

CLAUDE.md "Calibration" (Ron, 2026-11-08) sets the rules: one smooth knob,
two at most; values in the slope TOML with provenance; the physics code and
its module defaults stay identical to ~/avalanche-release-areas, and a slope
without [calibration] runs exactly as before.

    [calibration]
    version     = "2026-11-xx.1"        # stamped on every set, report and record
    c_fit       = 0.045                 # Eq. 20 prefactor (arrest_indices.C_FIT)
    size_scale  = 1.0                   # multiplies every [scenarios] size factor
    # provenance (free-form, not read by the model):
    events = ["2026-01-18"]; no_event_dates = [...]; objective = "..."; fitted = "..."; commit = "..."

Knobs, and why only these two:
  c_fit       Eq. 20's C. Read at call time by arrest_indices.arrest_length, so
              it moves both the trigger's A_ca_brittle (analyze) and the BFS
              Meloche arrest (scenarios). Analyze must be re-run after a change.
  size_scale  The scenario size factors scale the BFS distance caps and the
              rectangle fallback; their weights are centred on the median log
              factor, so a uniform scale leaves the weights unchanged.
The Λ factors are not knobs: with USE_MELOCHE_ARREST they never run.
"""
from __future__ import annotations

KNOBS = ("c_fit", "size_scale")
PROVENANCE = ("version", "events", "no_event_dates", "objective", "fitted",
              "commit", "note", "sweep")


def _check(cal: dict) -> None:
    unknown = set(cal) - set(KNOBS) - set(PROVENANCE)
    if unknown:
        raise ValueError(f"[calibration]: unknown key(s) {sorted(unknown)}; "
                         f"knobs are {KNOBS}")
    if "c_fit" in cal and not float(cal["c_fit"]) > 0:
        raise ValueError("[calibration] c_fit must be > 0")
    if "size_scale" in cal and not float(cal["size_scale"]) > 0:
        raise ValueError("[calibration] size_scale must be > 0")


def apply(cfg, quiet: bool = False) -> dict:
    """Set the module values a slope's [calibration] overrides. Returns what
    was applied ({} when uncalibrated). Call before analyze or scenarios."""
    cal = dict(getattr(cfg, "calibration", None) or {})
    _check(cal)
    import arrest_indices as _ai
    applied = {}
    if "c_fit" in cal:
        _ai.C_FIT = float(cal["c_fit"])
        applied["c_fit"] = _ai.C_FIT
    if "size_scale" in cal:
        applied["size_scale"] = float(cal["size_scale"])
    if applied and not quiet:
        print(f"  calibration {cal.get('version', '(no version)')}: "
              + ", ".join(f"{k}={v:g}" for k, v in applied.items()), flush=True)
    return applied


def size_scale(cfg) -> float:
    return float((getattr(cfg, "calibration", None) or {}).get("size_scale", 1.0))


def version(cfg):
    v = (getattr(cfg, "calibration", None) or {}).get("version")
    return str(v) if v is not None else None
