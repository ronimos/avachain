"""Per-slope release-model calibration: the slope TOML's [calibration].

CLAUDE.md "Calibration" (Ron, 2026-11-08) sets the rules: one smooth knob,
two at most; values in the slope TOML with provenance; the physics code and
its module defaults stay identical to ~/avalanche-release-areas, and a slope
without [calibration] runs exactly as before.

    [calibration]
    version     = "2026-11-xx.1"        # stamped on every set, report and record
    c_fit       = 0.045                 # Eq. 20 prefactor (model_params.C_FIT)
    size_scale  = 1.0                   # multiplies every [scenarios] size factor
    facet_a     = 0.5                   # facet strength knockdown (FACET_STRENGTH_FACTOR)
    distance_caps = true                # BFS distance caps (USE_DISTANCE_CAPS)
    tau_g_floor = 350.0                 # Pa, BFS driving-stress floor (TAU_G_ABS_FLOOR)
    tau_ratio_floor = 0.0               # BFS tau_g/tau_p floor, 0 = off (TAU_RATIO_FLOOR)
    # provenance (free-form, not read by the model):
    events = ["2026-01-18"]; no_event_dates = [...]; objective = "..."; fitted = "..."; commit = "..."

Knobs, and why only these two:
  c_fit       Eq. 20's C. Read at call time by arrest_indices.arrest_length, so
              it moves both the trigger's A_ca_brittle (analyze) and the BFS
              Meloche arrest (scenarios). Analyze must be re-run after a change.
  size_scale  The scenario size factors scale the BFS distance caps and the
              rectangle fallback; their weights are centred on the median log
              factor, so a uniform scale leaves the weights unchanged.
  facet_a     model_params.FACET_STRENGTH_FACTOR, the J&J 1990 sigma_t
              knockdown for faceted layers (module default 0.5). It moves the
              sigma_t_mean / _wl / _ligament columns and the tensile gate, so
              analyze must be re-run (--force) after a change.
  distance_caps  model_params.USE_DISTANCE_CAPS. A structural switch, not
              a fitted value: false opens the BFS up/down/cross caps. With the
              caps off, size_scale and the size factors scale nothing in the
              BFS.
  tau_g_floor model_params.TAU_G_ABS_FLOOR, the per-cluster tau_g floor
              of the BFS (module default 350 Pa). Scenarios only; analyze is
              unaffected.
  tau_ratio_floor model_params.TAU_RATIO_FLOOR, the per-cluster tau_g /
              tau_p floor of the BFS (module default 0 = off). Scenarios only.
The Λ factors are not knobs: with USE_MELOCHE_ARREST they never run.
"""
from __future__ import annotations
import model_params as mp

KNOBS = ("c_fit", "size_scale", "facet_a", "distance_caps", "tau_g_floor", "tau_ratio_floor")
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
    if "facet_a" in cal and not 0 < float(cal["facet_a"]) <= 1:
        raise ValueError("[calibration] facet_a must be in (0, 1]")
    if "tau_g_floor" in cal and not float(cal["tau_g_floor"]) >= 0:
        raise ValueError("[calibration] tau_g_floor must be >= 0")
    if "tau_ratio_floor" in cal and not float(cal["tau_ratio_floor"]) >= 0:
        raise ValueError("[calibration] tau_ratio_floor must be >= 0")
    if "distance_caps" in cal and not isinstance(cal["distance_caps"], bool):
        raise ValueError("[calibration] distance_caps must be true or false")


def apply(cfg, quiet: bool = False) -> dict:
    """Set the model_params values a slope's [calibration] overrides. Returns what
    was applied ({} when uncalibrated). Call before analyze or scenarios."""
    cal = dict(getattr(cfg, "calibration", None) or {})
    _check(cal)
    applied = {}
    if "c_fit" in cal:
        mp.C_FIT = float(cal["c_fit"])
        applied["c_fit"] = mp.C_FIT
    if "size_scale" in cal:
        applied["size_scale"] = float(cal["size_scale"])
    if "facet_a" in cal:
        mp.FACET_STRENGTH_FACTOR = float(cal["facet_a"])
        applied["facet_a"] = mp.FACET_STRENGTH_FACTOR
    if "distance_caps" in cal:
        mp.USE_DISTANCE_CAPS = bool(cal["distance_caps"])
        applied["distance_caps"] = mp.USE_DISTANCE_CAPS
    if "tau_g_floor" in cal:
        mp.TAU_G_ABS_FLOOR = float(cal["tau_g_floor"])
        applied["tau_g_floor"] = mp.TAU_G_ABS_FLOOR
    if "tau_ratio_floor" in cal:
        mp.TAU_RATIO_FLOOR = float(cal["tau_ratio_floor"])
        applied["tau_ratio_floor"] = mp.TAU_RATIO_FLOOR
    if applied and not quiet:
        print(f"  calibration {cal.get('version', '(no version)')}: "
              + ", ".join(f"{k}={v:g}" if not isinstance(v, bool) else f"{k}={v}"
                          for k, v in applied.items()), flush=True)
    return applied


def size_scale(cfg) -> float:
    return float((getattr(cfg, "calibration", None) or {}).get("size_scale", 1.0))


def version(cfg):
    v = (getattr(cfg, "calibration", None) or {}).get("version")
    return str(v) if v is not None else None
