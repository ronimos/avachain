"""
workers.py

Description:
    The ceiling on concurrent com1DFA simulations in one run.

    execution.max_workers is what a run asks for. execution.max_workers_cap
    is the most it gets. A profile without the cap key still gets
    DEFAULT_MAX_WORKERS_CAP, so `--set execution.max_workers=12` on its own
    cannot raise concurrency. Going above the default takes a second,
    deliberate `--set execution.max_workers_cap=N`.

    Why (2026-11-02): two runs side by side at max_workers=12 put 24
    simulations on the shared 256 GB host. The large releases grew to
    11-14 GB each, the kernel's OOM killer took sshd, user sessions and
    another user's job before reaching ours, and the pool parents then
    waited forever on workers that no longer existed. A healthy simulation
    here holds ~0.4 GB, so 4 leaves room even for a runaway.

    The cap is per run. It does not stop two runs from starting at once;
    don't do that.
"""

import logging
import os

DEFAULT_MAX_WORKERS_CAP = 4


def capped_workers(execution, cpu_count=None):
    # Pool size for one run, after the cap.
    # @param execution: the config's execution section (may be None)
    # @param cpu_count: os.cpu_count() stand-in, for tests
    # @returns: int >= 1
    execution = execution or {}
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)

    requested = execution.get('max_workers')
    if requested is None:                      # 'auto' = one per CPU
        requested = cpus
    requested = int(requested)

    cap = execution.get('max_workers_cap', DEFAULT_MAX_WORKERS_CAP)
    cap = DEFAULT_MAX_WORKERS_CAP if cap is None else int(cap)
    if cap < 1:
        raise ValueError(f"execution.max_workers_cap must be >= 1, got {cap}")

    if requested > cap:
        logging.warning(
            f"execution.max_workers={requested} is above "
            f"execution.max_workers_cap={cap}; running {cap} at once. Each "
            f"com1DFA simulation can grow to >10 GB, and the host is shared.")
        return cap
    return max(1, requested)
