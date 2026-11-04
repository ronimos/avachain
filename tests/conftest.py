"""Suite-wide fixtures."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))


@pytest.fixture(autouse=True)
def _no_wrf_wait(monkeypatch):
    """No test may sleep in nwp_ingest.wait_for_wrf (it waits up to 30 min
    for the CAIC rebuild in operation). Fresh files count as settled; the
    tests of the wait itself set these back explicitly."""
    import nwp_ingest
    monkeypatch.setattr(nwp_ingest, "WRF_WAIT_TIMEOUT_S", 0)
    monkeypatch.setattr(nwp_ingest, "WRF_SETTLE_S", 0)
