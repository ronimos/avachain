"""Suite-wide fixtures."""
import os
import sys
from pathlib import Path

# No test may send real mail, including the scripts tests run as
# subprocesses (they inherit the environment): sendmail points nowhere and
# the operator's recipients file is never read. Tests of the mail itself set
# their own fake sendmail and recipients.
os.environ["AVACHAIN_SENDMAIL"] = "/nonexistent/sendmail-disabled-in-tests"
os.environ["AVACHAIN_ALERTS_FILE"] = "/nonexistent/alerts-disabled-in-tests.toml"

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
