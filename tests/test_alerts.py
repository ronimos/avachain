"""alerts.py: someone is told when the operational chain needs attention.

Before 2026-11-04 a FAILED survey, a failed runout or a failed daily run only
showed up in a log or a marker; nobody was told. Pinned here: the message,
recipients from the TOML, never failing the caller, the daily-run check, and
(end to end through the generated scripts) that a FAILED survey and a failed
daily run do send. The runout case is in test_runout_step.
"""
import datetime as dt
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "avachain"))
import alerts  # noqa: E402

sys.path.insert(0, str(REPO / "tests"))
import test_survey_markers as tsm  # noqa: E402


def _subjects(out):
    import email
    from email import policy
    raw = out.read_text().split("\nSubject: ")
    msgs = ["Subject: " + r if i else r for i, r in enumerate(raw) if r.strip()]
    return [email.message_from_string(m, policy=policy.default)["Subject"] for m in msgs]


def _fake_sendmail(tmp_path, rc=0):
    out = tmp_path / "sent.eml"
    sm = tmp_path / "sendmail"
    sm.write_text(f"#!/bin/bash\ncat >> {out}\nexit {rc}\n")
    sm.chmod(0o755)
    return sm, out


def _toml(tmp_path, email='["a@x.org", "b@y.org"]', extra=""):
    t = tmp_path / "slope.toml"
    t.write_text(f'[slope]\nname = "P"\n[paths]\nlogs_dir = "{tmp_path / "logs"}"\n'
                 f'[snowpack]\nseason_start = "2026-11-07"\nseason_end = "2027-05-15"\n'
                 + (f'[alerts]\nemail = {email}\n' if email is not None else "") + extra)
    return t


def test_sends_to_the_toml_recipients_with_the_log_tail(tmp_path, monkeypatch):
    sm, out = _fake_sendmail(tmp_path)
    monkeypatch.setattr(alerts, "SENDMAIL", str(sm))
    log = tmp_path / "run.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)))
    assert alerts.send(_toml(tmp_path), "survey 2026-12-01 FAILED", "why", log) is True
    msg = out.read_text()
    assert _subjects(out) == ["[avachain P] survey 2026-12-01 FAILED"]
    assert "To: a@x.org, b@y.org" in msg
    assert "line 99" in msg and "line 59" not in msg          # the last 40 lines


def test_no_recipients_prints_and_sends_nothing(tmp_path, monkeypatch, capsys):
    sm, out = _fake_sendmail(tmp_path)
    monkeypatch.setattr(alerts, "SENDMAIL", str(sm))
    assert alerts.send(_toml(tmp_path, email=None), "x") is False
    assert "not sent" in capsys.readouterr().out and not out.exists()


def test_a_broken_sendmail_never_fails_the_caller(tmp_path, monkeypatch):
    sm, _ = _fake_sendmail(tmp_path, rc=75)
    monkeypatch.setattr(alerts, "SENDMAIL", str(sm))
    assert alerts.send(_toml(tmp_path), "x") is False
    monkeypatch.setattr(alerts, "SENDMAIL", str(tmp_path / "missing"))
    assert alerts.send(_toml(tmp_path), "x") is False
    assert alerts.main(["--toml", str(_toml(tmp_path)), "send", "--subject", "x"]) == 0


class TestCheckDaily:
    def _log(self, tmp_path, hours_old):
        d = tmp_path / "logs"
        d.mkdir(exist_ok=True)
        p = d / "daily_20261201_130000.log"
        p.write_text("x")
        t = dt.datetime(2026, 12, 2, 17, 0).timestamp() - hours_old * 3600
        os.utime(p, (t, t))

    NOW = dt.datetime(2026, 12, 2, 17, 0)

    def test_fresh_daily_log(self, tmp_path):
        self._log(tmp_path, 4)
        assert alerts.check_daily(_toml(tmp_path), now=self.NOW) is None

    def test_no_daily_run_for_a_day(self, tmp_path):
        self._log(tmp_path, 28)
        assert "28 h old" in alerts.check_daily(_toml(tmp_path), now=self.NOW)

    def test_no_daily_log_at_all_in_the_season(self, tmp_path):
        assert "no daily_*.log" in alerts.check_daily(_toml(tmp_path), now=self.NOW)

    def test_quiet_outside_the_season(self, tmp_path):
        assert alerts.check_daily(_toml(tmp_path), now=dt.datetime(2026, 11, 1)) is None
        assert alerts.check_daily(_toml(tmp_path), now=dt.datetime(2027, 6, 1)) is None


def test_a_failed_survey_sends_an_alert(tmp_path):
    sm, out = _fake_sendmail(tmp_path)
    tif = tsm._survey(tmp_path, tsm.D1, "READY")
    h = tsm.TestOperationalScript()
    script = h._script(tmp_path, 'CURRENT_STEP="Step 3: SNOWPACK"\nfalse')
    toml = tmp_path / "slope_config.toml"
    toml.write_text(toml.read_text() + '\n[alerts]\nemail = ["ops@x.org"]\n')
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                       env=dict(os.environ, AVACHAIN_SENDMAIL=str(sm)))
    assert r.returncode == 1
    assert _subjects(out) == [f"[avachain Professor_PTC] survey {tsm.D1} FAILED (exit 1) at Step 3: SNOWPACK"]
    assert "To: ops@x.org" in out.read_text()


def test_a_failed_daily_run_sends_an_alert(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("install_mod_alerts", REPO / "install.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with open(REPO / "slopes" / "little_prof" / "slope_config.toml", "rb") as fh:
        sh = m._render_run_daily(tomllib.load(fh))
    head = sh[:sh.index("trap 'exit 129' HUP")] + "trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM\n"
    assert "trap _daily_exit EXIT" in head
    head = head.replace('PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                        f'PROJECT_DIR="{REPO}"')
    toml = tsm._toml(tmp_path)
    toml.write_text(toml.read_text() + '\n[alerts]\nemail = ["ops@x.org"]\n')
    head = "\n".join(f"TOML={toml}" if l.startswith("TOML=") else l for l in head.splitlines())
    script = tmp_path / "daily.sh"
    for ok in (True, False):
        script.write_text(head + '\necho ">>> Step 4: SNOWPACK daily cycle"\n'
                          + ("true\n" if ok else "false\n"))
        sm, out = _fake_sendmail(tmp_path)
        r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env=dict(os.environ, AVACHAIN_SENDMAIL=str(sm)))
        if ok:
            assert r.returncode == 0 and not out.exists(), r.stderr
        else:
            assert r.returncode == 1
            assert _subjects(out) == ["[avachain Professor_PTC] daily run FAILED (exit 1) at Step 4: SNOWPACK daily cycle"]


def test_tests_never_reach_real_mail():
    assert alerts.SENDMAIL.startswith("/nonexistent/")
    assert str(alerts.DEFAULT_RECIPIENTS_FILE).startswith("/nonexistent/")


def test_recipients_file_outside_git_is_added(tmp_path):
    f = tmp_path / "alerts.toml"
    f.write_text('email = ["ron@x.org", "a@x.org"]\n')
    t = _toml(tmp_path, extra=f'recipients_file = "{f}"\n')
    assert alerts.settings(t)["to"] == ["a@x.org", "b@y.org", "ron@x.org"]


class TestCheckReport:
    NOW = dt.datetime(2026, 12, 2, 3, 15)

    def _t(self, tmp_path):
        return _toml(tmp_path, extra=f'[runout]\npath_profile = "P"\n'
                                     f'publish_root = "{tmp_path / "www"}"\n')

    def test_missing_report_at_the_deadline(self, tmp_path):
        why = alerts.check_report(self._t(tmp_path), now=self.NOW)
        assert "no report for 20261202 at 03:15" in why

    def test_published_report(self, tmp_path):
        t = self._t(tmp_path)
        d = tmp_path / "www" / "forecasts" / "P" / "20261202"
        d.mkdir(parents=True)
        (d / "report.html").write_text("x")
        assert alerts.check_report(t, now=self.NOW) is None

    def test_quiet_outside_the_season_or_without_runout(self, tmp_path):
        assert alerts.check_report(self._t(tmp_path), now=dt.datetime(2026, 11, 1, 3, 15)) is None
        assert alerts.check_report(_toml(tmp_path), now=self.NOW) is None
