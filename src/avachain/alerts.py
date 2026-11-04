"""alerts.py — tell a person when the operational chain needs attention.

    python alerts.py --toml T send --subject "..." [--log F] [--body "..."]
    python alerts.py --toml T check-daily [--max-age-h 26]

Before 2026-11-04 a FAILED survey, a failed runout or a failed daily run only
showed up in a log or a marker file; nobody was told. The run scripts now call
`send` on each of those, and cron runs `check-daily` once a day, which catches
the case no run can report itself: no daily run at all (cron or the host down).

Recipients are the slope TOML's [alerts] email list, sent through the local
sendmail. With no recipients configured an alert is printed and nothing is
sent. An alert never fails its caller: every path exits 0, because the run that
asks for it has already failed or finished and must keep its own exit status.
"""
import argparse
import datetime as dt
import os
import socket
import subprocess
import sys
import tomllib
from email.message import EmailMessage
from pathlib import Path

SENDMAIL = os.environ.get("AVACHAIN_SENDMAIL", "/usr/sbin/sendmail")
LOG_TAIL_LINES = 40


def settings(toml_path: Path) -> dict:
    data = tomllib.loads(Path(toml_path).read_text())
    al = data.get("alerts", {})
    to = al.get("email") or []
    if isinstance(to, str):
        to = [to]
    return {"to": [a for a in to if a],
            "from": al.get("from") or f"avachain@{socket.gethostname()}",
            "slope": data.get("slope", {}).get("name", "?"),
            "logs_dir": data.get("paths", {}).get("logs_dir"),
            "season_start": data.get("snowpack", {}).get("season_start"),
            "season_end": data.get("snowpack", {}).get("season_end")}


def _tail(path, n=LOG_TAIL_LINES) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError as e:
        return f"(log not readable: {e})"
    return "\n".join(lines[-n:])


def send(toml_path: Path, subject: str, body: str = "", log=None) -> bool:
    """Send one alert. True when handed to sendmail; never raises."""
    try:
        s = settings(toml_path)
    except Exception as e:                       # a broken TOML must not hide the alert
        s = {"to": [], "from": "avachain", "slope": "?"}
        body = f"(TOML not readable: {e})\n\n{body}"
    subject = f"[avachain {s['slope']}] {subject}"
    text = body.rstrip() + "\n"
    if log:
        text += f"\nLog: {log}\nLast {LOG_TAIL_LINES} lines:\n\n{_tail(log)}\n"
    text += f"\nHost {socket.gethostname()}, {dt.datetime.now().isoformat(timespec='seconds')}\n"
    if not s["to"]:
        print(f"ALERT (no [alerts] email configured, not sent): {subject}", flush=True)
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, s["from"], ", ".join(s["to"])
    msg.set_content(text)
    try:
        r = subprocess.run([SENDMAIL, "-t", "-oi"], input=msg.as_bytes(), timeout=60)
        ok = r.returncode == 0
    except (OSError, subprocess.SubprocessError) as e:
        print(f"ALERT not sent ({e}): {subject}", flush=True)
        return False
    print(f"ALERT {'sent to ' + ', '.join(s['to']) if ok else 'sendmail failed'}: {subject}",
          flush=True)
    return ok


def check_daily(toml_path: Path, max_age_h: float = 26.0, now=None) -> str | None:
    """Why the daily run looks missing, or None. Outside the season: None."""
    s = settings(toml_path)
    now = now or dt.datetime.now()
    today = now.date()
    if s["season_start"] and today < dt.date.fromisoformat(str(s["season_start"])):
        return None
    if s["season_end"] and today > dt.date.fromisoformat(str(s["season_end"])):
        return None
    logs = sorted(Path(s["logs_dir"]).glob("daily_*.log")) if s["logs_dir"] else []
    if not logs:
        return f"no daily_*.log in {s['logs_dir']} during the season"
    newest = max(logs, key=lambda p: p.stat().st_mtime)
    age_h = (now.timestamp() - newest.stat().st_mtime) / 3600
    if age_h > max_age_h:
        return f"the newest daily log {newest.name} is {age_h:.0f} h old (limit {max_age_h:g} h)"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--toml", required=True, type=Path)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("--subject", required=True)
    s.add_argument("--body", default="")
    s.add_argument("--log", default=None)
    c = sub.add_parser("check-daily")
    c.add_argument("--max-age-h", type=float, default=26.0)
    args = ap.parse_args(argv)
    if args.cmd == "send":
        send(args.toml, args.subject, args.body, args.log)
        return 0
    try:
        why = check_daily(args.toml, args.max_age_h)
    except Exception as e:
        why = f"check-daily could not run: {e}"
    if why:
        send(args.toml, "no daily run", why + "\n\nCheck cron (/etc/cron.d), the "
             "flock and the host. The daily run carries the snowpack forward between surveys.")
    else:
        print("daily run OK (or outside the season)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
