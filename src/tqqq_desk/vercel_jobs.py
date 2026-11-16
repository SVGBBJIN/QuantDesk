"""Serverless entry points (Vercel Cron). All configuration comes from environment variables:

  required  ALPACA_KEY, ALPACA_SECRET      Alpaca API key id / secret (paper keys unless going live)
            CRON_SECRET                    Vercel sends `Authorization: Bearer $CRON_SECRET` to cron paths
            QD_GITHUB_TOKEN                fine-grained token, Contents: read & write on this repo only
  optional  QD_GITHUB_REPO                 owner/name holding the state branch (default SVGBBJIN/QuantDesk)
            QD_STATE_BRANCH                state branch (default data)
            QD_OBJECTIVE                   risk_adjusted | max_return | robust (overrides config.yaml)
            ALPACA_LIVE, I_UNDERSTAND_THE_RISK   both "true" for real money (see README, Going live)

Each job: authenticate -> pull the state branch into /tmp -> run exactly the same code as the CLI ->
push changed files back as one commit -> append to runs.json (shown on the status page)."""
from __future__ import annotations

import hmac
import io
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .logutil import RedactSecrets, setup_logging

log = logging.getLogger("tqqq_desk.vercel")
TRUE = ("1", "true", "yes", "on")
TRADE_FLAGS = ("dry_run", "force", "reset_kill_switch", "adopt_broker_positions")
MAX_RUNS = 60


def authorized(headers, env=os.environ) -> tuple[bool, str]:
    secret = env.get("CRON_SECRET", "")
    if len(secret) < 16:
        return False, "CRON_SECRET is not set (or shorter than 16 characters) in the Vercel project env vars"
    got = headers.get("Authorization") or headers.get("authorization") or ""
    if not hmac.compare_digest(got.encode(), f"Bearer {secret}".encode()):
        return False, "unauthorized"
    return True, ""


def parse_flags(path: str) -> dict[str, bool]:
    q = parse_qs(urlparse(path).query)
    return {k: (q.get(k, ["false"])[-1].lower() in TRUE) for k in TRADE_FLAGS}


class _Capture:
    """Collect this job's log lines (redacted) to return in the HTTP response."""

    def __enter__(self):
        setup_logging()
        self.buf = io.StringIO()
        self.h = logging.StreamHandler(self.buf)
        self.h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        self.h.addFilter(RedactSecrets())
        logging.getLogger().addHandler(self.h)
        return self

    def __exit__(self, *a):
        logging.getLogger().removeHandler(self.h)

    def text(self, limit: int = 20000) -> str:
        t = self.buf.getvalue()
        return t[-limit:]


def _store(env):
    from .ghstate import GitHubState
    return GitHubState(env.get("QD_GITHUB_REPO", "SVGBBJIN/QuantDesk"), env.get("QD_GITHUB_TOKEN", ""),
                       env.get("QD_STATE_BRANCH", "data"))


def _record_run(state_dir: Path, entry: dict) -> None:
    p = state_dir / "runs.json"
    try:
        runs = json.loads(p.read_text()) if p.exists() else []
    except ValueError:
        runs = []
    runs.append(entry)
    p.write_text(json.dumps(runs[-MAX_RUNS:], indent=2, default=str))


def _workdir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="qd-", dir=os.environ.get("TMPDIR", "/tmp")))
    os.environ.setdefault("QD_CACHE_DIR", str(d / "cache"))
    return d


def run_job(job: str, flags: dict | None = None, env=os.environ, store=None, runner=None) -> tuple[int, dict]:
    """job: "trade" or "validate". Returns (http_status, json_body)."""
    flags = flags or {}
    started = pd.Timestamp.now(tz="UTC")
    t0 = time.monotonic()
    with _Capture() as cap:
        work = _workdir()
        state_dir = work / "state"
        body: dict = {"job": job, "flags": flags, "started_at": started.isoformat()}
        code, store_ok = 4, False
        try:
            if job == "trade" and not flags.get("dry_run") and env.get("VERCEL_ENV", "production") != "production":
                raise PermissionError(f"refusing to trade from a {env.get('VERCEL_ENV')} deployment "
                                      "(only production may submit orders; use ?dry_run=1)")
            store = store or _store(env)
            store.pull(state_dir)
            store_ok = True
            code, extra = (runner or (_trade if job == "trade" else _validate))(state_dir, flags, env)
            body.update(extra)
        except Exception as e:  # noqa: BLE001
            log.error("%s job failed: %s\n%s", job, e, traceback.format_exc())
            body["error"] = f"{type(e).__name__}: {e}"
            code = 2 if isinstance(e, PermissionError) else 4
        body["exit_code"] = code
        body["seconds"] = round(time.monotonic() - t0, 1)
        if store_ok and not (job == "trade" and flags.get("dry_run")):
            _record_run(state_dir, {"job": job, "started_at": started.isoformat(),
                                    "finished_at": pd.Timestamp.now(tz="UTC").isoformat(),
                                    "status": body.get("status", "error"), "exit_code": code,
                                    "summary": body.get("summary") or body.get("error", ""),
                                    "deployment": env.get("VERCEL_URL", "")})
            try:
                store.push(state_dir, f"bot: {job} run {started:%Y-%m-%dT%H:%MZ} (vercel)")
            except Exception as e:  # noqa: BLE001
                log.error("state push failed: %s", e)
                body["error"] = (body.get("error", "") + f"; state push failed: {e}").strip("; ")
                code = body["exit_code"] = 4
        body["log"] = cap.text()
    shutil.rmtree(work, ignore_errors=True)
    return (200 if code == 0 else 500), body


def _trade(state_dir: Path, flags: dict, env) -> tuple[int, dict]:
    from .broker import AlpacaBroker
    from .config import load_config
    from .data import load_bars
    from .live_guard import resolve_trading_mode
    from .run_daily import run

    cfg = load_config()
    mode = resolve_trading_mode(env, str(env.get("I_UNDERSTAND_THE_RISK", "")).lower() == "true",
                                dry_run=flags.get("dry_run", False))
    res = run(cfg, now=pd.Timestamp.now(tz="UTC"), broker=AlpacaBroker(mode), bars_loader=lambda: load_bars(cfg),
              state_dir=state_dir, artifacts_dir=state_dir / "artifacts", mode=mode,
              dry_run=flags.get("dry_run", False), force=flags.get("force", False),
              reset_kill_switch=flags.get("reset_kill_switch", False),
              adopt_broker_positions=flags.get("adopt_broker_positions", False))
    d = res.decision or {}
    summary = (f"{res.status}: {d.get('strategy', '?')} -> {d.get('state', '?')} "
               f"w={d.get('weight')}" if d else f"{res.status}: {'; '.join(res.reasons)}")
    return res.exit_code, {"status": res.status, "mode": mode, "signal_date": res.signal_date, "orders": res.orders,
                           "reasons": res.reasons, "decision": d, "summary": summary}


def _validate(state_dir: Path, flags: dict, env) -> tuple[int, dict]:
    from .config import load_config
    from .data import load_bars
    from .report import write_artifacts, write_report
    from .research import run_research

    cfg = load_config()
    bars = load_bars(cfg, refresh=True)
    r = run_research(bars, cfg, sensitivity=True)
    write_report(r, cfg, state_dir / "reports", "ALPACA")
    write_artifacts(r, cfg, state_dir / "artifacts", "ALPACA")
    sel = r.selection
    return 0, {"status": "validated", "active_strategy": sel["active"], "objective": sel["objective"],
               "by_objective": sel["by_objective"], "summary": sel["message"]}


def http_handler(job: str):
    """Base class for a Vercel Python `handler` (BaseHTTPRequestHandler) running `job`."""
    from http.server import BaseHTTPRequestHandler

    class JobHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict):
            data = json.dumps(body, indent=2, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            ok, why = authorized(self.headers)
            if not ok:
                return self._send(401 if why == "unauthorized" else 500, {"error": why})
            flags = parse_flags(self.path) if job == "trade" else {}
            status, body = run_job(job, flags)
            self._send(status, body)

        do_POST = do_GET

    return JobHandler
