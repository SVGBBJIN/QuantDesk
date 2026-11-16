"""Serverless (Vercel) path: auth, flags, GitHub-API state sync against an in-memory fake GitHub,
the job runner end to end with FakeBroker, the HTTP handler, and env overrides."""
import base64
import hashlib
import importlib.util
import itertools
import json
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pandas as pd
import pytest

from tqqq_desk import mktcal
from tqqq_desk.broker import FakeBroker
from tqqq_desk.config import apply_env_overrides, load_config
from tqqq_desk.data import synthetic_bars
from tqqq_desk.ghstate import GitHubState, git_blob_sha
from tqqq_desk.run_daily import run
from tqqq_desk.vercel_jobs import authorized, parse_flags, run_job

ROOT = Path(__file__).resolve().parents[1]
SECRET = "s" * 32


# ------------------------------------------------------------------ fake GitHub git-data API
class _Resp:
    def __init__(self, status, body):
        self.status_code, self._b = status, body
        self.text = json.dumps(body)

    def json(self):
        return self._b


class FakeGitHub:
    """Minimal git object store behind the REST endpoints GitHubState uses."""

    def __init__(self):
        self.blobs, self.trees, self.commits, self.refs = {}, {}, {}, {}
        self.headers = {}
        self.ids = itertools.count()
        self.reject_next_ref_update = False

    def _id(self, prefix):
        return hashlib.sha1(f"{prefix}{next(self.ids)}".encode()).hexdigest()

    def files(self, branch="data"):
        sha = self.refs.get(branch)
        if sha is None:
            return {}
        t = self.trees[self.commits[sha]["tree"]]
        return {p: base64.b64decode(self.blobs[s]).decode() for p, s in t.items()}

    def seed(self, files: dict[str, str], branch="data"):
        tree = {}
        for p, c in files.items():
            sha = git_blob_sha(c.encode())
            self.blobs[sha] = base64.b64encode(c.encode()).decode()
            tree[p] = sha
        tid = self._id("t")
        self.trees[tid] = tree
        cid = self._id("c")
        self.commits[cid] = {"tree": tid, "parents": []}
        self.refs[branch] = cid

    def request(self, method, url, timeout=None, json=None, params=None):
        path = url.split("/repos/o/r", 1)[1]
        if method == "GET" and path.startswith("/git/ref/heads/"):
            b = path.rsplit("/", 1)[1]
            return _Resp(200, {"object": {"sha": self.refs[b]}}) if b in self.refs else _Resp(404, {})
        if method == "GET" and path.startswith("/git/commits/"):
            c = self.commits[path.rsplit("/", 1)[1]]
            return _Resp(200, {"tree": {"sha": c["tree"]}})
        if method == "GET" and path.startswith("/git/trees/"):
            t = self.trees[path.rsplit("/", 1)[1]]
            return _Resp(200, {"tree": [{"path": p, "sha": s, "type": "blob"} for p, s in t.items()]})
        if method == "GET" and path.startswith("/git/blobs/"):
            return _Resp(200, {"content": self.blobs[path.rsplit("/", 1)[1]]})
        if method == "POST" and path == "/git/blobs":
            sha = git_blob_sha(base64.b64decode(json["content"]))
            self.blobs[sha] = json["content"]
            return _Resp(201, {"sha": sha})
        if method == "POST" and path == "/git/trees":
            tree = dict(self.trees[json["base_tree"]]) if "base_tree" in json else {}
            tree.update({e["path"]: e["sha"] for e in json["tree"]})
            tid = self._id("t")
            self.trees[tid] = tree
            return _Resp(201, {"sha": tid})
        if method == "POST" and path == "/git/commits":
            cid = self._id("c")
            self.commits[cid] = {"tree": json["tree"], "parents": json["parents"], "message": json["message"]}
            return _Resp(201, {"sha": cid})
        if method == "PATCH" and path.startswith("/git/refs/heads/"):
            b = path.rsplit("/", 1)[1]
            new = json["sha"]
            if self.reject_next_ref_update or self.refs[b] not in self.commits[new]["parents"]:
                self.reject_next_ref_update = False
                return _Resp(422, {"message": "Update is not a fast forward"})
            self.refs[b] = new
            return _Resp(200, {})
        if method == "POST" and path == "/git/refs":
            self.refs[json["ref"].rsplit("/", 1)[1]] = json["sha"]
            return _Resp(201, {})
        raise AssertionError(f"unexpected {method} {path}")


def _store(gh):
    return GitHubState("o/r", "tok", "data", session=gh)


def test_state_roundtrip_and_conflict_rebase(tmp_path):
    gh = FakeGitHub()
    gh.seed({"README.md": "hi\n", "state.json": '{"a": 1}', "trades.csv": "x\n"})
    st = _store(gh)
    st.pull(tmp_path)
    assert (tmp_path / "state.json").read_text() == '{"a": 1}'
    (tmp_path / "state.json").write_text('{"a": 2}')
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "model_status.json").write_text("{}")
    assert set(st.changed_files(tmp_path)) == {"state.json", "artifacts/model_status.json"}
    # someone else commits in between -> our push must rebase, keeping their file and ours
    other = _store(gh)
    other.pull(tmp_path / "other")
    (tmp_path / "other" / "trades.csv").write_text("x\ny\n")
    other.push(tmp_path / "other", "other run")
    st.push(tmp_path, "our run")
    f = gh.files()
    assert f["state.json"] == '{"a": 2}' and f["trades.csv"] == "x\ny\n" and f["README.md"] == "hi\n"
    assert f["artifacts/model_status.json"] == "{}"


def test_state_branch_created_when_missing(tmp_path):
    gh = FakeGitHub()
    st = _store(gh)
    assert st.pull(tmp_path) is None and (tmp_path / "README.md").exists()
    (tmp_path / "state.json").write_text("{}")
    st.push(tmp_path, "first")
    assert set(gh.files()) == {"README.md", "state.json"}


def test_no_push_without_changes(tmp_path):
    gh = FakeGitHub()
    gh.seed({"state.json": "{}"})
    st = _store(gh)
    head = st.pull(tmp_path)
    assert st.push(tmp_path, "noop") == head and gh.refs["data"] == head


# ------------------------------------------------------------------ auth / flags / env
def test_authorization():
    ok, why = authorized({}, env={})
    assert not ok and "CRON_SECRET" in why
    assert authorized({"Authorization": "Bearer nope"}, env={"CRON_SECRET": SECRET}) == (False, "unauthorized")
    assert authorized({"Authorization": f"Bearer {SECRET}"}, env={"CRON_SECRET": SECRET})[0]


def test_flags_parse():
    f = parse_flags("/api/cron/trade?dry_run=1&force=true&adopt_broker_positions=no")
    assert f == {"dry_run": True, "force": True, "reset_kill_switch": False, "adopt_broker_positions": False}
    assert not any(parse_flags("/api/cron/trade").values())


def test_objective_env_override():
    cfg = load_config()
    apply_env_overrides(cfg, {"QD_OBJECTIVE": "robust"})
    assert cfg["selection"]["objective"] == "robust"


# ------------------------------------------------------------------ job runner end to end
def _fake_trade_runner(now):
    n = len(mktcal.schedule("2016-01-04", str(mktcal.last_completed_session(now))))
    bars = synthetic_bars(n, seed=11, start="2016-01-04")
    broker = FakeBroker()

    def runner(state_dir, flags, env):
        res = run(load_config(), now=now, broker=broker, bars_loader=lambda: bars, state_dir=state_dir,
                  artifacts_dir=state_dir / "artifacts", mode="paper", dry_run=flags.get("dry_run", False))
        return res.exit_code, {"status": res.status, "summary": res.status}

    return runner, broker


def test_trade_job_pulls_runs_and_pushes(tmp_path):
    gh = FakeGitHub()
    gh.seed({"README.md": "x"})
    runner, broker = _fake_trade_runner(pd.Timestamp("2024-07-05 19:30", tz="America/New_York"))
    status, body = run_job("trade", {}, env={"VERCEL_ENV": "production"}, store=_store(gh), runner=runner)
    assert status == 200 and body["exit_code"] == 0, body
    f = gh.files()
    assert {"state.json", "signals.csv", "equity.csv", "runs.json"} <= set(f)
    runs = json.loads(f["runs.json"])
    assert runs[-1]["job"] == "trade" and runs[-1]["exit_code"] == 0
    assert json.loads(f["state.json"])["last_signal_date"] == "2024-07-05"


def test_dry_run_does_not_push_and_preview_cannot_trade(tmp_path):
    gh = FakeGitHub()
    gh.seed({"README.md": "x"})
    head = gh.refs["data"]
    runner, broker = _fake_trade_runner(pd.Timestamp("2024-07-05 19:30", tz="America/New_York"))
    status, body = run_job("trade", {"dry_run": True}, env={"VERCEL_ENV": "preview"}, store=_store(gh),
                           runner=runner)
    assert status == 200 and gh.refs["data"] == head and not broker.orders
    status, body = run_job("trade", {}, env={"VERCEL_ENV": "preview"}, store=_store(gh), runner=runner)
    assert status == 500 and "refusing to trade" in body["error"] and gh.refs["data"] == head


def test_failed_job_still_records_run(tmp_path):
    gh = FakeGitHub()
    gh.seed({"README.md": "x"})

    def boom(state_dir, flags, env):
        raise RuntimeError("alpaca down")

    status, body = run_job("validate", {}, env={}, store=_store(gh), runner=boom)
    assert status == 500 and "alpaca down" in body["error"]
    assert json.loads(gh.files()["runs.json"])[-1]["status"] == "error"


# ------------------------------------------------------------------ HTTP entry points
@pytest.mark.parametrize("job", ["trade", "validate"])
def test_vercel_entrypoint_rejects_unauthenticated(job, monkeypatch):
    spec = importlib.util.spec_from_file_location(f"vc_{job}", ROOT / "api" / "cron" / f"{job}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("CRON_SECRET", SECRET)
    srv = HTTPServer(("127.0.0.1", 0), mod.handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/api/cron/{job}",
                                 headers={"Authorization": "Bearer wrong"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 401
    srv.server_close()


def test_vercel_json_crons_and_functions():
    v = json.loads((ROOT / "vercel.json").read_text())
    paths = {c["path"] for c in v["crons"]}
    assert paths == {"/api/cron/trade", "/api/cron/validate"}
    for f in ("api/cron/trade.py", "api/cron/validate.py"):
        assert (ROOT / f).exists() and v["functions"][f]["maxDuration"] <= 300
    # every trade cron must land inside the OPG window on weekday evenings ET (both DST regimes)
    for c in v["crons"]:
        if c["path"] == "/api/cron/trade":
            minute, hour = int(c["schedule"].split()[0]), int(c["schedule"].split()[1])
            for date in ("2024-07-09", "2024-01-09"):          # summer (EDT) and winter (EST)
                for slack in (0, 59 - minute):                   # Hobby crons may fire anywhere in the hour
                    t = pd.Timestamp(f"{date} {hour:02d}:{minute + slack:02d}", tz="UTC").tz_convert("America/New_York")
                    assert t.hour >= 19 or t.hour < 9, (c, t)
