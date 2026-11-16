"""Bot state on the `data` branch, read and written through the GitHub REST API (no git binary).

Serverless functions (Vercel) have no persistent disk, so each run:
  1. pull():  downloads every file of the state branch into a temp directory,
  2. runs the job against that directory exactly like a local run,
  3. push():  commits the files that changed as ONE commit (blobs -> tree -> commit -> ref update).
If the branch moved in between (another run committed), the commit is rebuilt on the new head with our
changed files on top (last writer wins per file, like `git rebase` in commit_state.sh) and retried.
Files are never deleted by push()."""
from __future__ import annotations

import base64
import hashlib
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)
API = "https://api.github.com"
README = "# tqqq-desk bot state\nWritten by the bot (Vercel cron / GitHub Actions). Do not edit by hand unless reconciling.\n"


class GitHubStateError(RuntimeError):
    pass


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class GitHubState:
    def __init__(self, repo: str, token: str, branch: str = "data", session: requests.Session | None = None,
                 timeout: float = 30.0):
        if not token:
            raise GitHubStateError("QD_GITHUB_TOKEN is not set (needs contents: read & write on the repo)")
        self.repo, self.branch, self.timeout = repo, branch, timeout
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "tqqq-desk-bot"})
        self.base_sha: str | None = None
        self.base_tree: dict[str, str] = {}

    # ------------------------------------------------------------------ http
    def _req(self, method: str, path: str, ok=(200, 201), **kw):
        for attempt in range(4):
            try:
                r = self.s.request(method, f"{API}/repos/{self.repo}{path}", timeout=self.timeout, **kw)
            except requests.RequestException as e:
                err = e
            else:
                if r.status_code in ok or r.status_code in (404, 409, 422):
                    return r
                err = GitHubStateError(f"{method} {path}: HTTP {r.status_code} {r.text[:200]}")
                if r.status_code in (401, 403) and "rate limit" not in r.text.lower():
                    raise err
            time.sleep(2 ** attempt)
        raise GitHubStateError(f"{method} {path} failed after retries: {err}")

    def _head(self) -> str | None:
        r = self._req("GET", f"/git/ref/heads/{self.branch}")
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise GitHubStateError(f"cannot read branch {self.branch}: HTTP {r.status_code}")
        return r.json()["object"]["sha"]

    def _tree(self, commit_sha: str) -> tuple[str, dict[str, str]]:
        c = self._req("GET", f"/git/commits/{commit_sha}").json()
        t = self._req("GET", f"/git/trees/{c['tree']['sha']}", params={"recursive": "1"}).json()
        if t.get("truncated"):
            raise GitHubStateError("state tree too large (truncated listing)")
        return c["tree"]["sha"], {e["path"]: e["sha"] for e in t["tree"] if e["type"] == "blob"}

    # ------------------------------------------------------------------ pull / push
    def pull(self, dest: Path) -> str | None:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self.base_sha = self._head()
        if self.base_sha is None:
            log.warning("state branch %s does not exist yet; starting from empty state", self.branch)
            self.base_tree = {}
            (dest / "README.md").write_text(README)
            return None
        _, self.base_tree = self._tree(self.base_sha)
        for path, sha in self.base_tree.items():
            blob = self._req("GET", f"/git/blobs/{sha}").json()
            f = dest / path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(base64.b64decode(blob["content"]))
        log.info("pulled %d state files from %s@%s", len(self.base_tree), self.branch, self.base_sha[:7])
        return self.base_sha

    def changed_files(self, src: Path) -> dict[str, bytes]:
        out = {}
        for f in sorted(Path(src).rglob("*")):
            if not f.is_file() or f.name.endswith(".tmp") or ".git" in f.parts:
                continue
            rel = f.relative_to(src).as_posix()
            data = f.read_bytes()
            if self.base_tree.get(rel) != git_blob_sha(data):
                out[rel] = data
        return out

    def push(self, src: Path, message: str, max_attempts: int = 4) -> str | None:
        files = self.changed_files(src)
        if not files:
            log.info("no state changes to push")
            return self.base_sha
        blobs = {}
        for path, data in files.items():
            r = self._req("POST", "/git/blobs", json={"content": base64.b64encode(data).decode(), "encoding": "base64"})
            if r.status_code not in (200, 201):
                raise GitHubStateError(f"blob upload failed for {path}: HTTP {r.status_code}")
            blobs[path] = r.json()["sha"]
        parent = self.base_sha
        for attempt in range(max_attempts):
            entries = [{"path": p, "mode": "100644", "type": "blob", "sha": s} for p, s in blobs.items()]
            body = {"tree": entries}
            if parent:
                body["base_tree"] = self._req("GET", f"/git/commits/{parent}").json()["tree"]["sha"]
            tree = self._req("POST", "/git/trees", json=body).json()["sha"]
            commit = self._req("POST", "/git/commits", json={
                "message": message, "tree": tree, "parents": [parent] if parent else []}).json()["sha"]
            if parent:
                r = self._req("PATCH", f"/git/refs/heads/{self.branch}", json={"sha": commit, "force": False})
            else:
                r = self._req("POST", "/git/refs", json={"ref": f"refs/heads/{self.branch}", "sha": commit})
            if r.status_code in (200, 201):
                log.info("pushed %d changed state files to %s@%s", len(files), self.branch, commit[:7])
                self.base_sha = commit
                return commit
            log.warning("state push rejected (HTTP %s, attempt %d); rebasing onto the new head",
                        r.status_code, attempt + 1)
            parent = self._head()
            time.sleep(1 + attempt)
        raise GitHubStateError("could not push state after retries")
