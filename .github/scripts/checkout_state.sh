#!/usr/bin/env bash
# Check out the `data` branch (bot state) into ./state as a git worktree; create it as an orphan if missing.
set -euo pipefail
git config --global user.name  "tqqq-desk-bot"
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
if git ls-remote --exit-code --heads origin data >/dev/null 2>&1; then
  git fetch --depth=1 origin data
  git worktree add -B data state FETCH_HEAD
else
  git worktree add --detach state
  (cd state && git checkout --orphan data && git rm -rfq . && \
   printf '# tqqq-desk bot state\nWritten by GitHub Actions. Do not edit by hand unless reconciling.\n' > README.md)
fi
