#!/usr/bin/env bash
# Commit + push ./state to the `data` branch (retries on transient push failures).
set -euo pipefail
cd state
git add -A
if git diff --cached --quiet; then echo "no state changes"; exit 0; fi
git commit -qm "bot: ${1:-state update} $(date -u +%Y-%m-%dT%H:%MZ)"
for i in 1 2 3 4; do
  if git push origin HEAD:data; then exit 0; fi
  sleep $((2 ** i))
  git fetch --depth=1 origin data && git rebase FETCH_HEAD || true
done
echo "push failed" >&2; exit 1
