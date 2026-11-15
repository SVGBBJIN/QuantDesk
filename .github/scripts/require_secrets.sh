#!/usr/bin/env bash
# Fail fast, with a readable error, when the Alpaca API secrets are not configured for Actions.
set -euo pipefail
missing=()
[ -n "${ALPACA_KEY:-}" ]    || missing+=(ALPACA_KEY)
[ -n "${ALPACA_SECRET:-}" ] || missing+=(ALPACA_SECRET)
if [ ${#missing[@]} -gt 0 ]; then
  url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-<owner>/<repo>}/settings/secrets/actions"
  msg="Missing repository secret(s): ${missing[*]}. Add your Alpaca API key ID and secret key at ${url} (Settings -> Secrets and variables -> Actions -> New repository secret), then re-run this workflow."
  echo "::error title=Alpaca secrets not configured::${msg}"
  {
    echo "### ❌ Alpaca secrets not configured"
    echo
    echo "${msg}"
  } >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
  exit 1
fi
echo "Alpaca secrets present."
