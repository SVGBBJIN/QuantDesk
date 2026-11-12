"""Paper is the default. Live trading requires BOTH env ALPACA_LIVE=true AND the CLI flag
--i-understand-the-risk. Any other combination refuses to start. Never trades inside a PR workflow."""
from __future__ import annotations

from typing import Mapping

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class LiveModeError(RuntimeError):
    pass


def resolve_trading_mode(env: Mapping[str, str], risk_flag: bool, dry_run: bool = False) -> str:
    event = env.get("GITHUB_EVENT_NAME", "")
    if event.startswith("pull_request") and not dry_run:
        raise LiveModeError("refusing to trade inside a pull-request workflow (use --dry-run)")
    raw = env.get("ALPACA_LIVE", "")
    live_env = raw.strip().lower() == "true"
    if raw and not live_env and raw.strip().lower() not in ("false", "0", ""):
        raise LiveModeError(f"ALPACA_LIVE has an unrecognised value; use 'true' or 'false'")
    if live_env and risk_flag:
        return "live"
    if live_env:
        raise LiveModeError("ALPACA_LIVE=true but --i-understand-the-risk was not passed; refusing to start live")
    if risk_flag:
        raise LiveModeError("--i-understand-the-risk passed but ALPACA_LIVE is not 'true'; refusing (ambiguous intent)")
    return "paper"


def check_account_matches_mode(mode: str, account_number: str, base_url: str) -> None:
    """Belt and braces: paper accounts are 'PA...' and talk to paper-api."""
    is_paper_acct = str(account_number).upper().startswith("PA")
    is_paper_url = "paper-api" in base_url
    if mode == "paper" and not (is_paper_acct and is_paper_url):
        raise LiveModeError("paper mode but broker endpoint/account is not a paper account; refusing")
    if mode == "live" and (is_paper_acct or is_paper_url):
        raise LiveModeError("live mode but broker endpoint/account is a paper account; refusing")
