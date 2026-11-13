import logging
import os
import subprocess
import sys

import pytest

from tqqq_desk.live_guard import LiveModeError, check_account_matches_mode, resolve_trading_mode


def test_paper_is_default():
    assert resolve_trading_mode({}, False) == "paper"
    assert resolve_trading_mode({"ALPACA_LIVE": "false"}, False) == "paper"


def test_live_requires_both():
    assert resolve_trading_mode({"ALPACA_LIVE": "true"}, True) == "live"
    assert resolve_trading_mode({"ALPACA_LIVE": "TRUE "}, True) == "live"
    with pytest.raises(LiveModeError):
        resolve_trading_mode({"ALPACA_LIVE": "true"}, False)
    with pytest.raises(LiveModeError):
        resolve_trading_mode({}, True)
    with pytest.raises(LiveModeError):
        resolve_trading_mode({"ALPACA_LIVE": "yes"}, True)        # ambiguous value


def test_refuses_inside_pull_request():
    with pytest.raises(LiveModeError):
        resolve_trading_mode({"GITHUB_EVENT_NAME": "pull_request"}, False)
    with pytest.raises(LiveModeError):
        resolve_trading_mode({"GITHUB_EVENT_NAME": "pull_request_target"}, False)
    assert resolve_trading_mode({"GITHUB_EVENT_NAME": "pull_request"}, False, dry_run=True) == "paper"


def test_account_cross_check():
    check_account_matches_mode("paper", "PA123", "https://paper-api.alpaca.markets")
    check_account_matches_mode("live", "123456", "https://api.alpaca.markets")
    with pytest.raises(LiveModeError):
        check_account_matches_mode("paper", "123456", "https://api.alpaca.markets")
    with pytest.raises(LiveModeError):
        check_account_matches_mode("live", "PA123", "https://paper-api.alpaca.markets")


def test_cli_refuses_live_without_flag(tmp_path):
    env = dict(os.environ, ALPACA_LIVE="true", GITHUB_EVENT_NAME="")
    r = subprocess.run([sys.executable, "-m", "tqqq_desk.run_daily", "--state-dir", str(tmp_path)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "REFUSING TO START" in r.stderr


def test_secrets_are_redacted(monkeypatch, capsys):
    from tqqq_desk.logutil import setup_logging
    monkeypatch.setenv("ALPACA_KEY", "PKTESTKEY123456")
    monkeypatch.setenv("ALPACA_SECRET", "s3cr3tVALUEabcdef")
    root = logging.getLogger()
    old = list(root.handlers)
    root.handlers = [logging.StreamHandler(sys.stderr)]
    try:
        setup_logging()
        logging.getLogger("x").error("key=%s secret=%s", "PKTESTKEY123456", "s3cr3tVALUEabcdef")
        err = capsys.readouterr().err
        assert "PKTESTKEY123456" not in err and "s3cr3tVALUEabcdef" not in err and "REDACTED" in err
    finally:
        root.handlers = old
