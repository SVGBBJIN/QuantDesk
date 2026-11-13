"""Offline tests for the Yahoo scraper parsing, the quote hub and the relay server."""
import pandas as pd
import pytest

from tqqq_desk import yahoo
from tqqq_desk.stream import QuoteHub

PAYLOAD = {
    "meta": {"symbol": "QQQ", "regularMarketPrice": 110.0, "chartPreviousClose": 100.0, "currency": "USD",
             "regularMarketTime": 1_700_000_000, "exchangeTimezoneName": "America/New_York"},
    "timestamp": [1_699_900_000, 1_699_986_400, 1_700_072_800],
    "indicators": {"quote": [{"open": [1, 2, 3], "high": [2, 3, 4], "low": [0.5, 1, 2], "close": [1.5, None, 3.5],
                              "volume": [10, 20, 30]}],
                   "adjclose": [{"adjclose": [0.75, None, 3.5]}]},
}


def test_parse_quote_change():
    q = yahoo.parse_quote(PAYLOAD)
    assert q.symbol == "QQQ" and q.price == 110.0
    assert q.change == pytest.approx(10.0) and q.change_pct == pytest.approx(10.0)


def test_parse_quote_requires_price():
    with pytest.raises(yahoo.YahooError):
        yahoo.parse_quote({"meta": {"symbol": "X"}})


def test_parse_history_drops_nan_close_and_applies_adjustment():
    df = yahoo.parse_history(PAYLOAD)
    assert len(df) == 2 and list(df.columns) == yahoo.OHLCV
    assert df["close"].iloc[0] == pytest.approx(0.75)           # adjclose wins
    assert df["open"].iloc[0] == pytest.approx(0.5)             # open scaled by 0.75/1.5
    assert df.index.is_monotonic_increasing


def test_parse_history_empty():
    assert yahoo.parse_history({"meta": {}}).empty


def test_hub_trade_quote_and_change():
    hub = QuoteHub()
    seen = []
    unsub = hub.subscribe(seen.append)
    hub.seed("QQQ", 99.0, 100.0)
    hub.on_quote("QQQ", 100.0, 102.0)           # mid used only because no trade yet... seed price exists
    hub.on_trade("QQQ", 105.0, 10)
    snap = hub.snapshot()["QQQ"]
    assert snap["price"] == 105.0 and snap["bid"] == 100.0 and snap["ask"] == 102.0
    assert snap["change_pct"] == pytest.approx(5.0)
    n = len(seen)
    unsub()
    hub.on_trade("QQQ", 106.0)
    assert len(seen) == n


def test_hub_bad_listener_does_not_break_updates():
    hub = QuoteHub()
    hub.subscribe(lambda r: 1 / 0)
    hub.on_trade("SPY", 1.0)
    assert hub.get("SPY")["price"] == 1.0


def test_server_endpoints_without_alpaca_keys(monkeypatch):
    from fastapi.testclient import TestClient
    for k in ("ALPACA_KEY", "ALPACA_SECRET", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("QD_SYMBOLS", "QQQ")
    monkeypatch.setattr(yahoo, "quotes", lambda syms: {"QQQ": yahoo.parse_quote(PAYLOAD)})
    monkeypatch.setattr(yahoo, "history", lambda s, r="1y", i="1d": pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [9.0]},
        index=pd.DatetimeIndex(["2026-01-02"], name="date")))
    from tqqq_desk.server import app
    with TestClient(app) as c:
        assert "Yahoo data only" in c.get("/api/health").json()["error"]
        quotes = {}
        for _ in range(50):  # the yahoo loop runs in the background
            quotes = c.get("/api/quotes").json()["quotes"]
            if quotes.get("QQQ", {}).get("price"):
                break
            import time; time.sleep(0.1)
        assert quotes["QQQ"]["price"] == 110.0 and quotes["QQQ"]["source"] == "yahoo"
        h = c.get("/api/history/qqq").json()
        assert h["symbol"] == "QQQ" and h["bars"][0]["close"] == 1.5
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "snapshot"
            app.state.hub.on_trade("QQQ", 111.0)
            assert ws.receive_json()["quote"]["price"] == 111.0
