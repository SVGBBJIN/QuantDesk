"""Alpaca real-time market-data WebSocket.

``QuoteHub`` keeps the latest trade/quote per symbol and fans updates out to subscribers.
``AlpacaStream`` runs alpaca-py's ``StockDataStream`` in a background thread and feeds the hub.

The free Alpaca plan only has the IEX feed (a fraction of consolidated volume) - the default here.
Data-only: this module never places orders.

    python -m tqqq_desk.stream QQQ TQQQ SQQQ      # print live updates (needs ALPACA_KEY / ALPACA_SECRET)
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class QuoteHub:
    """Thread-safe latest-value store + pub/sub. Safe to call from the stream thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.status: dict[str, Any] = {"connected": False, "feed": None, "error": None, "updated": None}

    # --- writers
    def update(self, symbol: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            row = self._state.setdefault(symbol, {"symbol": symbol})
            row.update({k: v for k, v in fields.items() if v is not None})
            row["ts"] = fields.get("ts") or time.time()
            self.status["updated"] = row["ts"]
            snap = dict(row)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(snap)
            except Exception:  # noqa: BLE001 - a bad listener must not kill the stream
                log.exception("quote listener failed")
        return snap

    def on_trade(self, symbol: str, price: float, size: float | None = None, ts: float | None = None) -> None:
        self.update(symbol, price=float(price), size=size, ts=ts, source="alpaca")

    def on_quote(self, symbol: str, bid: float, ask: float, ts: float | None = None) -> None:
        # only fall back to the mid when no trade has printed yet
        row = self.get(symbol) or {}
        self.update(symbol, bid=float(bid), ask=float(ask), ts=ts, source="alpaca",
                    price=None if "price" in row else (bid + ask) / 2 if bid and ask else None)

    def seed(self, symbol: str, price: float, prev_close: float | None) -> None:
        """Seed prev_close (for day change) and a price if the stream hasn't produced one yet."""
        row = self.get(symbol) or {}
        self.update(symbol, prev_close=prev_close, price=None if "price" in row else price,
                    source=row.get("source") or "yahoo", ts=row.get("ts") or time.time())

    # --- readers
    def get(self, symbol: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._state.get(symbol)
            return dict(row) if row else None

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            out = {}
            for s, r in self._state.items():
                row = dict(r)
                p, pc = row.get("price"), row.get("prev_close")
                if p is not None and pc:
                    row["change"] = p - pc
                    row["change_pct"] = (p - pc) / pc * 100
                out[s] = row
            return out

    def subscribe(self, fn: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._listeners:
                    self._listeners.remove(fn)
        return unsubscribe


class AlpacaStream:
    def __init__(self, symbols: list[str], hub: QuoteHub, feed: str = "iex",
                 key: str | None = None, secret: str | None = None) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.hub = hub
        self.feed = feed.lower()
        self._key, self._secret = key, secret
        self._thread: threading.Thread | None = None
        self._stream = None
        self._stop = threading.Event()

    # handlers run on the stream thread's event loop
    async def _on_trade(self, t) -> None:
        self.hub.on_trade(t.symbol, t.price, getattr(t, "size", None), t.timestamp.timestamp())

    async def _on_quote(self, q) -> None:
        self.hub.on_quote(q.symbol, q.bid_price, q.ask_price, q.timestamp.timestamp())

    def _build(self):
        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream

        from .data import credentials
        key, sec = (self._key, self._secret) if self._key and self._secret else credentials()
        stream = StockDataStream(key, sec, feed=DataFeed(self.feed))
        stream.subscribe_trades(self._on_trade, *self.symbols)
        stream.subscribe_quotes(self._on_quote, *self.symbols)
        return stream

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._stream = self._build()
                self.hub.status.update(connected=True, feed=self.feed, error=None)
                log.info("alpaca stream connecting (%s): %s", self.feed, ",".join(self.symbols))
                self._stream.run()  # blocks until stop() or a fatal error
                backoff = 1.0
            except Exception as e:  # noqa: BLE001 - reconnect on any failure (e.g. 406 connection limit)
                self.hub.status.update(connected=False, error=f"{type(e).__name__}: {e}")
                log.warning("alpaca stream error: %s (retry in %.0fs)", e, backoff)
            finally:
                self.hub.status["connected"] = False
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 60.0)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alpaca-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:  # noqa: BLE001
                log.debug("stream.stop() failed", exc_info=True)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Print live Alpaca trades/quotes")
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    hub = QuoteHub()
    hub.subscribe(lambda r: print(f"{r['symbol']:<6} {r.get('price', float('nan')):>10.2f} "
                                  f"bid={r.get('bid')} ask={r.get('ask')}"))
    s = AlpacaStream(a.symbols, hub, a.feed)
    s.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        s.stop()


if __name__ == "__main__":
    main()
