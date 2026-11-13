"""Local market-data relay: Alpaca WebSocket in, REST + WebSocket out, Yahoo as free fallback.

Alpaca keys stay on this process; the dashboard only talks to localhost.

    uvicorn tqqq_desk.server:app --port 8765
    QD_SYMBOLS=QQQ,TQQQ,SQQQ,SPY ALPACA_KEY=... ALPACA_SECRET=... uvicorn tqqq_desk.server:app

    GET /api/health            stream status
    GET /api/quotes            latest quote per symbol
    GET /api/history/{symbol}  daily bars from Yahoo (?range=1y&interval=1d)
    WS  /ws                    pushes every quote update as JSON
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from . import yahoo
from .stream import AlpacaStream, QuoteHub

log = logging.getLogger(__name__)
DEFAULT_SYMBOLS = "QQQ,TQQQ,SQQQ,SPY,TLT"
YAHOO_REFRESH_S = 15.0


def symbols_from_env() -> list[str]:
    return [s.strip().upper() for s in os.environ.get("QD_SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]


async def _yahoo_loop(hub: QuoteHub, symbols: list[str], stop: asyncio.Event) -> None:
    """Free fallback: seeds prev_close and keeps prices fresh whenever the Alpaca stream is down."""
    while not stop.is_set():
        got = await asyncio.to_thread(yahoo.quotes, symbols)
        for s, q in got.items():
            row = hub.get(s) or {}
            hub.update(s, prev_close=q.prev_close)
            if not hub.status["connected"] or "price" not in row:
                hub.update(s, price=q.price, source="yahoo", ts=q.ts)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), YAHOO_REFRESH_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub = QuoteHub()
    symbols = symbols_from_env()
    app.state.hub, app.state.symbols = hub, symbols
    stop = asyncio.Event()
    stream = None
    if (os.environ.get("ALPACA_KEY") or os.environ.get("APCA_API_KEY_ID")) and \
            (os.environ.get("ALPACA_SECRET") or os.environ.get("APCA_API_SECRET_KEY")):
        stream = AlpacaStream(symbols, hub, feed=os.environ.get("QD_FEED", "iex"))
        stream.start()
    else:
        hub.status["error"] = "ALPACA_KEY / ALPACA_SECRET not set - serving Yahoo data only"
        log.warning(hub.status["error"])
    task = asyncio.create_task(_yahoo_loop(hub, symbols, stop))
    try:
        yield
    finally:
        stop.set()
        if stream:
            stream.stop()
        await task


app = FastAPI(title="QuantDesk market data", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"symbols": app.state.symbols, **app.state.hub.status}


@app.get("/api/quotes")
def get_quotes():
    return {"status": app.state.hub.status, "quotes": app.state.hub.snapshot()}


@app.get("/api/history/{symbol}")
async def get_history(symbol: str, range: str = "1y", interval: str = "1d"):
    try:
        df = await asyncio.to_thread(yahoo.history, symbol, range, interval)
    except yahoo.YahooError as e:
        raise HTTPException(502, str(e)) from e
    df = df.reset_index()
    df["date"] = df["date"].astype(str)
    return {"symbol": symbol.upper(), "bars": df.to_dict("records")}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    hub: QuoteHub = app.state.hub
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)

    def push(row):  # called from the Alpaca/Yahoo threads
        def put():
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()  # drop oldest rather than block the producer
            q.put_nowait(row)
        loop.call_soon_threadsafe(put)

    unsub = hub.subscribe(push)
    try:
        await sock.send_json({"type": "snapshot", "quotes": hub.snapshot()})
        while True:
            await sock.send_json({"type": "quote", "quote": await q.get()})
    except WebSocketDisconnect:
        pass
    finally:
        unsub()
