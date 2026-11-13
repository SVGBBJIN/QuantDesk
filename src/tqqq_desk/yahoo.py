"""Free Yahoo Finance scraper (no API key, no yfinance dependency).

Uses Yahoo's public ``/v8/finance/chart`` JSON endpoint. This is an unofficial endpoint: it can
rate-limit or change without notice, so this module is for dashboards / sanity checks / research,
NOT for the trading path (which stays on Alpaca bars).

    python -m tqqq_desk.yahoo AAPL MSFT            # quotes
    python -m tqqq_desk.yahoo QQQ --history 6mo    # daily bars
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass

import pandas as pd
import requests

log = logging.getLogger(__name__)

HOSTS = ("query1", "query2")  # either mirror may be rate-limited independently
BASE = "https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}"
# Yahoo 429s some long browser UA strings from datacenter IPs; a short one passes.
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
OHLCV = ["open", "high", "low", "close", "volume"]


class YahooError(RuntimeError):
    pass


@dataclass
class Quote:
    symbol: str
    price: float
    prev_close: float | None
    change: float | None
    change_pct: float | None
    currency: str | None
    market_state: str | None
    ts: float
    source: str = "yahoo"

    def to_dict(self) -> dict:
        return asdict(self)


_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session


def _fetch(symbol: str, params: dict, retries: int = 3, timeout: float = 10.0) -> dict:
    """GET one chart payload; retry on 429/5xx with exponential backoff."""
    sym = requests.utils.quote(symbol.upper(), safe="")
    last: Exception | None = None
    for attempt in range(retries):
        url = BASE.format(host=HOSTS[attempt % len(HOSTS)], symbol=sym)
        try:
            r = _get_session().get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise YahooError(f"HTTP {r.status_code} for {symbol}")
            if r.status_code == 404:
                raise YahooError(f"unknown symbol {symbol}")  # not retryable
            r.raise_for_status()
            chart = r.json().get("chart", {})
            if chart.get("error"):
                raise YahooError(f"{symbol}: {chart['error'].get('description', chart['error'])}")
            res = chart.get("result")
            if not res:
                raise YahooError(f"{symbol}: empty result")
            return res[0]
        except YahooError as e:
            last = e
            if "unknown symbol" in str(e) or "No data found" in str(e):
                break
        except (requests.RequestException, ValueError) as e:
            last = e
        if attempt < retries - 1:
            time.sleep(0.5 * 2 ** attempt)
    raise YahooError(str(last))


def parse_history(payload: dict) -> pd.DataFrame:
    """chart payload -> OHLCV frame indexed by exchange-local date (daily) or UTC time (intraday)."""
    ts = payload.get("timestamp") or []
    q = ((payload.get("indicators") or {}).get("quote") or [{}])[0]
    if not ts or not q:
        return pd.DataFrame(columns=OHLCV)
    df = pd.DataFrame({c: q.get(c) for c in OHLCV}, index=pd.to_datetime(ts, unit="s", utc=True))
    adj = ((payload.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if adj:  # scale OHLC by the adjustment factor so returns are total returns
        f = pd.Series(adj, index=df.index) / df["close"]
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * f
    tz = (payload.get("meta") or {}).get("exchangeTimezoneName")
    if tz:
        df.index = df.index.tz_convert(tz)
    df = df.dropna(subset=["close"]).astype(float)
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def history(symbol: str, range_: str = "1y", interval: str = "1d", adjusted: bool = True) -> pd.DataFrame:
    """Historical bars. interval: 1m,5m,15m,30m,60m,1d,1wk,1mo; range_: 1d,5d,1mo,3mo,6mo,1y,5y,max."""
    payload = _fetch(symbol, {"range": range_, "interval": interval, "includeAdjustedClose": "true",
                              "events": "div,splits"})
    df = parse_history(payload)
    if not adjusted and not df.empty:  # re-fetch is wasteful; the raw close is in the payload
        raw = pd.Series(payload["indicators"]["quote"][0]["close"], index=pd.to_datetime(
            payload["timestamp"], unit="s", utc=True)).reindex(df.index.tz_convert("UTC"))
        df["close"] = raw.values
    if interval in ("1d", "1wk", "1mo") and not df.empty:
        df.index = pd.DatetimeIndex(df.index.date, name="date")  # tz-naive session date, like data.py
    return df


def parse_quote(payload: dict) -> Quote:
    m = payload.get("meta") or {}
    price = m.get("regularMarketPrice")
    if price is None:
        raise YahooError(f"{m.get('symbol', '?')}: no price in payload")
    prev = m.get("chartPreviousClose", m.get("previousClose"))
    chg = (price - prev) if prev else None
    return Quote(symbol=m.get("symbol", ""), price=float(price), prev_close=prev, change=chg,
                 change_pct=(chg / prev * 100) if prev else None, currency=m.get("currency"),
                 market_state=m.get("marketState"), ts=float(m.get("regularMarketTime") or time.time()))


def quote(symbol: str) -> Quote:
    return parse_quote(_fetch(symbol, {"range": "1d", "interval": "1m"}))


def quotes(symbols: list[str]) -> dict[str, Quote]:
    """Quotes for many symbols; symbols that fail are skipped (and logged), never raised."""
    out: dict[str, Quote] = {}
    for s in symbols:
        try:
            out[s.upper()] = quote(s)
        except YahooError as e:
            log.warning("yahoo quote failed: %s", e)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Free Yahoo Finance quotes / history")
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--history", metavar="RANGE", help="print daily bars for this range (e.g. 6mo, 1y)")
    ap.add_argument("--interval", default="1d")
    a = ap.parse_args(argv)
    for s in a.symbols:
        if a.history:
            print(f"== {s} ==\n{history(s, a.history, a.interval).tail(15)}")
        else:
            q = quote(s)
            print(f"{q.symbol:<8}{q.price:>10.2f}  {q.change_pct or 0:+.2f}%  [{q.market_state}]")


if __name__ == "__main__":
    main()
