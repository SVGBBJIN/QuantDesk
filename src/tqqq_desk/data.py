"""Daily bar loading from Alpaca (with local CSV cache) + a synthetic generator for tests.

All frames: index = tz-naive session date (America/New_York), columns open/high/low/close/volume,
split+dividend adjusted (Adjustment.ALL) so returns are total returns.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
OHLCV = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    pass


def credentials() -> tuple[str, str]:
    key = os.environ.get("ALPACA_KEY") or os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("ALPACA_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise DataError("ALPACA_KEY / ALPACA_SECRET env vars are not set")
    return key, sec


def all_symbols(cfg) -> list[str]:
    d = cfg["data"]
    return [d["signal_symbol"], *d["trade_symbols"].values(), *d["context_symbols"]]


def fetch_alpaca_bars(symbols: list[str], start: str, end: str | None, feed: str,
                      adjustment: str) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, sec = credentials()
    client = StockHistoricalDataClient(key, sec)
    req = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
        start=pd.Timestamp(start, tz="UTC").to_pydatetime(),
        end=pd.Timestamp(end, tz="UTC").to_pydatetime() if end else None,
        adjustment=Adjustment(adjustment), feed=DataFeed(feed),
    )
    raw = client.get_stock_bars(req).df
    out: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out
    for sym in symbols:
        if sym not in raw.index.get_level_values(0):
            continue
        df = raw.xs(sym, level=0)[OHLCV].copy()
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_convert("America/New_York") if idx.tz is not None else idx
        df.index = pd.DatetimeIndex(idx.date).rename("date")
        out[sym] = df[~df.index.duplicated(keep="last")].sort_index().astype(float)
    return out


def load_bars(cfg, refresh: bool = True, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Fetch full history from config start (so live features match the backtest) and cache it."""
    d = cfg["data"]
    syms = all_symbols(cfg)
    cache = Path(d["cache_dir"])
    if refresh:
        bars = fetch_alpaca_bars(syms, d["start"], end, d["feed"], d["adjustment"])
        cache.mkdir(parents=True, exist_ok=True)
        for s, df in bars.items():
            df.to_csv(cache / f"{s}.csv")
    else:
        bars = {}
        for s in syms:
            f = cache / f"{s}.csv"
            if f.exists():
                bars[s] = pd.read_csv(f, index_col=0, parse_dates=True)
    missing = [s for s in [d["signal_symbol"], *d["trade_symbols"].values()] if s not in bars]
    if missing:
        raise DataError(f"required symbols missing from data: {missing}")
    return bars


def drop_incomplete(bars: dict[str, pd.DataFrame], last_session: dt.date) -> dict[str, pd.DataFrame]:
    """Drop any bar dated after the last *completed* session (an in-progress daily bar)."""
    cut = pd.Timestamp(last_session)
    return {s: df[df.index <= cut] for s, df in bars.items()}


def staleness_report(bars: dict[str, pd.DataFrame], symbols: list[str],
                     expected_last: dt.date) -> dict[str, str]:
    """Return {symbol: problem} for symbols that are missing or whose last bar != expected."""
    problems = {}
    for s in symbols:
        df = bars.get(s)
        if df is None or df.empty:
            problems[s] = "missing"
            continue
        last = df.index[-1].date()
        if last != expected_last:
            problems[s] = f"last bar {last} != expected {expected_last}"
        elif not np.isfinite(df.iloc[-1][["open", "close"]].astype(float)).all():
            problems[s] = "non-finite last bar"
    return problems


# --------------------------------------------------------------------------- synthetic
def _ohlc_from_close(close: np.ndarray, rng: np.random.Generator, gap_sd: float) -> pd.DataFrame:
    prev = np.r_[close[0], close[:-1]]
    open_ = prev * np.exp(rng.normal(0, gap_sd, len(close)))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, gap_sd, len(close))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, gap_sd, len(close))))
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": rng.integers(1e6, 5e6, len(close)).astype(float)})


def synthetic_bars(n: int = 2600, seed: int = 0, start: str = "2016-01-04",
                   regime_switching: bool = True) -> dict[str, pd.DataFrame]:
    """Regime-switching GBM for QQQ; TQQQ/SQQQ built as daily-rebalanced +/-3x of QQQ's
    open->close and close->open legs minus fee/financing drag (so they decay like the real ETFs).
    FOR TESTS AND SMOKE RUNS ONLY - never evidence of edge."""
    from .mktcal import schedule
    rng = np.random.default_rng(seed)
    sess = schedule(start, str((pd.Timestamp(start) + pd.Timedelta(days=int(n * 1.6))).date())).index[:n]
    n = len(sess)
    # 2-state Markov regime: calm-bull / stressed-bear
    state = np.zeros(n, dtype=int)
    for i in range(1, n if regime_switching else 1):
        p_stay = 0.995 if state[i - 1] == 0 else 0.98
        state[i] = state[i - 1] if rng.random() < p_stay else 1 - state[i - 1]
    mu = np.where(state == 0, 0.0008, -0.0010) if regime_switching else np.zeros(n)
    sd = np.where(state == 0, 0.011, 0.028)
    r = rng.normal(mu, sd)
    qqq_close = 100 * np.exp(np.cumsum(r))
    q = _ohlc_from_close(qqq_close, rng, 0.003)
    gap = q["open"].values / np.r_[q["close"].values[0], q["close"].values[:-1]] - 1
    intra = q["close"].values / q["open"].values - 1
    drag = 0.0095 / 252 + 0.02 / 252  # expense ratio + financing

    def lev(k):
        c = 100 * np.cumprod((1 + k * gap) * (1 + k * intra) * (1 - drag))
        o = np.r_[100, c[:-1]] * (1 + k * gap)
        hi, lo = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
        return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1e7})

    def ctx(beta, sd_, drift):
        cl = 100 * np.exp(np.cumsum(beta * r + rng.normal(drift, sd_, n)))
        return _ohlc_from_close(cl, rng, 0.002)

    out = {"QQQ": q, "TQQQ": lev(3), "SQQQ": lev(-3), "SPY": ctx(0.8, 0.003, 0.0),
           "TLT": ctx(-0.2, 0.008, 0.0001), "UUP": ctx(-0.05, 0.004, 0.0),
           "VIXY": ctx(-3.0, 0.03, -0.002)}
    for df in out.values():
        df.index = pd.DatetimeIndex(sess, name="date")
    return out
