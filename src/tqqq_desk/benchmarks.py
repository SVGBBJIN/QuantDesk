"""Benchmark target-weight series (same engine, same costs, same execution timing)."""
from __future__ import annotations

import pandas as pd


def buy_and_hold(idx: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    return pd.DataFrame({symbol: 1.0}, index=idx)


def sma200_rule(qqq_close: pd.Series, idx: pd.DatetimeIndex, long_sym: str = "TQQQ", n: int = 200) -> pd.DataFrame:
    """TQQQ when QQQ close > its 200-day SMA (known at the close of t), else cash."""
    sma = qqq_close.rolling(n, min_periods=n).mean()
    w = (qqq_close > sma).astype(float).where(sma.notna(), 0.0)
    return pd.DataFrame({long_sym: w.reindex(idx).fillna(0.0)})


def all_benchmarks(bars, cfg, idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    sig = cfg["data"]["signal_symbol"]
    long_sym = cfg["data"]["trade_symbols"]["long"]
    return {
        f"B&H {sig}": buy_and_hold(idx, sig),
        f"B&H {long_sym}": buy_and_hold(idx, long_sym),
        f"{long_sym} if {sig}>SMA200": sma200_rule(bars[sig]["close"], idx, long_sym),
    }
