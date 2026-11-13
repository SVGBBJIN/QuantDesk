"""Position sizing: min(volatility target, fractional Kelly, hard per-ETF cap).

Vol target:  w_vol = target_vol / sigma_ETF  (sigma from the ETF's OWN trailing realized vol)
Kelly:       for an h-day bet whose sign is right with prob p and whose magnitude ~ E|R_h|,
             mu ~= (2p - 1) * E|R_h|,  f* = mu / Var(R_h),  w_kelly = k * f*  with k <= 0.25.
Both inputs are trailing (causal) estimates of the ETF's h-day returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_KELLY


def vol_target_weight(etf_vol: float, target_vol: float) -> float:
    if not np.isfinite(etf_vol) or etf_vol <= 0:
        return 0.0
    return target_vol / etf_vol


def kelly_weight(p: float, mean_abs_h: float, var_h: float, kelly_fraction: float) -> float:
    if kelly_fraction > MAX_KELLY:
        raise ValueError(f"kelly_fraction {kelly_fraction} exceeds hard cap {MAX_KELLY}")
    if not (np.isfinite(p) and np.isfinite(mean_abs_h) and np.isfinite(var_h)) or var_h <= 0:
        return 0.0
    mu = (2 * p - 1) * mean_abs_h
    return max(0.0, kelly_fraction * mu / var_h)


def final_weight(symbol: str, p: float | None, etf_vol: float, mean_abs_h: float, var_h: float, s) -> float:
    """p=None -> rule-based mode: vol targeting + cap only (no probability -> no Kelly)."""
    w = vol_target_weight(etf_vol, s["target_vol"])
    if p is not None:
        w = min(w, kelly_weight(p, mean_abs_h, var_h, s["kelly_fraction"]))
    w = min(w, s["max_weight"][symbol])
    return 0.0 if w < s["min_weight"] else float(w)


def etf_risk_inputs(close: pd.Series, horizon: int, vol_lookback: int, kelly_lookback: int) -> pd.DataFrame:
    """Trailing ETF statistics known at the close of t."""
    lr = np.log(close).diff()
    rh = close / close.shift(horizon) - 1
    return pd.DataFrame({
        "vol": lr.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252),
        "mean_abs_h": rh.abs().rolling(kelly_lookback, min_periods=kelly_lookback // 2).mean(),
        "var_h": rh.rolling(kelly_lookback, min_periods=kelly_lookback // 2).var(),
    })
