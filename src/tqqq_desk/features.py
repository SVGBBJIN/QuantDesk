"""Feature engineering. Every feature at date t uses only bars dated <= t (rolling / shift(+k)
windows only; no centered windows, no negative shifts, no full-sample statistics)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .mktcal import trading_days_to_month_end

log = logging.getLogger(__name__)
ANN = np.sqrt(252)


def _rv(logret: pd.Series, w: int) -> pd.Series:
    return logret.rolling(w, min_periods=w).std() * ANN


def _days_since_peak(close: pd.Series) -> pd.Series:
    peak = close.cummax()
    at_peak = close >= peak
    grp = at_peak.cumsum()
    return close.groupby(grp).cumcount().astype(float)


def qqq_features(q: pd.DataFrame) -> pd.DataFrame:
    c, h, l = q["close"], q["high"], q["low"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=q.index)
    sma = {n: c.rolling(n, min_periods=n).mean() for n in (50, 100, 200)}
    # trend
    for n, s in sma.items():
        f[f"px_vs_sma{n}"] = c / s - 1
    f["sma50_vs_sma200"] = sma[50] / sma[200] - 1
    f["sma50_slope"] = sma[50] / sma[50].shift(10) - 1
    f["sma200_slope"] = sma[200] / sma[200].shift(20) - 1
    for n in (20, 60, 120):
        f[f"mom{n}"] = c / c.shift(n) - 1
    f["dist_52w_high"] = c / c.rolling(252, min_periods=252).max() - 1
    # volatility
    for n in (10, 20, 60):
        f[f"rv{n}"] = _rv(lr, n)
    f["vol_of_vol"] = f["rv20"].rolling(60, min_periods=60).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr14_pct"] = tr.rolling(14, min_periods=14).mean() / c
    f["vol_ratio_10_60"] = f["rv10"] / f["rv60"]
    f["dd_depth"] = c / c.cummax() - 1
    f["dd_duration"] = _days_since_peak(c)
    return f


def context_features(qc: pd.Series, ctx: dict[str, pd.Series]) -> pd.DataFrame:
    f = pd.DataFrame(index=qc.index)
    if "SPY" in ctx:
        rs = qc / ctx["SPY"]
        f["qqq_spy_rs20"] = rs / rs.shift(20) - 1
        f["qqq_spy_rs60"] = rs / rs.shift(60) - 1
    if "TLT" in ctx:
        t = ctx["TLT"]
        f["tlt_mom20"] = t / t.shift(20) - 1
        f["tlt_mom60"] = t / t.shift(60) - 1
    if "UUP" in ctx:
        u = ctx["UUP"]
        f["uup_mom20"] = u / u.shift(20) - 1
    if "VIXY" in ctx:
        v = ctx["VIXY"]
        f["vixy_vs_sma20"] = v / v.rolling(20, min_periods=20).mean() - 1
        f["vixy_mom10"] = v / v.shift(10) - 1
    return f


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.day_of_week.astype(float)
    f["days_to_month_end"] = trading_days_to_month_end(idx).values
    return f


CALENDAR_COLS = ["dow", "days_to_month_end"]


def build_features(bars: dict[str, pd.DataFrame], cfg, include_calendar: bool = True) -> pd.DataFrame:
    sig = cfg["data"]["signal_symbol"]
    q = bars[sig]
    parts = [qqq_features(q)]
    ctx = {}
    min_cov = cfg["features"]["context_min_coverage"]
    for s in cfg["data"]["context_symbols"]:
        if s not in bars:
            log.warning("context symbol %s unavailable - skipped", s)
            continue
        px = bars[s]["close"].reindex(q.index)
        cov = px.notna().mean()
        if cov < min_cov:
            log.warning("context symbol %s coverage %.0f%% < %.0f%% - skipped", s, 100 * cov, 100 * min_cov)
            continue
        ctx[s] = px.ffill(limit=3)  # forward-fill only: uses past values, never future
    parts.append(context_features(q["close"], ctx))
    if include_calendar:
        parts.append(calendar_features(q.index))
    feats = pd.concat(parts, axis=1)
    return feats.replace([np.inf, -np.inf], np.nan)


def make_labels(close: pd.Series, horizon: int, vol_threshold: float) -> pd.DataFrame:
    """Labels at t look FORWARD (t+1..t+h). They are targets, never features.
    y_up:    1 if close[t+h] > close[t]
    y_hivol: 1 if annualized std of daily log returns over t+1..t+h > vol_threshold
    fwd_ret: close[t+h]/close[t]-1 (for diagnostics). Last h rows are NaN."""
    lr = np.log(close).diff()
    fwd = close.shift(-horizon) / close - 1
    fwd_vol = lr[::-1].rolling(horizon, min_periods=horizon).std()[::-1].shift(-1) * ANN
    y = pd.DataFrame(index=close.index)
    y["fwd_ret"] = fwd
    y["fwd_vol"] = fwd_vol
    y["y_up"] = np.where(fwd.notna(), (fwd > 0).astype(float), np.nan)
    y["y_hivol"] = np.where(fwd_vol.notna(), (fwd_vol > vol_threshold).astype(float), np.nan)
    return y
