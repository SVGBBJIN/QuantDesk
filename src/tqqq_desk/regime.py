"""Rule-based, interpretable regime classifier with hysteresis and a minimum holding period.

BULL:     QQQ well above SMA200 and SMA50 > SMA200
BEAR:     QQQ well below SMA200 and SMA50 < SMA200
HIGH_VOL: 20d realized vol above hv_enter (until it falls below hv_exit)
NEUTRAL:  everything else
Separate entry/exit thresholds prevent flip-flopping around a single line."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BULL, NEUTRAL, BEAR, HIGH_VOL = "BULL", "NEUTRAL", "BEAR", "HIGH_VOL"


@dataclass
class RegimeState:
    regime: str | None = None
    held: int = 0


def _trend(d: float, s: float, r) -> str:
    if d > r["bull_enter"] and s > 0:
        return BULL
    if d < r["bear_enter"] and s < 0:
        return BEAR
    return NEUTRAL


def regime_step(st: RegimeState, d: float, s: float, rv: float, r) -> RegimeState:
    """d = close/SMA200 - 1, s = SMA50/SMA200 - 1, rv = 20d realized vol (annualized)."""
    if not (np.isfinite(d) and np.isfinite(s) and np.isfinite(rv)):
        return RegimeState(None, 0)
    cur = st.regime
    if cur is None:
        cand = HIGH_VOL if rv > r["hv_enter"] else _trend(d, s, r)
        return RegimeState(cand, 1)
    if cur == HIGH_VOL:
        cand = _trend(d, s, r) if rv < r["hv_exit"] else HIGH_VOL
    elif rv > r["hv_enter"]:
        cand = HIGH_VOL
    elif cur == BULL:
        cand = BULL if d >= r["bull_exit"] else (BEAR if _trend(d, s, r) == BEAR else NEUTRAL)
    elif cur == BEAR:
        cand = BEAR if d <= r["bear_exit"] else (BULL if _trend(d, s, r) == BULL else NEUTRAL)
    else:
        cand = _trend(d, s, r)
    if cand != cur:
        can_switch = st.held >= r["min_hold_days"] or (cand == HIGH_VOL and r["highvol_overrides_min_hold"])
        if can_switch:
            return RegimeState(cand, 1)
    return RegimeState(cur, st.held + 1)


def classify_regimes(feats: pd.DataFrame, r) -> pd.Series:
    d = feats[f"px_vs_sma{r['sma_long']}"].to_numpy()
    s = feats["sma50_vs_sma200"].to_numpy()
    rv = feats["rv20"].to_numpy()
    st = RegimeState()
    out = []
    for i in range(len(feats)):
        st = regime_step(st, d[i], s[i], rv[i], r)
        out.append(st.regime)
    return pd.Series(out, index=feats.index, name="regime", dtype=object)
