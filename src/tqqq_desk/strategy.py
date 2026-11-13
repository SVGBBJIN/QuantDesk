"""Assemble regime + probabilities + decision state machine + sizing into a daily signal path."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decision import CASH, LONG, SHORT, PosState, decision_step
from .regime import classify_regimes
from .sizing import etf_risk_inputs, final_weight


def risk_inputs(bars, cfg) -> dict[str, pd.DataFrame]:
    s, h = cfg["sizing"], cfg["labels"]["horizon"]
    ts = cfg["data"]["trade_symbols"]
    return {sym: etf_risk_inputs(bars[sym]["close"], h, s["vol_lookback"], s["kelly_lookback"])
            for sym in (ts["long"], ts["short"])}


def size_for(state_pos: str, p_up, mode, risk_row: dict, cfg) -> float:
    if state_pos == CASH:
        return 0.0
    p = None if mode == "RULE" else (p_up if state_pos == LONG else 1 - p_up)
    r = risk_row[state_pos]
    return final_weight(state_pos, p, r["vol"], r["mean_abs_h"], r["var_h"], cfg["sizing"])


def build_signal_path(regime: pd.Series, probs: pd.DataFrame | None, bars, cfg, mode: str,
                      idx: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """mode: "ML" (needs probs) or "RULE". Rows = decision dates (decided after close of t)."""
    if idx is None:
        idx = probs.index if probs is not None else regime.index
    risk = {k: v.reindex(idx) for k, v in risk_inputs(bars, cfg).items()}
    rk = {k: {c: v[c].to_numpy() for c in v.columns} for k, v in risk.items()}
    reg = regime.reindex(idx).to_numpy()
    pu = probs["p_up"].reindex(idx).to_numpy() if probs is not None else np.full(len(idx), np.nan)
    ph = probs["p_hivol"].reindex(idx).to_numpy() if probs is not None else np.full(len(idx), np.nan)
    st = PosState()
    rows = []
    for i in range(len(idx)):
        r = reg[i] if isinstance(reg[i], str) else None
        st, _, reason = decision_step(st, r, pu[i], ph[i], cfg["decision"], cfg["fallback"], mode)
        row_risk = {sym: {c: rk[sym][c][i] for c in rk[sym]} for sym in rk}
        w = size_for(st.position, pu[i], mode, row_risk, cfg)
        if st.position != CASH and w == 0:
            reason += " | size below min weight -> effectively cash"
        rows.append((r, pu[i], ph[i], st.position, w if st.position == LONG else 0.0,
                     w if st.position == SHORT else 0.0, st.days_held, reason))
    return pd.DataFrame(rows, index=idx, columns=["regime", "p_up", "p_hivol", "state",
                                                  "w_TQQQ", "w_SQQQ", "days_held", "reason"])


def to_target_weights(sig: pd.DataFrame, cfg) -> pd.DataFrame:
    ts = cfg["data"]["trade_symbols"]
    return pd.DataFrame({ts["long"]: sig["w_TQQQ"], ts["short"]: sig["w_SQQQ"]}, index=sig.index)


def rule_signals(feats, bars, cfg, idx=None) -> pd.DataFrame:
    return build_signal_path(classify_regimes(feats, cfg["regime"]), None, bars, cfg, "RULE", idx)
