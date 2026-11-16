"""ML threshold sensitivity grid (rule strategies use parameter neighbourhoods in research.py). We report the grid and recommend the centre of the most stable
plateau (best full 3x3 neighbourhood mean Sharpe; interior cells only), NOT the single best cell. Nothing is auto-applied:
choosing thresholds on the OOS period would itself be in-sample selection."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .metrics import sharpe
from .strategy import build_signal_path, to_target_weights


def _cost(cfg):
    return cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"]


def _sr(sig, bars, cfg):
    res = run_backtest(to_target_weights(sig, cfg), bars, _cost(cfg), cfg["sizing"]["rebalance_band"])
    return sharpe(res.returns.iloc[1:].to_numpy())


def ml_grid(regime, probs, bars, cfg) -> pd.DataFrame:
    g = cfg["sensitivity"]["ml"]
    d = cfg["decision"]
    gap_up = d["up_enter"] - d["up_exit"]
    out = pd.DataFrame(index=pd.Index(g["up_enter"], name="up_enter"),
                       columns=pd.Index(g["vol_cap"], name="vol_cap"), dtype=float)
    for ue, vc in itertools.product(g["up_enter"], g["vol_cap"]):
        c = cfg.copy_with({"decision.up_enter": ue, "decision.up_exit": ue - gap_up, "decision.vol_cap": vc,
                           "decision.down_enter": max(d["down_enter"], ue),
                           "decision.down_exit": max(d["down_exit"], ue - gap_up)})
        out.loc[ue, vc] = _sr(build_signal_path(regime, probs, bars, c, "ML"), bars, c)
    return out


def plateau(grid: pd.DataFrame) -> dict:
    a = grid.to_numpy(float)
    n, m = a.shape
    sm = np.full_like(a, np.nan)
    worst = np.full_like(a, np.nan)
    for i in range(n):
        for j in range(m):
            blk = a[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            sm[i, j] = np.nanmean(blk)
            worst[i, j] = np.nanmin(blk)
    bi, bj = np.unravel_index(np.nanargmax(a), a.shape)
    # plateau centre must have a full 3x3 neighbourhood (edge cells have less evidence of stability)
    interior = np.full_like(sm, np.nan)
    if n >= 3 and m >= 3:
        interior[1:-1, 1:-1] = sm[1:-1, 1:-1]
    else:
        interior = sm
    pi, pj = np.unravel_index(np.nanargmax(interior), sm.shape)
    return {
        "best_point": {grid.index.name: grid.index[bi], grid.columns.name: grid.columns[bj], "sharpe": a[bi, bj]},
        "plateau_center": {grid.index.name: grid.index[pi], grid.columns.name: grid.columns[pj],
                           "sharpe": a[pi, pj], "neighbourhood_mean": sm[pi, pj], "neighbourhood_min": worst[pi, pj]},
        "smoothed": pd.DataFrame(sm, index=grid.index, columns=grid.columns),
    }
