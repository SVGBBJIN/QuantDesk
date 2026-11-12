"""Daily backtest engine.

Timing (matches live execution with market-on-open orders):
  * target weights in row t are decided after the close of t,
  * they are executed at the OPEN of t+1 at the actual ETF open price (+ costs),
  * the overnight gap close[t] -> open[t+1] accrues to the OLD position,
    the intraday move open[t+1] -> close[t+1] to the NEW one.
Leveraged ETF returns come from TQQQ/SQQQ's own price history (decay included), never 3x QQQ.
Uninvested capital earns 0 (conservative; see README limitations)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def apply_rebalance_band(target: dict[str, float], current: dict[str, float], band: float) -> dict[str, float]:
    """Skip small re-sizings; always allow entries and exits (to/from zero) and symbol switches.
    Shared by backtest and live so both trade identically."""
    syms = set(target) | set(current)
    t = {s: target.get(s, 0.0) for s in syms}
    c = {s: current.get(s, 0.0) for s in syms}
    same_book = all((t[s] > 0) == (c[s] > 0) for s in syms)
    if same_book and any(t[s] > 0 for s in syms) and all(abs(t[s] - c[s]) < band for s in syms):
        return c
    return t


@dataclass
class BacktestResult:
    returns: pd.Series                     # daily portfolio returns (close-to-close)
    equity: pd.Series
    weights: pd.DataFrame                  # post-trade weights at each day's open
    turnover: pd.Series                    # sum |traded notional| / equity per day
    costs: pd.Series
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_backtest(target_w: pd.DataFrame, bars: dict[str, pd.DataFrame], cost_bps: float,
                 band: float = 0.0, initial_equity: float = 1.0) -> BacktestResult:
    target_w = target_w.fillna(0.0)
    syms = list(target_w.columns)
    idx = target_w.index
    opens = pd.DataFrame({s: bars[s]["open"].reindex(idx) for s in syms})
    closes = pd.DataFrame({s: bars[s]["close"].reindex(idx) for s in syms})
    O, C, W = opens.to_numpy(float), closes.to_numpy(float), target_w.to_numpy(float)
    n, k = W.shape
    cost_rate = cost_bps / 1e4

    eq = np.full(n, np.nan)
    ret = np.zeros(n)
    wts = np.zeros((n, k))
    tov = np.zeros(n)
    cst = np.zeros(n)
    trades = []
    v = np.zeros(k)          # dollar value per symbol at the previous close
    cash = initial_equity
    eq[0] = initial_equity
    for d in range(1, n):
        # 1) overnight gap on existing holdings
        held = v != 0
        if held.any():
            if not np.isfinite(O[d, held]).all() or not np.isfinite(C[d - 1, held]).all():
                raise ValueError(f"missing price for a held symbol on {idx[d].date()}")
            v[held] *= O[d, held] / C[d - 1, held]
        E_open = cash + v.sum()
        # 2) execute yesterday's decision at today's open
        cur = {syms[j]: v[j] / E_open for j in range(k)} if E_open > 0 else {}
        tgt = {syms[j]: W[d - 1, j] for j in range(k)}
        tgt = apply_rebalance_band(tgt, cur, band)
        new_v = np.array([tgt[s] * E_open for s in syms])
        if (new_v != 0).any() and not np.isfinite(O[d, new_v != 0]).all():
            raise ValueError(f"missing open price for a target symbol on {idx[d].date()}")
        traded = np.abs(new_v - v)
        if traded.sum() > 1e-12 * E_open:
            c = traded.sum() * cost_rate
            for j in np.nonzero(traded > 1e-12 * E_open)[0]:
                trades.append({"date": idx[d], "symbol": syms[j], "from_w": v[j] / E_open,
                               "to_w": tgt[syms[j]], "notional": new_v[j] - v[j]})
            cash -= (new_v - v).sum() + c
            v = new_v
            tov[d] = traded.sum() / E_open
            cst[d] = c / E_open
        wts[d] = v / E_open if E_open > 0 else 0
        # 3) intraday move
        held = v != 0
        if held.any():
            v[held] *= C[d, held] / O[d, held]
        eq[d] = cash + v.sum()
        ret[d] = eq[d] / eq[d - 1] - 1
    return BacktestResult(
        returns=pd.Series(ret, idx, name="ret"),
        equity=pd.Series(eq, idx, name="equity"),
        weights=pd.DataFrame(wts, idx, syms),
        turnover=pd.Series(tov, idx, name="turnover"),
        costs=pd.Series(cst, idx, name="costs"),
        trades=pd.DataFrame(trades),
    )
