"""Performance metrics + stationary block bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult

TD = 252


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TD)) if sd > 0 else 0.0


def sortino(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / dd * np.sqrt(TD)) if dd > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def cagr(equity: pd.Series) -> float:
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (TD / n) - 1)


def episode_returns(res: BacktestResult) -> pd.Series:
    """Compounded return of each contiguous holding episode (same non-zero symbol)."""
    held = res.weights.idxmax(axis=1).where(res.weights.abs().sum(axis=1) > 0, "CASH")
    ep_id = (held != held.shift()).cumsum()
    out = []
    for _, grp in res.returns.groupby(ep_id):
        if held.loc[grp.index[0]] != "CASH":
            out.append((1 + grp).prod() - 1)
    return pd.Series(out, dtype=float)


def summarize(res: BacktestResult) -> dict[str, float]:
    r = res.returns.iloc[1:]
    gross = res.weights.abs().sum(axis=1).iloc[1:]
    invested = gross > 0
    eps = episode_returns(res)
    years = len(r) / TD
    return {
        "CAGR": cagr(res.equity),
        "AnnVol": float(r.std(ddof=1) * np.sqrt(TD)),
        "Sharpe": sharpe(r.values),
        "Sortino": sortino(r.values),
        "MaxDD": max_drawdown(res.equity),
        "Calmar": cagr(res.equity) / abs(max_drawdown(res.equity)) if max_drawdown(res.equity) < 0 else np.nan,
        "Turnover/yr": float(res.turnover.sum() / years) if years > 0 else 0.0,
        "Exposure": float(invested.mean()),
        "AvgGross": float(gross.mean()),
        "HitRate(days)": float((r[invested] > 0).mean()) if invested.any() else np.nan,
        "Trades": int(len(eps)),
        "WinRate(trades)": float((eps > 0).mean()) if len(eps) else np.nan,
        "CostDrag/yr": float(res.costs.sum() / years) if years > 0 else 0.0,
    }


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths, wrap-around."""
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new_block = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _sharpe_rows(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=1, ddof=1)
    return np.where(sd > 0, m.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(TD), 0.0)


def bootstrap_sharpe_ci(r: np.ndarray, n_boot: int = 2000, block: int = 20, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float]:
    r = np.asarray(r, float)
    idx = stationary_bootstrap_indices(len(r), n_boot, block, np.random.default_rng(seed))
    s = _sharpe_rows(r[idx])
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def paired_bootstrap_prob(r_a: np.ndarray, r_b: np.ndarray, n_boot: int = 2000, block: int = 20,
                          seed: int = 0) -> float:
    """P(Sharpe_a > Sharpe_b) using the SAME resampled days for both (preserves correlation)."""
    a, b = np.asarray(r_a, float), np.asarray(r_b, float)
    idx = stationary_bootstrap_indices(len(a), n_boot, block, np.random.default_rng(seed))
    return float(np.mean(_sharpe_rows(a[idx]) > _sharpe_rows(b[idx])))
