import numpy as np
import pandas as pd
import pytest

from tqqq_desk.backtest import apply_rebalance_band, run_backtest
from tqqq_desk.benchmarks import sma200_rule
from tqqq_desk.metrics import (bootstrap_sharpe_ci, max_drawdown, paired_bootstrap_prob, sharpe,
                               sortino, summarize)


def _bars():
    idx = pd.bdate_range("2024-01-01", periods=5)
    a = pd.DataFrame({"open": [100, 102, 105, 103, 104], "close": [101, 104, 104, 102, 106]}, idx, dtype=float)
    b = pd.DataFrame({"open": [50, 49, 48, 50, 52], "close": [50, 48, 49, 51, 50]}, idx, dtype=float)
    return idx, {"A": a, "B": b}


def test_execution_at_next_open_and_overnight_attribution():
    idx, bars = _bars()
    w = pd.DataFrame({"A": [1, 0, 0, 0, 0], "B": [0, 1, 1, 0, 0]}, idx, dtype=float)
    res = run_backtest(w, bars, cost_bps=0)
    r = res.returns
    assert r.iloc[0] == 0
    assert r.iloc[1] == pytest.approx(104 / 102 - 1)            # bought A at day-1 open
    # day 2: overnight gap belongs to A, then switch to B at the open, intraday on B
    assert r.iloc[2] == pytest.approx((105 / 104) * (49 / 48) - 1)
    assert r.iloc[3] == pytest.approx(51 / 49 - 1)              # B held all of day 3
    assert r.iloc[4] == pytest.approx(52 / 51 - 1)              # exit at open: gap 51->52 accrues, then flat
    assert res.weights.iloc[2]["B"] == pytest.approx(1) and res.weights.iloc[2]["A"] == 0


def test_costs_charged_on_traded_notional():
    idx, bars = _bars()
    w = pd.DataFrame({"A": [1, 1, 0, 0, 0]}, idx, dtype=float)
    free = run_backtest(w, bars, 0)
    paid = run_backtest(w, bars, 10)  # 10 bps
    assert paid.costs.iloc[1] == pytest.approx(0.001)
    assert paid.turnover.iloc[1] == pytest.approx(1.0) and paid.turnover.iloc[3] > 0.99
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    # two trades each ~10 bps
    assert free.equity.iloc[-1] / paid.equity.iloc[-1] == pytest.approx(1 / ((1 - 0.001) * (1 - 0.001)), rel=2e-4)


def test_rebalance_band():
    assert apply_rebalance_band({"T": 0.55}, {"T": 0.5}, 0.1) == {"T": 0.5}       # small resize skipped
    assert apply_rebalance_band({"T": 0.7}, {"T": 0.5}, 0.1) == {"T": 0.7}
    assert apply_rebalance_band({"T": 0.0}, {"T": 0.05}, 0.1) == {"T": 0.0}       # exits always
    assert apply_rebalance_band({"T": 0.05}, {"T": 0.0}, 0.1) == {"T": 0.05}      # entries always
    out = apply_rebalance_band({"T": 0.0, "S": 0.3}, {"T": 0.3, "S": 0.0}, 0.5)
    assert out == {"T": 0.0, "S": 0.3}                                           # switches always


def test_sma200_rule_uses_close_of_t_only():
    c = pd.Series(np.r_[np.full(200, 100.0), 120.0, 80.0], index=pd.bdate_range("2020-01-01", periods=202))
    w = sma200_rule(c, c.index, "TQQQ")["TQQQ"]
    assert w.iloc[199] == 0 and w.iloc[200] == 1 and w.iloc[201] == 0


def test_metrics_known_values():
    r = np.array([0.01, -0.01] * 50)
    assert sharpe(r) == pytest.approx(0.0, abs=1e-12)
    r2 = np.array([0.02, -0.01, 0.015, -0.005])
    assert sharpe(r2) == pytest.approx(r2.mean() / r2.std(ddof=1) * np.sqrt(252))
    assert sortino(r2) == pytest.approx(r2.mean() / np.sqrt(np.mean(np.minimum(r2, 0) ** 2)) * np.sqrt(252))
    eq = pd.Series([1, 1.2, 0.9, 1.3, 1.04])
    assert max_drawdown(eq) == pytest.approx(0.9 / 1.2 - 1)


def test_bootstrap_ci_and_paired_prob():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1500)
    lo, hi = bootstrap_sharpe_ci(r, n_boot=500, block=10)
    assert lo < sharpe(r) < hi
    better = r + 0.0005
    assert paired_bootstrap_prob(better, r, n_boot=300) > 0.95
    assert paired_bootstrap_prob(r, r, n_boot=300) == 0.0


def test_summary_runs_on_synthetic(bars, cfg):
    idx = bars["QQQ"].index
    res = run_backtest(sma200_rule(bars["QQQ"]["close"], idx), bars, 3.0)
    s = summarize(res)
    assert 0 <= s["Exposure"] <= 1 and s["MaxDD"] <= 0 and s["Trades"] >= 1
