"""End-to-end research: features -> regime -> walk-forward OOS probabilities -> ML & rule
strategies -> benchmarks -> metrics, bootstrap CIs, gate, sensitivity."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .benchmarks import all_benchmarks
from .features import build_features, make_labels
from .gate import evaluate_gate
from .metrics import bootstrap_sharpe_ci, summarize
from .regime import classify_regimes
from .sensitivity import ml_grid, plateau, rule_grid
from .strategy import build_signal_path, to_target_weights
from .validation import oos_scores, select_calendar

log = logging.getLogger(__name__)
ML_NAME, RULE_NAME = "ML ensemble", "Rule regime (fallback)"


@dataclass
class Research:
    window: tuple
    preds: pd.DataFrame
    sig_ml: pd.DataFrame
    sig_rule: pd.DataFrame
    results: dict[str, BacktestResult]
    summary: pd.DataFrame
    gate: dict
    scores: dict
    calendar: dict
    sens: dict = field(default_factory=dict)
    bench_name: str = ""


def run_research(bars, cfg, sensitivity: bool = True) -> Research:
    feats = build_features(bars, cfg, include_calendar=True)
    labels = make_labels(bars[cfg["data"]["signal_symbol"]]["close"], cfg["labels"]["horizon"],
                         cfg["labels"]["high_vol_threshold"])
    keep_cal, cal_info, preds = select_calendar(feats, labels, cfg)
    if not keep_cal:
        feats = feats.drop(columns=[c for c in ("dow", "days_to_month_end") if c in feats])
    regime = classify_regimes(feats, cfg["regime"])
    idx = preds.index
    cost = cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"]
    band = cfg["sizing"]["rebalance_band"]

    sig_ml = build_signal_path(regime, preds, bars, cfg, "ML")
    sig_rule = build_signal_path(regime, None, bars, cfg, "RULE", idx)
    results = {ML_NAME: run_backtest(to_target_weights(sig_ml, cfg), bars, cost, band),
               RULE_NAME: run_backtest(to_target_weights(sig_rule, cfg), bars, cost, band)}
    benches = all_benchmarks(bars, cfg, idx)
    for name, w in benches.items():
        results[name] = run_backtest(w, bars, cost, band)
    bench_name = [k for k in benches if "SMA200" in k][0]

    v = cfg["validation"]
    rows = {}
    for name, res in results.items():
        s = summarize(res)
        lo, hi = bootstrap_sharpe_ci(res.returns.iloc[1:].to_numpy(), v["bootstrap_samples"], v["bootstrap_block"],
                                     seed=cfg["model"]["seed"])
        s["Sharpe95_lo"], s["Sharpe95_hi"] = lo, hi
        rows[name] = s
    summary = pd.DataFrame(rows).T
    gate = evaluate_gate(results[ML_NAME].returns, results[bench_name].returns, cfg)
    r = Research(window=(idx[0], idx[-1]), preds=preds, sig_ml=sig_ml, sig_rule=sig_rule, results=results,
                 summary=summary, gate=gate, scores=oos_scores(preds, labels), calendar=cal_info,
                 bench_name=bench_name)
    if sensitivity:
        g_ml = ml_grid(regime, preds, bars, cfg)
        g_rule = rule_grid(feats, bars, cfg, idx)
        r.sens = {"ml": (g_ml, plateau(g_ml)), "rule": (g_rule, plateau(g_rule))}
    return r
