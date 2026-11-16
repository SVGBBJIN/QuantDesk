"""End-to-end research: features -> candidates (rule strategies + walk-forward ML) -> benchmarks ->
metrics, bootstrap CIs, parameter neighbourhoods, objective-based selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .benchmarks import buy_and_hold
from .features import build_features, make_labels
from .metrics import bootstrap_sharpe_ci, sharpe, summarize
from .regime import classify_regimes
from .selection import FALLBACK, OBJECTIVES, evaluate, select
from .sensitivity import ml_grid, plateau
from .strategies import REGISTRY, make_inputs, perturbations, run_path, to_target_weights
from .strategy import build_signal_path
from .validation import oos_scores, select_calendar

log = logging.getLogger(__name__)
ML = "ml"
WARMUP = 260   # sessions before rule strategies have every input (SMA250 + vol windows)


@dataclass
class Research:
    window: tuple
    signals: dict[str, pd.DataFrame]
    results: dict[str, BacktestResult]
    summary: pd.DataFrame
    evaluations: dict[str, dict]
    selection: dict
    neighbourhoods: dict[str, pd.Series]
    preds: pd.DataFrame | None = None
    scores: dict = field(default_factory=dict)
    calendar: dict = field(default_factory=dict)
    sens: dict = field(default_factory=dict)
    n_folds: int = 0


def _cost(cfg) -> float:
    return cfg["costs"]["commission_bps"] + cfg["costs"]["slippage_bps"]


def _bt(sig, bars, cfg) -> BacktestResult:
    return run_backtest(to_target_weights(sig, cfg), bars, _cost(cfg), cfg["sizing"]["rebalance_band"])


def run_research(bars, cfg, sensitivity: bool = True) -> Research:
    cands = list(cfg["selection"]["candidates"])
    inp = make_inputs(bars, cfg)
    preds = feats = labels = None
    cal_info, scores = {}, {}
    if ML in cands:
        feats = build_features(bars, cfg, include_calendar=True)
        labels = make_labels(bars[cfg["data"]["signal_symbol"]]["close"], cfg["labels"]["horizon"],
                             cfg["labels"]["high_vol_threshold"])
        keep_cal, cal_info, preds = select_calendar(feats, labels, cfg)
        if not keep_cal:
            feats = feats.drop(columns=[c for c in ("dow", "days_to_month_end") if c in feats])
        scores = oos_scores(preds, labels)
        idx = preds.index
    else:
        idx = inp.idx[WARMUP:]
    if len(idx) < 252:
        raise ValueError("not enough history for an out-of-sample window of at least one year")

    signals: dict[str, pd.DataFrame] = {}
    for name in [c for c in cands if c != ML] + [FALLBACK]:
        signals[name] = run_path(name, inp, cfg, idx=idx)
    if ML in cands:
        regime = classify_regimes(feats, cfg["regime"])
        signals[ML] = build_signal_path(regime, preds, bars, cfg, "ML")
    results = {n: _bt(s, bars, cfg) for n, s in signals.items()}
    for sym in (cfg["data"]["signal_symbol"], cfg["data"]["trade_symbols"]["long"]):
        results[f"B&H {sym}"] = run_backtest(buy_and_hold(idx, sym), bars, _cost(cfg))

    v = cfg["validation"]
    rows = {}
    for name, res in results.items():
        s = summarize(res)
        s["Sharpe95_lo"], s["Sharpe95_hi"] = bootstrap_sharpe_ci(
            res.returns.iloc[1:].to_numpy(), v["bootstrap_samples"], v["bootstrap_block"], seed=cfg["model"]["seed"])
        rows[name] = s
    summary = pd.DataFrame(rows).T

    # parameter neighbourhoods (robustness evidence)
    neigh: dict[str, pd.Series] = {}
    for name in cands:
        if name == ML:
            continue
        vals = {}
        for label, p in perturbations(name, cfg):
            vals[label] = sharpe(_bt(run_path(name, inp, cfg, p, idx), bars, cfg).returns.iloc[1:].to_numpy())
        neigh[name] = pd.Series(vals, name="Sharpe")
    sens = {}
    if ML in cands and sensitivity:
        g = ml_grid(regime, preds, bars, cfg)
        sens["ml"] = (g, plateau(g))
        d = cfg["decision"]
        ri = int(np.argmin(np.abs(g.index.to_numpy(float) - d["up_enter"])))
        ci = int(np.argmin(np.abs(g.columns.to_numpy(float) - d["vol_cap"])))
        blk = g.iloc[max(0, ri - 1):ri + 2, max(0, ci - 1):ci + 2]
        neigh[ML] = pd.Series({f"up_enter={a}, vol_cap={b}": blk.loc[a, b] for a in blk.index for b in blk.columns},
                              name="Sharpe")

    bench = results[FALLBACK]
    evals = {}
    for name in cands:
        evals[name] = evaluate(name, results[name].returns, rows[name], bench.returns, rows[FALLBACK],
                               list(neigh[name].values) if name in neigh else None, cfg)
    sel = select(evals, cfg["selection"]["objective"])
    sel["by_objective"] = {o: select(evals, o)["active"] for o in OBJECTIVES}
    log.info("selection (%s): %s", sel["objective"], sel["message"])
    return Research(window=(idx[0], idx[-1]), signals=signals, results=results, summary=summary,
                    evaluations=evals, selection=sel, neighbourhoods=neigh, preds=preds, scores=scores,
                    calendar=cal_info, sens=sens,
                    n_folds=int(preds["fold"].nunique()) if preds is not None else 0)


__all__ = ["Research", "run_research", "ML", "REGISTRY"]
