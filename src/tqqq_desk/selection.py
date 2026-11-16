"""Strategy selection by objective. Replaces the old single ML-vs-SMA200 gate.

Every candidate is scored against the benchmark ("TQQQ if QQQ > SMA200", strategy `sma200`) under
all three objectives, so the report always shows the whole picture. The configured objective decides
which candidate trades:

  risk_adjusted  Sharpe > benchmark (+ min diff), Sortino > benchmark,
                 paired block bootstrap P(Sharpe > benchmark) >= min_bootstrap_prob.   score = Sharpe
  max_return     max drawdown within max_drawdown_cap, CAGR > benchmark,
                 bootstrap P(mean log-growth > benchmark) >= min_bootstrap_prob.        score = CAGR
  robust         share of calendar years with positive excess return >= min_year_win_rate,
                 bootstrap P(Sharpe > benchmark) >= min_bootstrap_prob, and the WORST Sharpe in the
                 3x3 parameter neighbourhood still beats the benchmark.                 score = worst
                 neighbourhood Sharpe

The eligible candidate with the highest score trades. If none is eligible, the benchmark itself
(`sma200`) trades: no validated edge means no extra complexity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TD, sharpe, sortino, stationary_bootstrap_indices

OBJECTIVES = ("risk_adjusted", "max_return", "robust")
FALLBACK = "sma200"
LABELS = {"risk_adjusted": "risk-adjusted (Sharpe/Sortino)", "max_return": "max return under a drawdown cap",
          "robust": "robustly beat SMA200"}


def _sharpe_rows(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=1, ddof=1)
    return np.where(sd > 0, m.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(TD), 0.0)


def bootstrap_probs(a: np.ndarray, b: np.ndarray, n_boot: int, block: int, seed: int) -> dict[str, float]:
    """Paired stationary bootstrap (same resampled days for both series)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = stationary_bootstrap_indices(len(a), n_boot, block, np.random.default_rng(seed))
    ra, rb = a[idx], b[idx]
    la, lb = np.log1p(np.clip(ra, -0.999, None)), np.log1p(np.clip(rb, -0.999, None))
    return {"sharpe": float(np.mean(_sharpe_rows(ra) > _sharpe_rows(rb))),
            "growth": float(np.mean(la.mean(axis=1) > lb.mean(axis=1)))}


def year_win_rate(r: pd.Series, b: pd.Series) -> float:
    ya = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    yb = b.groupby(b.index.year).apply(lambda x: (1 + x).prod() - 1)
    return float((ya - yb > 0).mean()) if len(ya) else float("nan")


def evaluate(name: str, ret: pd.Series, summ: dict, bench_ret: pd.Series, bench_summ: dict,
             neighbourhood: list[float] | None, cfg) -> dict:
    s, v = cfg["selection"], cfg["validation"]
    a, b = ret.iloc[1:], bench_ret.iloc[1:]
    pb = bootstrap_probs(a.to_numpy(), b.to_numpy(), v["bootstrap_samples"], v["bootstrap_block"],
                         cfg["model"]["seed"])
    sr, so = sharpe(a.to_numpy()), sortino(a.to_numpy())
    sr_b, so_b = sharpe(b.to_numpy()), sortino(b.to_numpy())
    p_min = s["min_bootstrap_prob"]
    ywr = year_win_rate(a, b)
    nb_min = float(np.min(neighbourhood)) if neighbourhood else float("nan")
    out = {}
    checks = {
        f"Sharpe {sr:.2f} > SMA200 {sr_b:.2f} + {s['min_sharpe_diff']:.2f}": sr > sr_b + s["min_sharpe_diff"],
        f"Sortino {so:.2f} > SMA200 {so_b:.2f}": so > so_b,
        f"P(Sharpe > SMA200) {pb['sharpe']:.2f} >= {p_min:.2f}": pb["sharpe"] >= p_min,
    }
    out["risk_adjusted"] = {"eligible": all(checks.values()), "score": sr, "checks": checks}
    cap = s["max_drawdown_cap"]
    checks = {
        f"MaxDD {summ['MaxDD']:.0%} within cap -{cap:.0%}": summ["MaxDD"] >= -cap,
        f"CAGR {summ['CAGR']:.1%} > SMA200 {bench_summ['CAGR']:.1%}": summ["CAGR"] > bench_summ["CAGR"],
        f"P(growth > SMA200) {pb['growth']:.2f} >= {p_min:.2f}": pb["growth"] >= p_min,
    }
    out["max_return"] = {"eligible": all(checks.values()), "score": summ["CAGR"], "checks": checks}
    checks = {
        f"beats SMA200 in {ywr:.0%} of years >= {s['min_year_win_rate']:.0%}": ywr >= s["min_year_win_rate"],
        f"P(Sharpe > SMA200) {pb['sharpe']:.2f} >= {p_min:.2f}": pb["sharpe"] >= p_min,
        (f"worst neighbourhood Sharpe {nb_min:.2f} > SMA200 {sr_b:.2f}" if neighbourhood else
         "parameter neighbourhood not evaluated"): bool(neighbourhood) and nb_min > sr_b,
    }
    out["robust"] = {"eligible": all(checks.values()), "score": nb_min if neighbourhood else float("nan"),
                     "checks": checks}
    for o in out.values():
        o["checks"] = {k: bool(x) for k, x in o["checks"].items()}
        o["eligible"] = bool(o["eligible"])
    out["_stats"] = {"bootstrap_sharpe": pb["sharpe"], "bootstrap_growth": pb["growth"], "year_win_rate": ywr,
                     "neighbourhood_min_sharpe": nb_min}
    return out


def select(evaluations: dict[str, dict], objective: str) -> dict:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective}")
    elig = {n: e[objective]["score"] for n, e in evaluations.items() if e[objective]["eligible"]}
    if not elig:
        return {"active": FALLBACK, "objective": objective, "passed": False, "eligible": [],
                "message": (f"No candidate passed the '{LABELS[objective]}' checks out-of-sample after costs. "
                            "The desk trades the plain benchmark: TQQQ when QQQ > SMA200, else cash.")}
    best = max(elig, key=elig.get)
    return {"active": best, "objective": objective, "passed": True, "eligible": sorted(elig),
            "message": (f"'{best}' is active: it passed the '{LABELS[objective]}' checks out-of-sample after "
                        f"costs and scored best among eligible candidates ({', '.join(sorted(elig))}).")}
