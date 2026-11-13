"""ML promotion gate. The ML ensemble trades only if, out-of-sample and after costs, it beats
the TQQQ/SMA200 benchmark on risk-adjusted terms. Otherwise the rule-based regime strategy runs."""
from __future__ import annotations

from .metrics import paired_bootstrap_prob, sharpe, sortino


def evaluate_gate(ml_ret, bench_ret, cfg) -> dict:
    g, v = cfg["gate"], cfg["validation"]
    a, b = ml_ret.iloc[1:].to_numpy(), bench_ret.iloc[1:].to_numpy()
    s_ml, s_b = sharpe(a), sharpe(b)
    so_ml, so_b = sortino(a), sortino(b)
    prob = paired_bootstrap_prob(a, b, v["bootstrap_samples"], v["bootstrap_block"], seed=cfg["model"]["seed"])
    checks = {
        f"Sharpe ML {s_ml:.2f} > SMA200 {s_b:.2f} + {g['min_sharpe_diff']:.2f}": s_ml > s_b + g["min_sharpe_diff"],
        f"P(Sharpe ML > SMA200) {prob:.2f} >= {g['min_bootstrap_prob']:.2f}": prob >= g["min_bootstrap_prob"],
    }
    if g["require_sortino_better"]:
        checks[f"Sortino ML {so_ml:.2f} > SMA200 {so_b:.2f}"] = so_ml > so_b
    passed = all(checks.values())
    return {
        "passed": passed,
        "mode": "ML" if passed else "RULE",
        "checks": {k: bool(x) for k, x in checks.items()},
        "sharpe_ml": s_ml, "sharpe_bench": s_b, "sortino_ml": so_ml, "sortino_bench": so_b,
        "bootstrap_prob": prob,
        "message": ("ML ensemble PASSED the gate: it beats the SMA200 benchmark out-of-sample after costs."
                    if passed else
                    "ML ensemble FAILED the gate: it does not beat the SMA200 benchmark out-of-sample after "
                    "costs on risk-adjusted terms. The system is running the RULE-BASED regime strategy."),
    }
