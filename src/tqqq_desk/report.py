"""Backtest report: markdown + CSV artifacts + model_status.json (consumed by the live runner)."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import drawdown_series
from .research import ML_NAME, RULE_NAME, Research, run_research

log = logging.getLogger(__name__)
PCT = {"CAGR", "AnnVol", "MaxDD", "Exposure", "HitRate(days)", "WinRate(trades)", "CostDrag/yr"}


def md_table(df: pd.DataFrame, fmt=None, index_name: str | None = None) -> str:
    fmt = fmt or (lambda c, v: f"{v:.2f}")
    cols = [index_name or (df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, row in df.iterrows():
        cells = [str(i)] + [("" if pd.isna(v) else fmt(c, v)) if isinstance(v, (int, float, np.floating)) else str(v)
                            for c, v in row.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _metric_fmt(c, v):
    if c in PCT:
        return f"{100 * v:.1f}%"
    if c == "Trades":
        return f"{int(v)}"
    return f"{v:.2f}"


def config_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def write_report(r: Research, cfg, out_dir: Path, source: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    g = r.gate
    cols = ["CAGR", "AnnVol", "Sharpe", "Sharpe95_lo", "Sharpe95_hi", "Sortino", "MaxDD", "Calmar",
            "Turnover/yr", "Exposure", "AvgGross", "HitRate(days)", "Trades", "WinRate(trades)", "CostDrag/yr"]
    lines = [f"# tqqq-desk backtest report", ""]
    if source != "ALPACA":
        lines += [f"> **DATA SOURCE: {source}. These numbers are a plumbing check, NOT evidence of edge.**", ""]
    lines += [
        f"- Generated: {now:%Y-%m-%d %H:%M UTC}  ·  data source: **{source}**  ·  config hash `{config_hash(cfg)}`",
        f"- Out-of-sample window: **{r.window[0].date()} → {r.window[1].date()}** "
        f"({len(r.preds)} sessions, {r.preds['fold'].nunique()} walk-forward folds, retrain every "
        f"{cfg['validation']['test_block']} sessions, purge {cfg['labels']['horizon']} + embargo {cfg['validation']['embargo']})",
        f"- Costs: {cfg['costs']['commission_bps']} bps commission + {cfg['costs']['slippage_bps']} bps slippage per side; "
        f"execution at next session's open; leveraged ETF returns from actual TQQQ/SQQQ prices; cash earns 0.",
        "",
        f"## Gate verdict: {'✅ ML ENSEMBLE ACTIVE' if g['passed'] else '⛔ ML FAILED — RULE-BASED FALLBACK ACTIVE'}",
        "", g["message"], "",
    ]
    lines += [f"- {'✅' if ok else '❌'} {k}" for k, ok in g["checks"].items()] + [""]
    lines += ["## Out-of-sample performance (after costs)", "",
              md_table(r.summary[cols], _metric_fmt, "Strategy"), "",
              "Sharpe95_lo/hi: 95% stationary block-bootstrap interval "
              f"(mean block {cfg['validation']['bootstrap_block']}d, {cfg['validation']['bootstrap_samples']} resamples). "
              "Hit rate = share of invested days with a positive return.", ""]
    sc = r.scores
    if sc:
        lines += ["## Model quality (OOS, calibrated ensemble)", "",
                  "| target | AUC | log-loss | base-rate log-loss | Brier |", "|---|---|---|---|---|"]
        for p, name in (("p_up", "next-5d up"), ("p_hivol", "next-5d high vol")):
            if f"{p}_auc" in sc:
                lines.append(f"| {name} | {sc[p + '_auc']:.3f} | {sc[p + '_logloss']:.4f} | "
                             f"{sc[p + '_logloss_base']:.4f} | {sc[p + '_brier']:.4f} |")
        lines += ["", "A log-loss above the base-rate log-loss means the model is worse than always predicting "
                      "the historical frequency.", ""]
    cal = r.calendar
    if cal.get("mode") == "auto":
        lines += [f"Calendar features (day-of-week, days-to-month-end): **{'kept' if cal['kept'] else 'dropped'}** "
                  f"(mean OOS log-loss with {cal['logloss_with']:.4f} vs without {cal['logloss_without']:.4f}).", ""]
    for key, title, params in (("ml", "ML decision thresholds", ("up_enter", "vol_cap")),
                               ("rule", "Rule-regime thresholds", ("bull_enter", "hv_enter"))):
        if key not in r.sens:
            continue
        grid, pl = r.sens[key]
        cur = (cfg["decision"] if key == "ml" else cfg["regime"])
        lines += [f"## Sensitivity: {title} (OOS Sharpe after costs)", "",
                  md_table(grid, lambda c, v: f"{v:.2f}", f"{params[0]} \\ {params[1]}"), "",
                  f"- Configured: {params[0]}={cur[params[0]]}, {params[1]}={cur[params[1]]}",
                  f"- Single best cell: {params[0]}={pl['best_point'][params[0]]}, {params[1]}={pl['best_point'][params[1]]} "
                  f"(Sharpe {pl['best_point']['sharpe']:.2f})",
                  f"- **Most stable plateau centre** (best 3×3 neighbourhood mean): {params[0]}={pl['plateau_center'][params[0]]}, "
                  f"{params[1]}={pl['plateau_center'][params[1]]} (cell {pl['plateau_center']['sharpe']:.2f}, "
                  f"neighbourhood mean {pl['plateau_center']['neighbourhood_mean']:.2f}, "
                  f"min {pl['plateau_center']['neighbourhood_min']:.2f})",
                  "- Not auto-applied: choosing thresholds on this OOS window would make it in-sample. "
                  + ("Per the gate rule, ML thresholds must not be tuned while the gate is failing." if key == "ml" and not g["passed"] else ""),
                  ""]
    lines += ["## Honest caveats", "",
              "- One historical path (~10 years of Alpaca data) with few independent bear markets; CIs are wide.",
              "- Sensitivity grids and the calendar-feature choice look at the OOS period, so they carry mild selection bias.",
              "- Market-on-open fills assumed at the official open price; real auction fills differ slightly.",
              "- Uninvested cash earns 0 here (real accounts earn T-bill-ish interest), which understates cash-heavy strategies.",
              ""]
    path = out_dir / "backtest_report.md"
    path.write_text("\n".join(lines))
    return path


def write_artifacts(r: Research, cfg, art: Path, source: str) -> None:
    art.mkdir(parents=True, exist_ok=True)
    eq = pd.DataFrame({k: v.equity for k, v in r.results.items()})
    eq.index.name = "date"
    eq.to_csv(art / "backtest_equity.csv")
    pd.DataFrame({k: drawdown_series(v.equity) for k, v in r.results.items()}).to_csv(art / "backtest_drawdown.csv")
    r.summary.to_csv(art / "backtest_summary.csv")
    r.preds.to_csv(art / "oos_predictions.csv")
    r.sig_ml.to_csv(art / "oos_signals_ml.csv")
    for key, (grid, _) in r.sens.items():
        grid.to_csv(art / f"sensitivity_{key}.csv")
    status = {
        "mode": r.gate["mode"], "gate_passed": r.gate["passed"], "message": r.gate["message"],
        "checks": r.gate["checks"], "validated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_source": source, "oos_start": str(r.window[0].date()), "oos_end": str(r.window[1].date()),
        "calendar_features": bool(r.calendar.get("kept", r.calendar.get("mode") == "on")),
        "config_hash": config_hash(cfg),
        "sharpe_ml": r.gate["sharpe_ml"], "sharpe_bench": r.gate["sharpe_bench"],
        "bootstrap_prob": r.gate["bootstrap_prob"],
    }
    if source != "ALPACA":
        status["mode"] = "RULE"  # never let a synthetic validation unlock ML trading
        status["message"] = "Validation ran on non-Alpaca data; ML cannot be enabled from it. " + status["message"]
    (art / "model_status.json").write_text(json.dumps(status, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest + ML gate + sensitivity report")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (plumbing check only)")
    ap.add_argument("--no-refresh", action="store_true", help="use cached Alpaca bars")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--artifacts", default="artifacts")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    if a.synthetic:
        from .data import synthetic_bars
        bars, source = synthetic_bars(2700, seed=11), "SYNTHETIC"
    else:
        from .data import load_bars
        bars, source = load_bars(cfg, refresh=not a.no_refresh), "ALPACA"
    r = run_research(bars, cfg, sensitivity=not a.no_sensitivity)
    p = write_report(r, cfg, Path(a.out), source)
    write_artifacts(r, cfg, Path(a.artifacts), source)
    print(p.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
