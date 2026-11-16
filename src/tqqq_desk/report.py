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
from .research import Research, run_research
from .selection import FALLBACK, LABELS, OBJECTIVES
from .strategies import REGISTRY

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


def _ok(x: bool) -> str:
    return "✅" if x else "❌"


def write_report(r: Research, cfg, out_dir: Path, source: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    sel = r.selection
    cols = ["CAGR", "AnnVol", "Sharpe", "Sharpe95_lo", "Sharpe95_hi", "Sortino", "MaxDD", "Calmar",
            "Turnover/yr", "Exposure", "AvgGross", "HitRate(days)", "Trades", "WinRate(trades)", "CostDrag/yr"]
    lines = ["# tqqq-desk backtest report", ""]
    if source != "ALPACA":
        lines += [f"> **DATA SOURCE: {source}. These numbers are research evidence only; the live desk can only "
                  "be unlocked by a validation on Alpaca data.**", ""]
    lines += [
        f"- Generated: {now:%Y-%m-%d %H:%M UTC}  ·  data source: **{source}**  ·  config hash `{config_hash(cfg)}`",
        f"- Out-of-sample window: **{r.window[0].date()} → {r.window[1].date()}** ({len(r.signals[FALLBACK])} sessions"
        + (f"; ML: {r.n_folds} walk-forward folds, retrain every {cfg['validation']['test_block']} sessions, purge "
           f"{cfg['labels']['horizon']} + embargo {cfg['validation']['embargo']}" if r.n_folds else "") + ")",
        f"- Costs: {cfg['costs']['commission_bps']} bps commission + {cfg['costs']['slippage_bps']} bps slippage per side; "
        "execution at next session's open; leveraged ETF returns from actual TQQQ/SQQQ prices; cash earns 0.",
        "",
        f"## Active strategy: `{sel['active']}`  (objective: {LABELS[sel['objective']]})",
        "", sel["message"], "",
        "| objective | would trade |", "|---|---|",
    ]
    lines += [f"| {LABELS[o]}{' **(configured)**' if o == sel['objective'] else ''} | `{a}` |"
              for o, a in sel["by_objective"].items()] + [""]
    lines += ["## Candidates vs objectives", "",
              "| candidate | designed for | " + " | ".join(LABELS[o] for o in OBJECTIVES) + " |",
              "|---|---|" + "---|" * len(OBJECTIVES)]
    for n, e in r.evaluations.items():
        spec = REGISTRY.get(n)
        design = LABELS.get(spec.objective, spec.objective) if spec else "ML ensemble"
        cells = []
        for o in OBJECTIVES:
            sc = e[o]["score"]
            sc_txt = "n/a" if not np.isfinite(sc) else (f"{100 * sc:.1f}%" if o == "max_return" else f"{sc:.2f}")
            cells.append(f"{_ok(e[o]['eligible'])} {sc_txt}")
        lines.append(f"| `{n}` | {design} | " + " | ".join(cells) + " |")
    lines += ["", "Scores: Sharpe (risk-adjusted), CAGR (max return), worst Sharpe across the 3×3 parameter "
              "neighbourhood (robust).", ""]
    for n, e in r.evaluations.items():
        lines += [f"<details><summary><code>{n}</code> checks</summary>", ""]
        for o in OBJECTIVES:
            lines.append(f"- **{LABELS[o]}**: " + "; ".join(f"{_ok(ok)} {k}" for k, ok in e[o]["checks"].items()))
        lines += ["", "</details>", ""]
    lines += ["## Out-of-sample performance (after costs)", "",
              md_table(r.summary[cols], _metric_fmt, "Strategy"), "",
              "Sharpe95_lo/hi: 95% stationary block-bootstrap interval "
              f"(mean block {cfg['validation']['bootstrap_block']}d, {cfg['validation']['bootstrap_samples']} resamples). "
              "Hit rate = share of invested days with a positive return. `sma200` = TQQQ if QQQ > SMA200 else cash.", ""]
    if r.neighbourhoods:
        lines += ["## Parameter neighbourhoods (OOS Sharpe)", "",
                  "Each strategy re-run on a 3×3 grid around its configured parameters. A strategy whose edge "
                  "only exists at one exact setting is curve-fit.", ""]
        for n, ser in r.neighbourhoods.items():
            lines += [f"**`{n}`**: min {ser.min():.2f} · median {ser.median():.2f} · max {ser.max():.2f}", "",
                      "| params | Sharpe |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in ser.items()] + [""]
    sc = r.scores
    if sc:
        lines += ["## ML model quality (OOS, calibrated ensemble)", "",
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
    if "ml" in r.sens:
        grid, pl = r.sens["ml"]
        lines += ["## Sensitivity: ML decision thresholds (OOS Sharpe after costs)", "",
                  md_table(grid, lambda c, v: f"{v:.2f}", "up_enter \\ vol_cap"), "",
                  f"- Configured: up_enter={cfg['decision']['up_enter']}, vol_cap={cfg['decision']['vol_cap']}",
                  f"- Most stable plateau centre: up_enter={pl['plateau_center']['up_enter']}, "
                  f"vol_cap={pl['plateau_center']['vol_cap']} (neighbourhood mean "
                  f"{pl['plateau_center']['neighbourhood_mean']:.2f}). Not auto-applied.", ""]
    lines += ["## Honest caveats", "",
              "- Rule strategies have no fitted parameters, but their DESIGN was chosen by looking at history "
              "(including a 1999-2018 Nasdaq proxy). Their 'out-of-sample' numbers are partly in-sample.",
              f"- Picking the best of {len(r.evaluations)} candidates inflates the winner's numbers; the bootstrap "
              "threshold and neighbourhood checks reduce, but do not remove, that selection bias.",
              "- One historical path with few independent bear markets; confidence intervals are wide.",
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
    if r.preds is not None:
        r.preds.to_csv(art / "oos_predictions.csv")
    for n, sig in r.signals.items():
        sig.to_csv(art / f"oos_signals_{n}.csv")
    for key, (grid, _) in r.sens.items():
        grid.to_csv(art / f"sensitivity_{key}.csv")
    sel = r.selection
    active = sel["active"]
    status = {
        "active_strategy": active, "objective": sel["objective"], "selection_passed": sel["passed"],
        "eligible": sel["eligible"], "by_objective": sel["by_objective"], "message": sel["message"],
        # backwards-compatible fields (dashboard / status page / older runners)
        "mode": "ML" if active == "ml" else "RULE", "gate_passed": sel["passed"],
        "validated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_source": source, "oos_start": str(r.window[0].date()), "oos_end": str(r.window[1].date()),
        "calendar_features": bool(r.calendar.get("kept", r.calendar.get("mode") == "on")),
        "config_hash": config_hash(cfg),
        "evaluations": {n: {o: {"eligible": e[o]["eligible"], "score": None if not np.isfinite(e[o]["score"])
                                else float(e[o]["score"]), "checks": e[o]["checks"]} for o in OBJECTIVES}
                        for n, e in r.evaluations.items()},
        "summary": {n: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                    for n, row in r.summary[["CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar"]].iterrows()},
    }
    if source != "ALPACA":
        status["active_strategy"], status["mode"] = "sma200", "RULE"   # research data can never unlock trading
        status["message"] = (f"Validation ran on {source} data; it cannot enable a strategy (research says "
                             f"'{active}'). " + status["message"])
    (art / "model_status.json").write_text(json.dumps(status, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest + ML gate + sensitivity report")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (plumbing check only)")
    ap.add_argument("--no-refresh", action="store_true", help="use cached Alpaca bars")
    ap.add_argument("--csv-dir", default=None,
                    help="load <SYMBOL>.csv daily bars from this directory (research only, e.g. the 1999-2018 "
                         "proxy from scripts/make_proxy_history.py)")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--artifacts", default="artifacts")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    if a.csv_dir:
        from .data import load_csv_dir
        bars, source = load_csv_dir(a.csv_dir, cfg), f"CSV:{Path(a.csv_dir).name}"
    elif a.synthetic:
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
