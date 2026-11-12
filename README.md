# tqqq-desk

A systematic **daily-swing** trading system that holds **TQQQ** (bullish), **SQQQ** (bearish) or **cash**,
executed through the Alpaca API on a GitHub Actions schedule, with a Streamlit dashboard.
**Paper trading is the default.**

> ⚠️ TQQQ and SQQQ are 3× daily-reset leveraged ETFs. They can lose most of their value in weeks,
> and SQQQ decays over time even in sideways markets. Nothing here is investment advice. Run it on
> paper for months before you even consider real money.

---

## How it decides (layered, not one opaque model)

```
 Alpaca daily bars (QQQ, TQQQ, SQQQ, SPY, TLT, UUP, VIXY; split+dividend adjusted)
        │
        ▼
 1. Features (causal: row t uses only bars ≤ close of t)
    trend · volatility · drawdown · cross-asset · calendar (kept only if it helps OOS)
        │
        ├──────────────► 2. Regime (rule-based, interpretable)
        │                   BULL / NEUTRAL / BEAR / HIGH_VOL, with hysteresis + min hold
        ▼
 3. Predictive layer: LightGBM (shallow, regularised) + logistic baseline,
    each calibrated (Platt/isotonic) on a purged holdout, probability-averaged:
      P(next-5d QQQ return > 0),  P(next-5d realised vol > 30%)
        │
        ▼
 4. Decision state machine  ─►  sizing = min(vol target, ¼-Kelly, per-ETF cap)
    TQQQ: regime BULL/NEUTRAL & P(up) ≥ 0.55 (hold while ≥ 0.50) & P(high vol) < 0.50
    SQQQ: regime BEAR & P(down) ≥ 0.60, max 10 sessions, then 5-day cooldown
    else CASH. A TQQQ↔SQQQ flip always spends one session in cash.
        │
        ▼
 5. ML gate (weekly): if the ML ensemble does not beat "TQQQ when QQQ > SMA200" out-of-sample,
    after costs, on risk-adjusted terms → the system runs the RULE-BASED regime strategy instead
    (TQQQ in BULL, sized by vol targeting) and says so in the report and dashboard.
```

Code map (`src/tqqq_desk/`): `data.py` · `features.py` · `regime.py` · `models.py` · `validation.py`
· `decision.py` · `sizing.py` · `strategy.py` · `backtest.py` · `benchmarks.py` · `metrics.py`
· `gate.py` · `sensitivity.py` · `research.py` · `report.py` · `risk.py` · `execution.py` · `broker.py`
· `live_guard.py` · `state.py` · `mktcal.py` · `run_daily.py`. Dashboard: `dashboard/app.py`.

### Validation (what "out-of-sample" means here)
* Expanding-window walk-forward, retrained every 63 sessions, starting after ~2 years of history.
* **Purging**: a training row is used only if its 5-day label window ends before the test block,
  plus an **embargo** of 5 more sessions. Calibration inside each training window is purged the same way.
  No random k-fold anywhere.
* Backtest executes at the **next session's open** at the actual TQQQ/SQQQ open price. The overnight
  gap goes to the old position and the intraday move to the new one. Costs are 2 bps + 1 bp slippage
  per side. Leveraged ETF returns come from **their own price history**, so the decay is real.
* Benchmarks: buy & hold QQQ, buy & hold TQQQ, and TQQQ when QQQ > 200-day SMA else cash.
* Reported: CAGR, vol, Sharpe (with 95% stationary block-bootstrap CI), Sortino, max drawdown,
  Calmar, turnover, exposure, hit rate, trade win rate, cost drag.
* **Gate**: ML Sharpe > SMA200 Sharpe, ML Sortino > SMA200 Sortino, **and** paired-bootstrap
  P(Sharpe_ML > Sharpe_SMA200) ≥ 0.60. Any failure means rule-based fallback.
* Sensitivity grids for the ML thresholds (`up_enter × vol_cap`) and the rule thresholds
  (`bull_enter × hv_enter`). The report shows the single best cell **and** the centre of the most
  stable plateau (best full 3×3 neighbourhood). Nothing is auto-applied. While the gate fails,
  the ML thresholds are not to be tuned.
* Sanity tests in `tests/`: a **leakage canary** (on a driftless random walk the OOS AUC must be
  about 0.5) and a **positive control** (a real signal must be found and calibrated).

---

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m pytest                      # ~80 tests, ~10 s, no network
```

### Secrets: never in the repo, never in logs
| Where | Name | Value |
|---|---|---|
| GitHub → Settings → Secrets and variables → Actions → **Secrets** | `ALPACA_KEY` | Alpaca API key ID |
| | `ALPACA_SECRET` | Alpaca secret key |
| GitHub → … → **Variables** (optional) | `ALPACA_LIVE` | `false` (default) or `true` |
| | `I_UNDERSTAND_THE_RISK` | unset, or `true` (see *Going live*) |

Locally, export `ALPACA_KEY` / `ALPACA_SECRET` in your shell. `.env` is git-ignored. Any log line that
contains a secret's value is redacted (`logutil.py`, tested).

Repository settings: Settings → Actions → General → Workflow permissions must allow **read and write**
so the bot can push to the `data` branch. If you protect branches, exempt `data` or allow
`github-actions[bot]` to push to it.

### First run (paper)
1. Add the two secrets.
2. **Actions → validate → Run workflow.** This pulls the full Alpaca history, runs the walk-forward
   backtest, the gate and the sensitivity grids, and writes `artifacts/` and `reports/backtest_report.md`
   to the `data` branch. The job summary shows the report.
3. **Actions → daily-trade → Run workflow** with *dry_run* ticked (the default). Read the log.
4. If your paper account already holds TQQQ/SQQQ, run once with *adopt_broker_positions*, or the
   position reconciliation will (correctly) halt.
5. From then on the schedule runs by itself every trading day after the close.

Local equivalents:
```bash
python scripts/backtest_report.py                         # needs ALPACA_KEY/SECRET
python -m tqqq_desk.run_daily --state-dir state --dry-run --force
python scripts/backtest_report.py --synthetic             # no keys: plumbing check, NOT evidence
python -m tqqq_desk.run_daily --fake-broker --dry-run --force --state-dir /tmp/st --now 2024-07-05T19:30:00-04:00
```

---

## Execution: order type, timing, idempotency

**Choice: market-on-open (Alpaca `time_in_force=OPG`), whole shares, submitted in the evening.**
* The signal uses the final daily bar, so the run happens after the close. Alpaca only accepts OPG
  orders from about **19:00 ET until 09:28 ET** the next session, so the scheduled run is ~19:37 ET.
* OPG orders fill in the opening auction, which is exactly the price the backtest assumes. A pre-market
  DAY order would add a second scheduled job and fill at a less well-defined price.
* Cost of this choice: OPG cannot be fractional, so positions are rounded down to whole shares.
  A buy that the buying-power check (with a 5% gap buffer) cannot cover is reduced.

**Scheduling with DST.** Actions cron is UTC and ignores DST, so the workflow has three crons
(23:37, 00:37, 02:37 UTC). The script exits early unless (a) today is an NYSE session per
`pandas_market_calendars` (holidays skipped; half-days handled because the run is hours after any close),
(b) it is after 19:00 ET, and (c) that session hasn't already been processed. The extra firings are
therefore harmless no-ops, and the third one is a backup in case Actions delays the earlier ones.

**Idempotency and dedupe.**
* `state.json.last_signal_date` blocks reprocessing a session.
* If the target equals the current position, or the change is inside the 10% rebalance band, no order is placed.
* `client_order_id = tqd-<YYYYMMDD>-<SYMBOL>-<side>-<qty>` is deterministic. The broker is checked
  for that id before submitting. Transient errors are retried and re-checked by id, and a duplicate-id
  rejection is treated as "already submitted". A lost HTTP response can therefore never double-trade.
* A `concurrency: tqqq-desk-state` group prevents two runs (daily or validate) from overlapping.

**State** lives on the orphan **`data` branch** (`state.json`, `trades.csv`, `signals.csv`,
`equity.csv`, `artifacts/`, `reports/`), committed by the bot so `main` history stays clean.

## Risk controls
| Control | Behaviour |
|---|---|
| Daily loss > 8% (account equity vs prior close) | **Latched** kill switch. Default action `flatten` (MOO sells to cash), then no trading |
| Drawdown > 30% from peak equity | same |
| Broker positions ≠ expected, unexpected open orders, both ETFs held, trading blocked | **Latched**, `freeze` (no orders: the book is unknown, so a human must look) |
| Stale or missing data (last bar ≠ last completed session) | Halt **this run only**. A later cron the same evening may retry |
| Post-trade check (orders present as planned, positions unchanged) fails | **Latched**, `freeze` |
| Never hold TQQQ and SQQQ together | enforced in the state machine, the order planner and reconciliation |

The daily runner exits with code 3 when halted, so the workflow run shows red. State is still committed.
To resume after investigating, run **daily-trade** with *reset_kill_switch*. Thresholds are in `config.yaml → risk`.

Model safety: the daily runner uses the ML ensemble only if `artifacts/model_status.json` says the
gate passed, the validation ran on **Alpaca** data, it is under 14 days old, **and** the
`config.yaml` hash matches. Otherwise it uses the rule-based strategy and says why.

## Going live (real money)
Live trading requires **both** of these, and refuses to start otherwise:
1. `ALPACA_LIVE=true` (repo variable or env), **and**
2. the CLI flag `--i-understand-the-risk` (in Actions: repo variable `I_UNDERSTAND_THE_RISK=true`).

The runner also checks that the account number and endpoint match the mode: paper accounts are `PA…`
on `paper-api.alpaca.markets`. Use live API keys in the secrets. Before going live:
- [ ] At least a few months of paper running with no reconciliation halts
- [ ] Read the latest `backtest_report.md` and note whether ML or the fallback is active, and why
- [ ] Decide the kill-switch thresholds you can actually live with
- [ ] Start with a small account. Margin/PDT rules apply to accounts under $25k

## Dashboard
```bash
git fetch origin data && git worktree add state origin/data   # once
git -C state pull                                             # to refresh
streamlit run dashboard/app.py                                # or TQQQ_STATE_DIR=/path ...
```
It shows a PAPER/LIVE banner, kill-switch state, whether ML or the rule fallback is active (and why),
today's regime and probabilities with a plain-language reason, current position and target, data and
model freshness, live equity and drawdown, OOS backtest equity vs benchmarks, drawdowns, recent trades and signals.

## Configuration
Everything tunable is in `config.yaml` (thresholds, horizons, model regularisation, sizing, costs, gate,
risk limits, sensitivity ranges). `config.py` validates it. For example, Kelly fraction > 0.25,
exit thresholds above entry thresholds, or an SQQQ entry bar below the TQQQ bar are all rejected.

## Known limitations (read these)
* **Short history.** Alpaca daily data starts around 2016, so the OOS window is roughly 2018–today. It
  contains only a handful of real bear phases (2018 Q4, 2020, 2022, 2025). Bootstrap CIs on Sharpe
  are wide, so treat point estimates with suspicion.
* **Selection bias.** The calendar-feature keep/drop decision and the sensitivity grids both look at
  the OOS period. The grids are diagnostic, and the gate is the only automatic decision.
* **Live vs backtest model cadence.** The backtest retrains every 63 sessions. The live runner retrains on
  all labelled history each evening. They use the same model class and settings but not identical refit dates.
* **Fills.** The backtest uses the official open with flat costs. Real auction fills, large size and
  OPG rejections are not modelled. Cash earns 0 in the backtest.
* **Adjusted prices.** Split- and dividend-adjusted history is rewritten by the vendor when new
  dividends occur. This is not lookahead (it is applied at the ex-date), but it means old feature
  values can shift slightly between data pulls.
* **Kelly with an estimated edge.** Calibrated probabilities near 0.5 produce small sizes by design.
  A badly calibrated model would mis-size, which is why the cap is quarter-Kelly.
* **Single broker, single account, no intraday risk management.** The kill switch acts at the next
  evening run, so a crash during the session is not stopped intraday. Consider broker-side stop orders
  if that matters to you.
* **Cash flows.** Deposits and withdrawals move account equity and therefore the daily-loss and
  peak-drawdown checks. After a withdrawal, reset `peak_equity` in `state.json` on the `data` branch.
  A large withdrawal can otherwise trip the kill switch.
* GitHub Actions schedules can be delayed or (rarely) skipped. A skipped evening means no trade that day.

## Tests
`python -m pytest` covers:
- no-lookahead: truncation invariance, future-perturbation invariance, label alignment, leakage canary
- purged/embargoed splits
- the regime hysteresis and min-hold
- the position state machine (SQQQ cap and cooldown, flip via cash, never both, randomised invariants)
- sizing math (vol target, ¼-Kelly cap, per-ETF caps)
- the backtest engine (next-open execution, overnight attribution, costs, band)
- metrics and bootstrap
- the gate and plateau selection
- the kill switch
- the live-mode guard (including the CLI refusing to start) and secret redaction
- order planning and dedupe/retry
- full daily-run integration against a fake broker: holidays, DST double-cron, reconciliation halts,
  flatten-on-loss, stale data, dry run, and ML-mode eligibility
