# tqqq-desk

A systematic **daily-swing** trading system that holds **TQQQ** (bullish), **SQQQ** (bearish) or **cash**,
executed through the Alpaca API by **Vercel Cron** jobs (credentials in **Vercel environment variables**),
with a static status page on Vercel and a Streamlit dashboard for local use. **Paper trading is the default.**

> ⚠️ TQQQ and SQQQ are 3× daily-reset leveraged ETFs. They can lose most of their value in weeks,
> and SQQQ decays over time even in sideways markets. Nothing here is investment advice. Run it on
> paper for months before you even consider real money.

---

## How it decides: three strategies, one objective

```
 Alpaca daily bars (QQQ, TQQQ, SQQQ, SPY, TLT, UUP, VIXY; split+dividend adjusted)
        │
        ▼
 Causal features (row t uses only bars ≤ close of t): SMAs 50/100/150/200/250, momentum,
 realised vol (QQQ and each ETF's own), vol-of-vol, drawdown, cross-asset context
        │
        ├─► sentinel     (designed for RISK-ADJUSTED)   trend score = 6 votes (price vs SMA50/100/200,
        │                 SMA50>SMA200, 60d & 120d momentum). TQQQ at a 40% vol target × score, enter ≥4/6,
        │                 hold ≥2/6; half size above 45% QQQ vol; flat on a vol spike below SMA50. Small SQQQ
        │                 sleeve (≤25%, vol-targeted) only when all 6 votes are bearish and QQQ ≥3% under SMA200.
        ├─► accelerator  (designed for MAX RETURN under a drawdown cap)  near-full TQQQ (75% vol target)
        │                 above SMA200 (±1% bands) or on a fast re-entry (above a rising SMA50 with positive
        │                 20d momentum, calm vol); drawdown brake shrinks size from 15% to 30% strategy
        │                 drawdown (high-water mark decays, half-life 126 sessions). SQQQ sleeve available, off by default.
        ├─► anchor       (designed for ROBUSTLY beating SMA200)  ensemble of SMA150/200/250 votes, each
        │                 with ±1% entry/exit bands; size = votes × 65% vol target; half size on a crash spike.
        ├─► ml           the original LightGBM + logistic ensemble (calibrated, walk-forward) + regime
        │                 state machine, kept as a candidate
        └─► sma200       benchmark and FALLBACK: TQQQ if QQQ > SMA200, else cash
        │
        ▼
 Shared position rules: never TQQQ and SQQQ together · a flip spends one session in cash · SQQQ at most
 10 sessions then a 5-session cooldown · weights < 5% → cash · 10% rebalance band
        │
        ▼
 Weekly validation scores EVERY candidate against sma200 under ALL THREE objectives and picks the best
 eligible one for the configured objective (`selection.objective` or env `QD_OBJECTIVE`):
   risk_adjusted  Sharpe > SMA200, Sortino > SMA200, paired-bootstrap P(Sharpe better) ≥ 0.60 → highest Sharpe
   max_return     max drawdown within 45%, CAGR > SMA200, bootstrap P(growth better) ≥ 0.60  → highest CAGR
   robust         beats SMA200 in ≥55% of calendar years, bootstrap P(Sharpe better) ≥ 0.60, and the WORST
                  Sharpe across a 3×3 parameter neighbourhood still beats SMA200        → highest worst-case
 Nothing eligible → the desk trades the plain sma200 benchmark.
```

Each rule strategy is one causal loop over history (`strategies.py`). The backtest and the live runner run
**the same path**; live simply takes tonight's last row, so there is no backtest/live drift in hysteresis,
holding-day counters or the drawdown brake (tested).

Code map (`src/tqqq_desk/`): `strategies.py` (sentinel/accelerator/anchor/sma200) · `selection.py`
(objectives) · `research.py` · `report.py` · `data.py` · `features.py` · `regime.py` · `models.py` ·
`validation.py` · `decision.py` + `strategy.py` + `sizing.py` (ml candidate) · `backtest.py` · `metrics.py`
· `sensitivity.py` · `risk.py` · `execution.py` · `broker.py` · `live_guard.py` · `state.py` · `mktcal.py`
· `run_daily.py` · `ghstate.py` + `vercel_jobs.py` (serverless). Vercel entry points: `api/cron/*.py`.

### Evidence so far (read the caveats)
Alpaca history starts in 2016, so the strategies were also stress-tested on a **1999–2018 proxy**
(`scripts/make_proxy_history.py`: real NASDAQ Composite OHLC, TQQQ/SQQQ simulated as daily-reset ±3×
with fees and financing). It includes the dot-com crash, 2008, 2011, 2015-16 and 2018 Q4. Window 2003→2018
(after the ML warm-up), after costs:

| | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| sentinel | 9.0% | 0.44 | -52% | 0.17 |
| accelerator | 13.5% | 0.54 | -56% | 0.24 |
| anchor | 14.0% | 0.54 | -57% | 0.25 |
| sma200 (benchmark) | 11.1% | 0.47 | -64% | 0.17 |
| ml | 3.2% | 0.25 | -58% | 0.06 |
| buy & hold TQQQ (proxy) | 5.0% | 0.40 | -96% | 0.05 |

Over the full 2000–2018 proxy (including the dot-com crash), sma200 lost 91% peak-to-trough while
sentinel/accelerator/anchor were at −52% / −64% / −75%, and all three beat it on Sharpe. **None of
them meets a 45% drawdown cap on the proxy**: a 3× ETF through 2000-2002 is brutal. These rules were
designed while looking at this history, so treat the numbers as partly in-sample. The weekly validation on
real Alpaca data decides what actually trades.

### Validation (what "out-of-sample" means here)
* The ML candidate: expanding-window walk-forward, retrained every 63 sessions, purged (5-day labels) plus
  a 5-session embargo. Its OOS window defines the evaluation window for every candidate, so all are compared on the same days.
* Rule candidates have no fitted parameters; robustness is checked with 3×3 parameter neighbourhoods.
* Backtest executes at the **next session's open** at the actual TQQQ/SQQQ open price, with the overnight gap
  attributed to the old position. Costs are 2 bps + 1 bp slippage per side. Leveraged ETF returns come from
  **their own price history**, so decay is real.
* Reported: CAGR, vol, Sharpe (with 95% stationary block-bootstrap CI), Sortino, max drawdown, Calmar,
  turnover, exposure, hit rate, trade win rate, cost drag; a candidates × objectives matrix with every check.
* Sanity tests: leakage canary, positive control, truncation and future-perturbation invariance for every
  strategy, live-equals-backtest parity.

---

## Setup

```bash
python3.12 -m venv .venv && . .venv/bin/activate        # 3.11 also works
pip install -r requirements-dev.txt && pip install -e . --no-deps
python -m pytest                                          # ~120 tests, no network
```
`requirements.txt` is the slim runtime set Vercel installs (keep it small: the Python function bundle limit
is 500 MB; it is ~390 MB today). `requirements-dev.txt` adds the dashboard, relay and test tools.

### Running on Vercel (the default)
`vercel.json` defines two Python functions and three cron jobs (UTC; ET in brackets):

| path | schedule | what |
|---|---|---|
| `/api/cron/trade` | `7 1 * * 2-6` | Mon–Fri evening run (20:07 EST / 21:07 EDT) |
| `/api/cron/trade` | `7 4 * * 2-6` | backup run (23:07 EST / 00:07 EDT); a no-op if the session was processed |
| `/api/cron/validate` | `23 15 * * 6` | Saturday validation: picks the active strategy |

Both times work all year on the Hobby plan, where crons fire once a day, anywhere within the scheduled hour.
Each run pulls the `data` branch into `/tmp` through the GitHub API, runs the same code as the CLI, commits the
changed files back as one commit, and appends to `runs.json` (shown on the status page). Function max
duration is 300 s; a validation on Alpaca data takes ~30 s.

**Environment variables** (Vercel → Project → Settings → Environment Variables, scope **Production**;
never commit them):

| name | required | value |
|---|---|---|
| `ALPACA_KEY` | yes | Alpaca API key ID (paper keys unless going live) |
| `ALPACA_SECRET` | yes | Alpaca secret key |
| `CRON_SECRET` | yes | random string ≥ 16 chars (`openssl rand -hex 32`). Vercel sends it as `Authorization: Bearer …` to cron paths; every request without it is rejected |
| `QD_GITHUB_TOKEN` | yes | fine-grained GitHub token, **this repo only**, permission *Contents: read and write* (to read/write the `data` branch) |
| `QD_OBJECTIVE` | no | `risk_adjusted` (default), `max_return` or `robust`. Changing it changes the config hash, so run the validate job afterwards |
| `QD_GITHUB_REPO` / `QD_STATE_BRANCH` | no | defaults `SVGBBJIN/QuantDesk` / `data` |
| `ALPACA_LIVE` + `I_UNDERSTAND_THE_RISK` | no | both `true` for real money (see *Going live*) |

Secret values are redacted from all logs (`logutil.py`, tested). Preview deployments refuse to place orders
(dry runs only), even though Vercel only runs crons on production.

For local work, `vercel env pull .env` gives you the same variables (`.env` is git-ignored); export them in your shell.

### First run (paper)
1. Set the env vars above and **redeploy** (env changes apply to new deployments).
2. Validate once now instead of waiting for Saturday:
   ```bash
   curl -H "Authorization: Bearer $CRON_SECRET" https://<your-app>.vercel.app/api/cron/validate
   ```
3. Dry run (computes everything, submits nothing, returns the decision and log as JSON):
   ```bash
   curl -H "Authorization: Bearer $CRON_SECRET" "https://<your-app>.vercel.app/api/cron/trade?dry_run=1&force=1"
   ```
4. If your paper account already holds TQQQ/SQQQ, run once with `?adopt_broker_positions=1` (evening, inside the
   order window), or position reconciliation will (correctly) halt.
5. From then on the crons run by themselves. Other flags: `force=1` (skip the time-of-day check),
   `reset_kill_switch=1` (after you have investigated a halt).

The GitHub Actions workflows `daily-trade` and `validate` remain as **manual-only fallbacks** (they need the
`ALPACA_KEY`/`ALPACA_SECRET` repository secrets). They no longer run on a schedule, so nothing trades twice.

Local equivalents:
```bash
python scripts/backtest_report.py                         # needs ALPACA_KEY/SECRET
python -m tqqq_desk.run_daily --state-dir state --dry-run --force
python scripts/backtest_report.py --synthetic             # no keys: plumbing check, NOT evidence
pip install arch && python scripts/make_proxy_history.py --out data/proxy
python scripts/backtest_report.py --csv-dir data/proxy --out reports/proxy --artifacts /tmp/proxy_art   # 1999-2018 stress test
python -m tqqq_desk.run_daily --fake-broker --dry-run --force --state-dir /tmp/st --now 2024-07-05T19:30:00-04:00
```

---

## Execution: order type, timing, idempotency

**Choice: market-on-open (Alpaca `time_in_force=OPG`), whole shares, submitted in the evening.**
* The signal uses the final daily bar, so the run happens after the close. Alpaca only accepts OPG
  orders from about **19:00 ET until 09:28 ET** the next session, so the scheduled run is 20:07–21:07 ET.
* OPG orders fill in the opening auction, which is exactly the price the backtest assumes. A pre-market
  DAY order would add a second scheduled job and fill at a less well-defined price.
* Cost of this choice: OPG cannot be fractional, so positions are rounded down to whole shares.
  A buy that the buying-power check (with a 5% gap buffer) cannot cover is reduced.

**Scheduling with DST.** Cron is UTC and ignores DST; 01:07 UTC is 20:07 EST and 21:07 EDT, and the
04:07 UTC backup is 23:07 EST / 00:07 EDT, both inside the order window year-round (a test checks this
for any minute of the scheduled hour). The script still exits early unless (a) the session is an NYSE
session per `pandas_market_calendars`, (b) it is after 19:00 ET, and (c) that session has not already
been processed.

**Idempotency and dedupe.**
* `state.json.last_signal_date` blocks reprocessing a session.
* If the target equals the current position, or the change is inside the 10% rebalance band, no order is placed.
* `client_order_id = tqd-<YYYYMMDD>-<SYMBOL>-<side>-<qty>` is deterministic. The broker is checked
  for that id before submitting. Transient errors are retried and re-checked by id, and a duplicate-id
  rejection is treated as "already submitted". A lost HTTP response can therefore never double-trade.
* State commits go through the GitHub API as one commit per run; if another run committed meanwhile, the
  commit is rebuilt on the new head (`ghstate.py`, tested). Broker-side `client_order_id` dedupe still
  protects against double orders if two runs ever overlap.

**State** lives on the orphan **`data` branch** (`state.json`, `trades.csv`, `signals.csv`,
`equity.csv`, `runs.json`, `artifacts/`, `reports/`), committed by the bot so `main` history stays clean.

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
To resume after investigating, call `/api/cron/trade?reset_kill_switch=1` (with the `CRON_SECRET` header). Thresholds are in `config.yaml → risk`.

Strategy safety: the daily runner trades the validated strategy only if `artifacts/model_status.json` was
produced on **Alpaca** data, is under 14 days old, **and** its config hash matches the current config
(including `QD_OBJECTIVE`). Otherwise it trades the plain sma200 benchmark and says why.

## Going live (real money)
Live trading requires **both** of these, and refuses to start otherwise:
1. `ALPACA_LIVE=true` (Vercel env var), **and**
2. `I_UNDERSTAND_THE_RISK=true` (Vercel env var; on the CLI the flag `--i-understand-the-risk`).

The runner also checks that the account number and endpoint match the mode: paper accounts are `PA…`
on `paper-api.alpaca.markets`. Use live API keys in `ALPACA_KEY`/`ALPACA_SECRET`. Before going live:
- [ ] At least a few months of paper running with no reconciliation halts
- [ ] Read the latest `backtest_report.md`: which strategy is active, for which objective, and why
- [ ] Decide the kill-switch thresholds you can actually live with
- [ ] Start with a small account. Margin/PDT rules apply to accounts under $25k

## Dashboard
```bash
git fetch origin data && git worktree add state origin/data   # once
git -C state pull                                             # to refresh
streamlit run dashboard/app.py                                # or TQQQ_STATE_DIR=/path ...
```
It shows a PAPER/LIVE banner, kill-switch state, which strategy is active (and why),
today's regime and probabilities with a plain-language reason, current position and target, data and
model freshness, live equity and drawdown, OOS backtest equity vs benchmarks, drawdowns, recent trades and signals,
split across tabs (Live market / Decision & freshness / Performance / Trades & signals).

### Status page (Vercel)
`index.html` at the repo root is a static, read-only status page for the Vercel deployment (Streamlit
cannot run on Vercel). In the browser it reads the bot state and the cron run log (`runs.json`) from the public
`data` branch, and CI runs from the GitHub API. It shows job results, the active strategy, the PAPER/LIVE banner, kill switch,
the latest decision, equity and recent trades/signals. It needs no build step and no keys.
Keep this file. Without it the Vercel URL returns 404.

### Live market data
Data-only, never places orders.
```bash
# free Yahoo Finance scraper: no key, unofficial endpoint, for dashboards/research (not the trading path)
python -m tqqq_desk.yahoo QQQ TQQQ                 # quotes
python -m tqqq_desk.yahoo QQQ --history 6mo        # daily bars

# Alpaca real-time WebSocket (free plan = IEX feed), printed to the terminal
ALPACA_KEY=... ALPACA_SECRET=... python -m tqqq_desk.stream QQQ TQQQ SQQQ

# local relay: Alpaca stream in, REST + WebSocket out (keys stay server-side); Yahoo fills in when the stream is down
QD_SYMBOLS=QQQ,TQQQ,SQQQ,SPY,TLT uvicorn tqqq_desk.server:app --port 8765
#   GET /api/health  /api/quotes  /api/history/{symbol}    WS /ws
```
The Live tab polls the relay (`QD_SERVER`, default `http://127.0.0.1:8765`) and falls back to Yahoo directly
if it is not running. Alpaca allows one stream connection per account, so do not run two relays on one key.

## Configuration
Everything tunable is in `config.yaml`: `selection` (objective, candidates, thresholds), `strategies`
(each strategy's parameters), ML model/decision settings, sizing, costs, risk limits. `config.py` validates
it. For example, leverage above 1× on top of the 3× ETF, an SQQQ sleeve above 50%, exit thresholds above
entry thresholds, or an unknown objective or strategy are all rejected. `QD_OBJECTIVE` overrides
`selection.objective` from the environment.

## Known limitations (read these)
* **Short history.** Alpaca daily data starts around 2016, so the OOS window is roughly 2018–today. It
  contains only a handful of real bear phases (2018 Q4, 2020, 2022, 2025). Bootstrap CIs on Sharpe
  are wide, so treat point estimates with suspicion.
* **Selection bias.** The rule strategies were designed while looking at history (including the 1999-2018
  proxy), and the selector picks the best of several candidates on the same window. The bootstrap
  threshold, year-by-year and neighbourhood checks reduce this but cannot remove it.
* **The proxy is not TQQQ.** It is 3× the NASDAQ Composite (not the Nasdaq-100) with estimated fees and
  financing. It shows behaviour through crashes; it is not a performance forecast.
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
* A missed cron (both the evening run and its backup) means no trade that session. Hobby-plan crons are
  hour-precise only and are not retried by Vercel.
* The serverless bundle is ~390 MB of the 500 MB Python limit; adding heavy dependencies to
  `requirements.txt` can break the deployment.

## Tests
`python -m pytest` covers:
- no-lookahead: truncation invariance, future-perturbation invariance, label alignment, leakage canary
- purged/embargoed splits
- the regime hysteresis and min-hold
- the position state machine (SQQQ cap and cooldown, flip via cash, never both, randomised invariants)
- every rule strategy: no lookahead (truncation and future-perturbation invariance), position invariants,
  live = backtest last-row parity, the drawdown brake, objective selection and fallback, config validation
- the Vercel path: cron auth, flags, GitHub-API state sync (incl. concurrent-commit rebase) against a fake
  GitHub, end-to-end job runs, preview deployments refusing to trade, cron times inside the order window
- sizing math (vol target, ¼-Kelly cap, per-ETF caps)
- the backtest engine (next-open execution, overnight attribution, costs, band)
- metrics and bootstrap
- plateau selection for the ML threshold grid
- the kill switch
- the live-mode guard (including the CLI refusing to start) and secret redaction
- order planning and dedupe/retry
- full daily-run integration against a fake broker: holidays, DST double-cron, reconciliation halts,
  flatten-on-loss, stale data, dry run, and strategy eligibility (stale or mismatched validation → sma200)
