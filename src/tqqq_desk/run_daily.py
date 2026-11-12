"""Daily runner: after the close, compute the signal and submit market-on-open orders.

Exit codes: 0 ok / nothing to do, 2 refused to start (live guard / config), 3 halted by kill switch,
4 unexpected error. State is persisted in every case where the broker was touched."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import mktcal
from .config import load_config
from .data import DataError, all_symbols, drop_incomplete, staleness_report
from .decision import CASH, PosState, decision_step
from .execution import plan_orders, reconcile, submit_with_dedupe, verify_submitted
from .features import build_features, make_labels
from .live_guard import LiveModeError, check_account_matches_mode, resolve_trading_mode
from .logutil import setup_logging
from .models import Ensemble
from .regime import classify_regimes
from .report import config_hash
from .risk import evaluate_kill_switch
from .state import StateStore
from .strategy import risk_inputs, size_for
from .validation import usable_rows

log = logging.getLogger("tqqq_desk.run")
ET = mktcal.ET


@dataclass
class RunResult:
    status: str                       # skipped | dry_run | traded | no_change | halted
    exit_code: int = 0
    signal_date: str | None = None
    orders: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    decision: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- model status
def resolve_model_mode(artifacts_dir: Path, cfg, now: pd.Timestamp) -> tuple[str, dict, str]:
    p = artifacts_dir / "model_status.json"
    if not p.exists():
        return "RULE", {}, "no validation status found -> rule-based fallback"
    st = json.loads(p.read_text())
    if st.get("mode") != "ML":
        return "RULE", st, st.get("message", "gate not passed -> rule-based fallback")
    if st.get("data_source") != "ALPACA":
        return "RULE", st, "validation not run on Alpaca data -> rule-based fallback"
    age = (now - pd.Timestamp(st["validated_at"])).days
    if age > cfg["risk"]["model_max_age_days"]:
        return "RULE", st, f"validation is {age} days old (> {cfg['risk']['model_max_age_days']}) -> rule-based fallback"
    if st.get("config_hash") != config_hash(cfg):
        return "RULE", st, "config changed since validation -> rule-based fallback until re-validated"
    return "ML", st, "ML ensemble passed the out-of-sample gate"


# ----------------------------------------------------------------------------- signal
def compute_live_signal(bars, cfg, prev: PosState, model_mode: str, calendar: bool, signal_date: dt.date) -> dict:
    sig_sym = cfg["data"]["signal_symbol"]
    feats = build_features(bars, cfg, include_calendar=calendar)
    t = pd.Timestamp(signal_date)
    if feats.index[-1] != t:
        raise DataError(f"feature row for {signal_date} missing (last {feats.index[-1].date()})")
    regime = classify_regimes(feats, cfg["regime"])
    r_today = regime.loc[t]
    p_up = p_hv = None
    trained_through = None
    if model_mode == "ML":
        labels = make_labels(bars[sig_sym]["close"], cfg["labels"]["horizon"], cfg["labels"]["high_vol_threshold"])
        rows = usable_rows(feats)
        X, Y = feats.loc[rows], labels.loc[rows]
        ens = Ensemble(cfg).fit(X, Y)  # labels of the last h rows are NaN and are excluded inside fit
        p = ens.predict(X.loc[[t]]).iloc[0]
        p_up, p_hv = float(p["p_up"]), float(p["p_hivol"])
        trained_through = str(Y["y_up"].last_valid_index().date())
    new_st, _, reason = decision_step(prev, r_today if isinstance(r_today, str) else None, p_up, p_hv,
                                      cfg["decision"], cfg["fallback"], model_mode)
    rk = {s: df.loc[t].to_dict() for s, df in risk_inputs(bars, cfg).items()}
    w = size_for(new_st.position, p_up, model_mode, rk, cfg)
    if new_st.position != CASH and w == 0:
        reason += " | size below min weight -> effectively cash"
    return {"regime": r_today, "p_up": p_up, "p_hivol": p_hv, "model_mode": model_mode, "state": new_st,
            "weight": w, "reason": reason, "trained_through": trained_through,
            "close": {s: float(bars[s]["close"].loc[t]) for s in cfg["data"]["trade_symbols"].values()}}


# ----------------------------------------------------------------------------- main flow
def run(cfg, *, now: pd.Timestamp, broker, bars_loader, state_dir: Path, artifacts_dir: Path, mode: str,
        dry_run: bool = False, force: bool = False, reset_kill_switch: bool = False,
        adopt_broker_positions: bool = False) -> RunResult:
    now = pd.Timestamp(now)
    now_et = now.tz_convert(ET)
    store = StateStore(state_dir)
    state = store.load()
    signal_date = mktcal.last_completed_session(now)
    sd = str(signal_date)
    managed = list(cfg["data"]["trade_symbols"].values())
    long_sym, short_sym = cfg["data"]["trade_symbols"]["long"], cfg["data"]["trade_symbols"]["short"]

    # --- time / calendar gate ------------------------------------------------------------
    hh, mm = map(int, cfg["execution"]["opg_window_start"].split(":"))
    eh, em = map(int, cfg["execution"]["opg_window_end"].split(":"))
    win_start = pd.Timestamp(dt.datetime.combine(signal_date, dt.time(hh, mm)), tz=ET)
    win_end = pd.Timestamp(dt.datetime.combine(mktcal.next_session(signal_date), dt.time(eh, em)), tz=ET)
    scheduled_ok = now_et.date() == signal_date and now_et >= win_start
    in_window = win_start <= now_et < win_end
    if not force and not scheduled_ok:
        why = ("market closed today" if not mktcal.is_session(now_et.date())
               else f"not the scheduled time (now {now_et:%H:%M} ET, runs after {cfg['execution']['opg_window_start']} ET)")
        log.info("skip: %s", why)
        return RunResult("skipped", 0, sd, reasons=[why])
    if not in_window and not dry_run:
        why = f"outside the market-on-open order window ({win_start:%m-%d %H:%M} -> {win_end:%m-%d %H:%M} ET); use --dry-run"
        log.warning("skip: %s", why)
        return RunResult("skipped", 0, sd, reasons=[why])
    if state.get("last_signal_date") == sd and not dry_run:
        log.info("skip: signal for %s already processed (idempotent)", sd)
        return RunResult("skipped", 0, sd, reasons=["already processed"])
    log.info("=== tqqq-desk %s run for session %s (%s) ===", mode.upper(), sd, "DRY RUN" if dry_run else "SUBMITTING ORDERS")
    if mktcal.is_half_day(signal_date):
        log.info("note: %s was a half-day session (early close)", sd)

    if reset_kill_switch and state["kill_switch"].get("tripped"):
        log.warning("kill switch manually RESET (was: %s)", state["kill_switch"].get("reason"))
        state["kill_switch"] = {"tripped": False, "reason": "", "action": "none", "tripped_at": None}

    # --- data -----------------------------------------------------------------------------
    data_problems: dict = {}
    bars = None
    try:
        bars = drop_incomplete(bars_loader(), signal_date)
        data_problems = staleness_report(bars, all_symbols(cfg), signal_date)
    except Exception as e:  # noqa: BLE001 - any data failure is a halt condition
        data_problems = {"ALL": f"data load failed: {type(e).__name__}: {e}"}

    # --- broker pre-checks ----------------------------------------------------------------
    acct = broker.account()
    check_account_matches_mode(mode, acct.account_number, getattr(broker, "base_url", ""))
    positions = {s: q for s, q in broker.positions().items() if s in managed}
    for p in state.get("pending_orders", []):
        o = broker.get_order(p["client_order_id"])
        if o is not None and o.status not in ("new", "accepted", "pending_new"):
            store.append("trades.csv", {"ts": now.isoformat(), "signal_date": p.get("signal_date"),
                                        "event": "RESULT", "symbol": o.symbol, "side": o.side, "qty": o.qty,
                                        "client_order_id": o.client_order_id, "status": o.status,
                                        "filled_qty": o.filled_qty, "fill_price": o.filled_avg_price, "mode": mode})
    stray = [o for o in broker.open_orders() if o.symbol in managed]
    if adopt_broker_positions:
        log.warning("adopting broker positions as expected state: %s", positions)
        state["expected_positions"] = dict(positions)
    mismatch = reconcile(state.get("expected_positions", {}), positions, managed)
    mismatch += [f"unexpected open order {o.client_order_id} {o.side} {o.qty} {o.symbol}" for o in stray]
    if acct.trading_blocked:
        mismatch.append("broker reports trading blocked")
    peak = max(float(state.get("peak_equity") or 0.0), acct.equity)
    ks = state["kill_switch"]
    kd = evaluate_kill_switch(equity=acct.equity, last_equity=acct.last_equity, peak_equity=peak,
                              data_problems=data_problems, mismatch=mismatch, already_tripped=ks.get("tripped", False),
                              tripped_reason=ks.get("reason", ""), tripped_action=ks.get("action", "freeze"),
                              risk_cfg=cfg["risk"])
    for r in kd.reasons:
        log.error("RISK: %s", r)

    # --- signal ---------------------------------------------------------------------------
    model_mode, model_status, model_msg = resolve_model_mode(artifacts_dir, cfg, now)
    log.info("model mode: %s (%s)", model_mode, model_msg)
    prev = PosState(**state["position"])
    sig = None
    if not data_problems:
        try:
            sig = compute_live_signal(bars, cfg, prev, model_mode, bool(model_status.get("calendar_features", False)),
                                      signal_date)
        except Exception as e:  # noqa: BLE001
            kd.halt, kd.action = True, (kd.action if kd.action != "none" else "freeze")
            kd.reasons.append(f"signal computation failed: {type(e).__name__}: {e}")
            log.error("signal computation failed:\n%s", traceback.format_exc())

    # --- target ---------------------------------------------------------------------------
    if kd.halt and kd.action == "flatten":
        target = {}
        log.error("KILL SWITCH: flattening all positions to cash")
    elif kd.halt:
        target = None  # freeze
    else:
        st = sig["state"]
        target = {st.position: sig["weight"]} if st.position != CASH and sig["weight"] > 0 else {}
    prices = sig["close"] if sig else ({s: float(bars[s]["close"].iloc[-1]) for s in managed}
                                       if bars and all(s in bars for s in managed) else {})
    orders, notes = [], []
    if target is not None:
        if not prices and positions:
            kd.reasons.append("cannot flatten: no prices available")
        else:
            try:
                orders, notes = plan_orders(target=target, positions=positions, equity=acct.equity, prices=prices,
                                            buying_power=acct.buying_power, managed=managed, cfg=cfg, signal_date=sd)
            except ValueError as e:
                kd.halt, kd.latch, kd.action = True, True, "freeze"
                kd.reasons.append(f"order planning refused: {e}")
                orders = []
    for n in notes:
        log.info("plan: %s", n)
    for o in orders:
        log.info("plan: %s %d %s (MOO, client_order_id=%s)", o.side.upper(), o.qty, o.symbol, o.client_order_id)

    decision = {
        "signal_date": sd, "regime": sig["regime"] if sig else None, "p_up": sig["p_up"] if sig else None,
        "p_hivol": sig["p_hivol"] if sig else None, "model_mode": model_mode, "model_message": model_msg,
        "state": sig["state"].position if sig else prev.position, "weight": sig["weight"] if sig else None,
        "reason": sig["reason"] if sig else "no signal (halted before signal)",
        "target": target, "orders": [o.to_dict() for o in orders], "halted": kd.halt, "halt_reasons": kd.reasons,
        "kill_action": kd.action, "positions": positions, "equity": acct.equity, "notes": notes,
    }
    log.info("decision: regime=%s P(up)=%s P(hivol)=%s -> %s w=%s | %s", decision["regime"], decision["p_up"],
             decision["p_hivol"], decision["state"], decision["weight"], decision["reason"])

    if dry_run:
        (Path(state_dir) / "last_dry_run.json").write_text(json.dumps(decision, indent=2, default=str))
        log.info("DRY RUN: no orders submitted, state not advanced")
        return RunResult("dry_run", 3 if kd.halt else 0, sd, [o.to_dict() for o in orders], kd.reasons, decision)

    # --- submit ---------------------------------------------------------------------------
    submitted = []
    for o in orders:
        try:
            info, how = submit_with_dedupe(broker, o)
        except Exception as e:  # noqa: BLE001
            kd.halt, kd.latch, kd.action = True, True, "freeze"
            kd.reasons.append(f"order submission failed for {o.client_order_id}: {e}")
            log.error("submit failed: %s", e)
            break
        submitted.append(o)
        store.append("trades.csv", {"ts": now.isoformat(), "signal_date": sd, "event": "SUBMIT", "symbol": o.symbol,
                                    "side": o.side, "qty": o.qty, "client_order_id": o.client_order_id,
                                    "status": info.status if info else how, "mode": mode, "note": how})
    # --- post-trade verification -------------------------------------------------------------
    after_pos = {s: q for s, q in broker.positions().items() if s in managed}
    post = verify_submitted(submitted, broker.open_orders())
    post += reconcile(positions, after_pos, managed)
    if post:
        kd.halt, kd.latch, kd.action = True, True, "freeze"
        kd.reasons += [f"post-trade check: {m}" for m in post]
        for m in post:
            log.error("POST-TRADE MISMATCH: %s", m)

    # --- persist --------------------------------------------------------------------------
    expected = dict(positions)
    for o in submitted:
        expected[o.symbol] = expected.get(o.symbol, 0.0) + (o.qty if o.side == "buy" else -o.qty)
    state["expected_positions"] = {s: q for s, q in expected.items() if abs(q) > 1e-9}
    state["pending_orders"] = [dict(o.to_dict(), signal_date=sd) for o in submitted]
    if kd.latch and not state["kill_switch"].get("tripped"):
        state["kill_switch"] = {"tripped": True, "reason": "; ".join(kd.reasons), "action": kd.action,
                                "tripped_at": now.isoformat()}
    if sig and not kd.halt:
        state["position"] = sig["state"].to_dict()
    elif kd.halt and kd.action == "flatten":
        state["position"] = PosState().to_dict()
    data_halt_only = kd.halt and not kd.latch and bool(data_problems)
    if not data_halt_only:  # let the second cron retry if data was merely late
        state["last_signal_date"] = sd
    state.update({
        "mode": mode, "last_run_at": now.isoformat(), "peak_equity": peak,
        "last_decision": {k: v for k, v in decision.items() if k != "positions"},
        "model": {"mode": model_mode, "message": model_msg, "validated_at": model_status.get("validated_at"),
                  "trained_through": sig.get("trained_through") if sig else None,
                  "gate_passed": model_status.get("gate_passed")},
        "data_last_bar": str(bars[cfg["data"]["signal_symbol"]].index[-1].date())
        if bars and cfg["data"]["signal_symbol"] in bars else None,
    })
    store.save(state)
    store.append("signals.csv", {"signal_date": sd, "regime": decision["regime"], "p_up": decision["p_up"],
                                 "p_hivol": decision["p_hivol"], "model_mode": model_mode, "state": decision["state"],
                                 "w_TQQQ": sig["weight"] if sig and sig["state"].position == long_sym else 0.0,
                                 "w_SQQQ": sig["weight"] if sig and sig["state"].position == short_sym else 0.0,
                                 "target_symbol": next(iter(target), "CASH") if target else "CASH",
                                 "target_qty": sum(o.qty for o in submitted if o.side == "buy"),
                                 "halted": kd.halt, "reason": decision["reason"]})
    store.append("equity.csv", {"date": sd, "equity": acct.equity, "last_equity": acct.last_equity, "cash": acct.cash,
                                "peak_equity": peak, "drawdown": acct.equity / peak - 1 if peak else 0.0, "mode": mode})
    status = "halted" if kd.halt else ("traded" if submitted else "no_change")
    return RunResult(status, 3 if kd.halt else 0, sd, [o.to_dict() for o in submitted], kd.reasons, decision)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="tqqq-desk daily runner (paper by default)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--artifacts-dir", default=None, help="defaults to <state-dir>/artifacts")
    ap.add_argument("--dry-run", action="store_true", help="compute everything, submit nothing")
    ap.add_argument("--force", action="store_true", help="skip the scheduled-time check (manual runs)")
    ap.add_argument("--reset-kill-switch", action="store_true")
    ap.add_argument("--adopt-broker-positions", action="store_true",
                    help="accept the broker's current TQQQ/SQQQ holdings as the expected state (first run / after manual fix)")
    ap.add_argument("--i-understand-the-risk", dest="risk_flag", action="store_true",
                    help="required (together with ALPACA_LIVE=true) for LIVE trading")
    ap.add_argument("--fake-broker", action="store_true", help="in-memory broker + synthetic data (smoke test)")
    ap.add_argument("--now", default=None, help="override current time (ISO, tz-aware) for testing")
    a = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(a.config)
        mode = resolve_trading_mode(os.environ, a.risk_flag, dry_run=a.dry_run)
    except (LiveModeError, ValueError) as e:
        log.error("REFUSING TO START: %s", e)
        return 2
    log.info("trading mode: %s", "LIVE (real money)" if mode == "live" else "PAPER")
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp.now(tz="UTC")
    state_dir = Path(a.state_dir)
    art = Path(a.artifacts_dir) if a.artifacts_dir else state_dir / "artifacts"
    try:
        if a.fake_broker:
            from .broker import FakeBroker
            from .data import synthetic_bars
            if mode != "paper":
                raise LiveModeError("--fake-broker is paper-only")
            sd = mktcal.last_completed_session(now)
            broker = FakeBroker()
            n_sess = len(mktcal.schedule("2016-01-04", str(sd)))
            bars_loader = lambda: synthetic_bars(n_sess, seed=11, start="2016-01-04")
        else:
            from .broker import AlpacaBroker
            from .data import load_bars
            broker = AlpacaBroker(mode)
            bars_loader = lambda: load_bars(cfg, refresh=True)
        res = run(cfg, now=now, broker=broker, bars_loader=bars_loader, state_dir=state_dir, artifacts_dir=art,
                  mode=mode, dry_run=a.dry_run, force=a.force, reset_kill_switch=a.reset_kill_switch,
                  adopt_broker_positions=a.adopt_broker_positions)
    except LiveModeError as e:
        log.error("REFUSING: %s", e)
        return 2
    except Exception:  # noqa: BLE001
        log.error("unexpected error:\n%s", traceback.format_exc())
        return 4
    log.info("result: %s %s", res.status, "; ".join(res.reasons))
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
