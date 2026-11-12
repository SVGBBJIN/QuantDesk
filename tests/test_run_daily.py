"""End-to-end daily runs against FakeBroker + synthetic bars aligned to the real NYSE calendar."""
import json

import pandas as pd
import pytest

from tqqq_desk import mktcal
from tqqq_desk.broker import FakeBroker
from tqqq_desk.data import synthetic_bars
from tqqq_desk.report import config_hash
from tqqq_desk.run_daily import run
from tqqq_desk.state import StateStore

ET = "America/New_York"
FRI = pd.Timestamp("2024-07-05 19:30", tz=ET)        # Jul 4 holiday, Jul 3 half day
MON = pd.Timestamp("2024-07-08 19:30", tz=ET)


def bars_until(date):
    n = len(mktcal.schedule("2016-01-04", str(date)))
    return synthetic_bars(n, seed=11, start="2016-01-04")


@pytest.fixture(scope="module")
def bars_fri():
    return bars_until("2024-07-05")


@pytest.fixture(scope="module")
def bars_mon():
    return bars_until("2024-07-08")


def go(cfg, tmp_path, broker, bars, now=FRI, **kw):
    return run(cfg, now=now, broker=broker, bars_loader=lambda: bars, state_dir=tmp_path,
               artifacts_dir=tmp_path / "artifacts", mode="paper", **kw)


def bull_cfg(cfg):
    # make the rule regime definitely long on the synthetic path, to exercise order flow
    return cfg.copy_with({"fallback.long_regimes": ["BULL", "NEUTRAL", "HIGH_VOL", "BEAR"]})


def test_skips_when_market_closed_or_wrong_time(cfg, tmp_path, bars_fri):
    b = FakeBroker()
    r = go(cfg, tmp_path, b, bars_fri, now=pd.Timestamp("2024-07-04 19:30", tz=ET))
    assert r.status == "skipped" and "market closed" in r.reasons[0]
    r = go(cfg, tmp_path, b, bars_fri, now=pd.Timestamp("2024-07-05 18:30", tz=ET))  # EST cron firing early
    assert r.status == "skipped" and "scheduled time" in r.reasons[0]
    # forced manual run mid-session: the Jul-3 signal's OPG window closed at 09:28 today
    r = go(cfg, tmp_path, b, bars_fri, now=pd.Timestamp("2024-07-05 10:00", tz=ET), force=True)
    assert r.status == "skipped" and "window" in r.reasons[0]
    assert not b.orders


def test_full_cycle_idempotent_and_reconciled(cfg, tmp_path, bars_fri, bars_mon):
    c = bull_cfg(cfg)
    b = FakeBroker()
    r = go(c, tmp_path, b, bars_fri)
    assert r.status == "traded" and len(r.orders) == 1 and r.orders[0]["side"] == "buy"
    assert all(o.tif == "opg" for o in b.orders.values())
    s = StateStore(tmp_path).load()
    assert s["last_signal_date"] == "2024-07-05" and s["expected_positions"] == {"TQQQ": r.orders[0]["qty"]}
    # the second (DST) cron for the same session is a no-op
    r2 = go(c, tmp_path, b, bars_fri, now=FRI + pd.Timedelta(hours=1))
    assert r2.status == "skipped" and len(b.orders) == 1
    # Monday open: OPG fills; Monday evening run reconciles and holds
    b.fill_open_orders({"TQQQ": float(bars_mon["TQQQ"]["open"].loc["2024-07-08"])})
    b.mark({"TQQQ": float(bars_mon["TQQQ"]["close"].loc["2024-07-08"])})
    r3 = go(c, tmp_path, b, bars_mon, now=MON)
    assert r3.status in ("no_change", "traded") and not r3.reasons
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert {"SUBMIT", "RESULT"} <= set(trades["event"]) and (trades["status"] == "filled").any()
    for f in ("signals.csv", "equity.csv", "state.json"):
        assert (tmp_path / f).exists()


def test_position_mismatch_halts_and_latches(cfg, tmp_path, bars_fri):
    b = FakeBroker(pos={"TQQQ": 50})          # someone traded manually; state expects nothing
    r = go(cfg, tmp_path, b, bars_fri)
    assert r.status == "halted" and r.exit_code == 3 and not b.orders
    s = StateStore(tmp_path).load()
    assert s["kill_switch"]["tripped"] and s["kill_switch"]["action"] == "freeze"


def test_adopt_positions_resolves_first_run(cfg, tmp_path, bars_fri):
    b = FakeBroker(pos={"TQQQ": 50})
    r = go(cfg, tmp_path, b, bars_fri, adopt_broker_positions=True)
    assert r.status != "halted"


def test_daily_loss_flattens(cfg, tmp_path, bars_fri):
    store = StateStore(tmp_path)
    st = store.load()
    st["expected_positions"] = {"TQQQ": 1000}
    st["peak_equity"] = 100_000
    store.save(st)
    b = FakeBroker(pos={"TQQQ": 1000}, equity=90_000, last_equity=100_000)
    r = go(bull_cfg(cfg), tmp_path, b, bars_fri)
    assert r.status == "halted"
    assert [(o["symbol"], o["side"], o["qty"]) for o in r.orders] == [("TQQQ", "sell", 1000)]
    s = store.load()
    assert s["kill_switch"]["tripped"] and s["position"]["position"] == "CASH"
    # still latched next session -> no new buys even though the signal is long
    b.fill_open_orders({"TQQQ": 50.0}); b.mark({"TQQQ": 50.0})
    r2 = go(bull_cfg(cfg), tmp_path, b, bars_until("2024-07-08"), now=MON)
    assert r2.status == "halted" and not [o for o in r2.orders if o["side"] == "buy"]
    # manual reset re-enables trading
    r3 = go(bull_cfg(cfg), tmp_path, b, bars_until("2024-07-09"), now=MON + pd.Timedelta(days=1),
            reset_kill_switch=True)
    assert r3.status == "traded"


def test_stale_data_halts_without_advancing(cfg, tmp_path, bars_fri):
    stale = {s: df.iloc[:-1] for s, df in bars_fri.items()}
    b = FakeBroker()
    r = go(cfg, tmp_path, b, stale)
    assert r.status == "halted" and any("data" in x for x in r.reasons) and not b.orders
    s = StateStore(tmp_path).load()
    assert s["last_signal_date"] is None and not s["kill_switch"]["tripped"]   # second cron may retry
    r2 = go(cfg, tmp_path, b, bars_fri, now=FRI + pd.Timedelta(hours=1))
    assert r2.status in ("traded", "no_change")


def test_dry_run_submits_nothing(cfg, tmp_path, bars_fri):
    b = FakeBroker()
    r = go(bull_cfg(cfg), tmp_path, b, bars_fri, dry_run=True)
    assert r.status == "dry_run" and r.orders and not b.orders
    assert StateStore(tmp_path).load()["last_signal_date"] is None
    assert (tmp_path / "last_dry_run.json").exists()


def test_ml_mode_used_only_with_valid_status(cfg, tmp_path, bars_fri):
    art = tmp_path / "artifacts"
    art.mkdir()
    status = {"mode": "ML", "data_source": "ALPACA", "validated_at": FRI.tz_convert("UTC").isoformat(),
              "config_hash": config_hash(cfg), "calendar_features": False, "gate_passed": True}
    (art / "model_status.json").write_text(json.dumps(status))
    r = go(cfg, tmp_path, FakeBroker(), bars_fri, dry_run=True)
    d = r.decision
    assert d["model_mode"] == "ML" and 0 <= d["p_up"] <= 1 and 0 <= d["p_hivol"] <= 1
    # changed config -> falls back to RULE until re-validated
    c2 = cfg.copy_with({"decision.up_enter": 0.56})
    r = go(c2, tmp_path, FakeBroker(), bars_fri, dry_run=True)
    assert r.decision["model_mode"] == "RULE" and "config changed" in r.decision["model_message"]
    # synthetic-data validation can never enable ML
    (art / "model_status.json").write_text(json.dumps(dict(status, data_source="SYNTHETIC")))
    assert go(cfg, tmp_path, FakeBroker(), bars_fri, dry_run=True).decision["model_mode"] == "RULE"
