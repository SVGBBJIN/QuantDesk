import pytest

from tqqq_desk.broker import FakeBroker, TransientBrokerError
from tqqq_desk.execution import (PlannedOrder, make_client_order_id, plan_orders, reconcile, submit_with_dedupe,
                                 verify_submitted)

M = ["TQQQ", "SQQQ"]
PX = {"TQQQ": 50.0, "SQQQ": 10.0}


def plan(cfg, target, positions, equity=100_000, bp=200_000):
    return plan_orders(target=target, positions=positions, equity=equity, prices=PX, buying_power=bp,
                       managed=M, cfg=cfg, signal_date="2024-07-05")


def test_idempotent_when_target_equals_current(cfg):
    orders, notes = plan(cfg, {"TQQQ": 1.0}, {"TQQQ": 2000})
    assert orders == []
    orders, notes = plan(cfg, {}, {})
    assert orders == [] and "no order" in notes[0]


def test_entry_exit_and_band(cfg):
    o, _ = plan(cfg, {"TQQQ": 0.5}, {})
    assert [(x.symbol, x.side, x.qty) for x in o] == [("TQQQ", "buy", 1000)]
    o, n = plan(cfg, {"TQQQ": 0.55}, {"TQQQ": 1000})        # +5% < 10% band
    assert o == [] and "band" in n[0]
    o, _ = plan(cfg, {"TQQQ": 0.8}, {"TQQQ": 1000})
    assert [(x.side, x.qty) for x in o] == [("buy", 600)]
    o, _ = plan(cfg, {}, {"TQQQ": 1000})
    assert [(x.side, x.qty) for x in o] == [("sell", 1000)]


def test_never_hold_both(cfg):
    with pytest.raises(ValueError):
        plan(cfg, {"SQQQ": 0.3}, {"TQQQ": 100})
    with pytest.raises(ValueError):
        plan(cfg, {"SQQQ": 0.3, "TQQQ": 0.3}, {})


def test_buying_power_limits_buy(cfg):
    o, n = plan(cfg, {"TQQQ": 1.0}, {}, bp=10_500)
    assert o[0].qty == int(10_500 // (50 * 1.05)) and "buying power" in n[0]


def test_client_order_id_deterministic(cfg):
    a = make_client_order_id("tqd", "2024-07-05", "TQQQ", "buy", 10)
    assert a == make_client_order_id("tqd", "2024-07-05", "TQQQ", "buy", 10) == "tqd-20240705-TQQQ-buy-10"
    o1, _ = plan(cfg, {"TQQQ": 0.5}, {})
    o2, _ = plan(cfg, {"TQQQ": 0.5}, {})
    assert o1[0].client_order_id == o2[0].client_order_id


def test_submit_dedupes_retries():
    b = FakeBroker()
    o = PlannedOrder("TQQQ", "buy", 10, "tqd-1")
    info, how = submit_with_dedupe(b, o, sleep=lambda s: None)
    assert how == "submitted"
    info, how = submit_with_dedupe(b, o, sleep=lambda s: None)    # retry of the same decision
    assert how == "already_submitted" and len(b.orders) == 1


def test_submit_retries_transient_and_lost_response():
    b = FakeBroker(fail_next_submits=2)
    info, how = submit_with_dedupe(b, PlannedOrder("TQQQ", "buy", 5, "tqd-2"), sleep=lambda s: None)
    assert how == "submitted" and len(b.orders) == 1
    b = FakeBroker(fail_after_accept=True)
    info, how = submit_with_dedupe(b, PlannedOrder("TQQQ", "buy", 5, "tqd-3"), sleep=lambda s: None)
    assert how == "submitted_after_retry" and len(b.orders) == 1   # no double order
    b = FakeBroker(fail_next_submits=10)
    with pytest.raises(TransientBrokerError):
        submit_with_dedupe(b, PlannedOrder("TQQQ", "buy", 5, "tqd-4"), attempts=3, sleep=lambda s: None)


def test_reconcile_and_verify():
    assert reconcile({"TQQQ": 10}, {"TQQQ": 10}, M) == []
    assert reconcile({"TQQQ": 10}, {}, M)
    assert any("more than one" in m for m in reconcile({"TQQQ": 1, "SQQQ": 1}, {"TQQQ": 1, "SQQQ": 1}, M))
    b = FakeBroker()
    p = PlannedOrder("TQQQ", "buy", 5, "tqd-5")
    assert verify_submitted([p], b.open_orders())
    b.submit_market("TQQQ", "buy", 5, "opg", "tqd-5")
    assert verify_submitted([p], b.open_orders()) == []
