"""Order planning, idempotent submission (client_order_id dedupe) and broker reconciliation.

Order choice: market-on-open (Alpaca time_in_force=OPG), whole shares, submitted in the evening
(~19:30 ET) after the daily bar is final. OPG orders participate in the opening auction of the next
session - the same price the backtest assumes. Alpaca accepts OPG orders roughly 19:00 -> 09:28 ET;
OPG cannot be fractional, hence whole shares."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass

from .backtest import apply_rebalance_band
from .broker import DuplicateOrderError, OrderInfo, TransientBrokerError

log = logging.getLogger(__name__)


@dataclass
class PlannedOrder:
    symbol: str
    side: str
    qty: int
    client_order_id: str

    def to_dict(self):
        return asdict(self)


def make_client_order_id(prefix: str, signal_date: str, symbol: str, side: str, qty: int) -> str:
    """Deterministic per (signal date, symbol, side, qty): a retry of the same decision reuses the
    same id, so the broker rejects duplicates instead of double-trading."""
    return f"{prefix}-{signal_date.replace('-', '')}-{symbol}-{side}-{int(qty)}"


def plan_orders(*, target: dict[str, float], positions: dict[str, float], equity: float,
                prices: dict[str, float], buying_power: float, managed: list[str], cfg,
                signal_date: str) -> tuple[list[PlannedOrder], list[str]]:
    """target: {symbol: weight}. Returns (orders, notes). Never results in two managed symbols held."""
    notes = []
    if sum(1 for s in managed if target.get(s, 0) > 0) > 1:
        raise ValueError("target holds more than one leveraged ETF")
    cur_qty = {s: float(positions.get(s, 0.0)) for s in managed}
    cur_w = {s: cur_qty[s] * prices[s] / equity if equity > 0 else 0.0 for s in managed}
    tgt_w = {s: float(target.get(s, 0.0)) for s in managed}
    banded = apply_rebalance_band(tgt_w, cur_w, cfg["sizing"]["rebalance_band"])
    if banded == {s: cur_w[s] for s in banded} and any(v > 0 for v in tgt_w.values()):
        notes.append("change within rebalance band -> no order")
        return [], notes
    tgt_qty = {s: math.floor(tgt_w[s] * equity / prices[s]) if tgt_w[s] > 0 else 0 for s in managed}
    prefix = cfg["execution"]["client_order_prefix"]
    orders: list[PlannedOrder] = []
    for s in managed:  # sells first
        if cur_qty[s] > tgt_qty[s]:
            q = math.floor(cur_qty[s] - tgt_qty[s] + 1e-9)
            if q < cur_qty[s] - tgt_qty[s] - 1e-9:
                notes.append(f"{s}: fractional remainder {cur_qty[s] - tgt_qty[s] - q:.4f} sh cannot go OPG; left in place")
            if q > 0:
                orders.append(PlannedOrder(s, "sell", q, make_client_order_id(prefix, signal_date, s, "sell", q)))
    for s in managed:
        if tgt_qty[s] > cur_qty[s]:
            others_held = [o for o in managed if o != s and cur_qty[o] > 0]
            if others_held:
                raise ValueError(f"refusing to buy {s} while holding {others_held}: never hold both ETFs")
            q = int(tgt_qty[s] - math.floor(cur_qty[s]))
            px = prices[s] * (1 + cfg["risk"]["buying_power_buffer"])
            if q * px > buying_power:
                q2 = int(buying_power // px)
                notes.append(f"{s}: buying power {buying_power:,.0f} limits buy {q} -> {q2}")
                q = q2
            if q > 0:
                orders.append(PlannedOrder(s, "buy", q, make_client_order_id(prefix, signal_date, s, "buy", q)))
    if not orders:
        notes.append("target equals current position -> no order (idempotent)")
    return orders, notes


def submit_with_dedupe(broker, o: PlannedOrder, tif: str = "opg", attempts: int = 3,
                       sleep=time.sleep) -> tuple[OrderInfo | None, str]:
    try:
        existing = broker.get_order(o.client_order_id)
    except TransientBrokerError:
        existing = None
    if existing is not None:
        return existing, "already_submitted"
    last_err = None
    for i in range(attempts):
        try:
            return broker.submit_market(o.symbol, o.side, o.qty, tif, o.client_order_id), "submitted"
        except DuplicateOrderError:
            return broker.get_order(o.client_order_id), "duplicate_existing"
        except TransientBrokerError as e:
            last_err = e
            log.warning("transient error submitting %s (attempt %d/%d)", o.client_order_id, i + 1, attempts)
            try:
                existing = broker.get_order(o.client_order_id)
            except TransientBrokerError:
                existing = None
            if existing is not None:
                return existing, "submitted_after_retry"
            sleep(2 ** i)
    raise TransientBrokerError(f"giving up on {o.client_order_id}: {last_err}")


def reconcile(expected: dict[str, float], actual: dict[str, float], managed: list[str]) -> list[str]:
    out = []
    for s in managed:
        e, a = float(expected.get(s, 0.0)), float(actual.get(s, 0.0))
        if abs(e - a) > 1e-6:
            out.append(f"{s}: expected {e:g} sh, broker has {a:g} sh")
    if sum(1 for s in managed if actual.get(s, 0) > 0) > 1:
        out.append(f"broker holds more than one of {managed} simultaneously")
    return out


def verify_submitted(planned: list[PlannedOrder], open_orders: list[OrderInfo]) -> list[str]:
    by_id = {o.client_order_id: o for o in open_orders}
    out = []
    for p in planned:
        o = by_id.get(p.client_order_id)
        if o is None:
            out.append(f"order {p.client_order_id} not found among open orders after submit")
        elif o.symbol != p.symbol or o.side != p.side or abs(o.qty - p.qty) > 1e-9:
            out.append(f"order {p.client_order_id} differs from plan ({o.symbol} {o.side} {o.qty})")
    return out
