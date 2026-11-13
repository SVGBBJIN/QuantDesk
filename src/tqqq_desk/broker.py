"""Broker abstraction: AlpacaBroker (alpaca-py) and FakeBroker (tests / dry runs without keys)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class DuplicateOrderError(RuntimeError):
    pass


class TransientBrokerError(RuntimeError):
    pass


@dataclass
class Account:
    equity: float
    last_equity: float
    buying_power: float
    cash: float
    account_number: str
    trading_blocked: bool = False


@dataclass
class OrderInfo:
    client_order_id: str
    symbol: str
    side: str           # "buy" | "sell"
    qty: float
    tif: str
    status: str
    filled_qty: float = 0.0
    filled_avg_price: float | None = None


class AlpacaBroker:
    def __init__(self, mode: str):
        from alpaca.trading.client import TradingClient

        from .data import credentials
        key, sec = credentials()
        self.client = TradingClient(key, sec, paper=(mode == "paper"))
        self.base_url = str(getattr(self.client, "_base_url", ""))

    @staticmethod
    def _order(o) -> OrderInfo:
        val = lambda x: getattr(x, "value", x)
        return OrderInfo(client_order_id=o.client_order_id, symbol=o.symbol, side=str(val(o.side)),
                         qty=float(o.qty or 0), tif=str(val(o.time_in_force)), status=str(val(o.status)),
                         filled_qty=float(o.filled_qty or 0),
                         filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None)

    def account(self) -> Account:
        a = self.client.get_account()
        return Account(equity=float(a.equity), last_equity=float(a.last_equity), buying_power=float(a.buying_power),
                       cash=float(a.cash), account_number=str(a.account_number),
                       trading_blocked=bool(a.trading_blocked or a.account_blocked))

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()}

    def open_orders(self) -> list[OrderInfo]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        return [self._order(o) for o in self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))]

    def get_order(self, client_order_id: str) -> OrderInfo | None:
        from alpaca.common.exceptions import APIError
        try:
            return self._order(self.client.get_order_by_client_id(client_order_id))
        except APIError as e:
            if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
                return None
            raise TransientBrokerError(str(e)) from e

    def submit_market(self, symbol: str, side: str, qty: int, tif: str, client_order_id: str) -> OrderInfo:
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        req = MarketOrderRequest(symbol=symbol, qty=int(qty), side=OrderSide(side),
                                 time_in_force=TimeInForce(tif), client_order_id=client_order_id)
        try:
            return self._order(self.client.submit_order(req))
        except APIError as e:
            msg = str(e).lower()
            if "client_order_id" in msg and ("unique" in msg or "duplicate" in msg):
                raise DuplicateOrderError(client_order_id) from e
            code = getattr(e, "status_code", None)
            if code is None or code >= 500 or code == 429:
                raise TransientBrokerError(str(e)) from e
            raise


@dataclass
class FakeBroker:
    """In-memory broker with Alpaca-like semantics (OPG orders fill at the next open)."""
    equity: float = 100_000.0
    last_equity: float = 100_000.0
    cash: float = 100_000.0
    account_number: str = "PA000FAKE"
    base_url: str = "https://paper-api.alpaca.markets"
    pos: dict = field(default_factory=dict)
    orders: dict = field(default_factory=dict)
    fail_next_submits: int = 0          # simulate transient network failures
    fail_after_accept: bool = False     # order accepted server-side but the response was lost

    def account(self) -> Account:
        return Account(self.equity, self.last_equity, self.cash * 2, self.cash, self.account_number)

    def positions(self) -> dict[str, float]:
        return {k: v for k, v in self.pos.items() if v}

    def open_orders(self) -> list[OrderInfo]:
        return [o for o in self.orders.values() if o.status in ("new", "accepted")]

    def get_order(self, client_order_id: str) -> OrderInfo | None:
        return self.orders.get(client_order_id)

    def submit_market(self, symbol, side, qty, tif, client_order_id) -> OrderInfo:
        if self.fail_next_submits > 0:
            self.fail_next_submits -= 1
            raise TransientBrokerError("simulated network error")
        if client_order_id in self.orders:
            raise DuplicateOrderError(client_order_id)
        o = OrderInfo(client_order_id, symbol, side, float(qty), tif, "accepted")
        self.orders[client_order_id] = o
        if self.fail_after_accept:
            self.fail_after_accept = False
            raise TransientBrokerError("simulated lost response")
        return o

    def fill_open_orders(self, prices: dict[str, float]) -> None:
        for o in self.open_orders():
            px = prices[o.symbol]
            sgn = 1 if o.side == "buy" else -1
            self.pos[o.symbol] = self.pos.get(o.symbol, 0) + sgn * o.qty
            self.cash -= sgn * o.qty * px
            o.status, o.filled_qty, o.filled_avg_price = "filled", o.qty, px

    def mark(self, prices: dict[str, float]) -> None:
        self.last_equity = self.equity
        self.equity = self.cash + sum(q * prices[s] for s, q in self.pos.items())
