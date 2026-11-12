"""Kill switch.

  * daily loss > max_daily_loss or drawdown from peak > max_drawdown -> LATCHED halt,
    action = config risk.kill_action (flatten -> exit to cash; freeze -> no orders)
  * broker/state position mismatch or both TQQQ & SQQQ held       -> LATCHED halt, freeze
  * stale / missing data                                         -> halt THIS run (freeze), not latched
A latched kill switch stays on until a human runs with --reset-kill-switch."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KillDecision:
    halt: bool
    action: str               # "none" | "flatten" | "freeze"
    latch: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_kill_switch(*, equity: float, last_equity: float, peak_equity: float,
                         data_problems: dict, mismatch: list[str], already_tripped: bool,
                         tripped_reason: str, tripped_action: str, risk_cfg) -> KillDecision:
    reasons, latch = [], False
    action = "none"
    if already_tripped:
        reasons.append(f"kill switch already latched: {tripped_reason}")
        latch = True
        action = tripped_action if tripped_action in ("flatten", "freeze") else "freeze"
    if mismatch:
        reasons += [f"position mismatch: {m}" for m in mismatch]
        latch, action = True, "freeze"
    if last_equity > 0:
        day = equity / last_equity - 1
        if day < -risk_cfg["max_daily_loss"]:
            reasons.append(f"daily loss {day:.2%} exceeds limit {risk_cfg['max_daily_loss']:.2%}")
            latch = True
            action = action if action == "freeze" else risk_cfg["kill_action"]
    peak = max(peak_equity, equity)
    if peak > 0:
        dd = equity / peak - 1
        if dd < -risk_cfg["max_drawdown"]:
            reasons.append(f"drawdown {dd:.2%} exceeds limit {risk_cfg['max_drawdown']:.2%}")
            latch = True
            action = action if action == "freeze" else risk_cfg["kill_action"]
    if data_problems:
        reasons += [f"data {s}: {p}" for s, p in data_problems.items()]
        if action == "none":
            action = "freeze"
    return KillDecision(halt=bool(reasons), action=action, latch=latch, reasons=reasons)
