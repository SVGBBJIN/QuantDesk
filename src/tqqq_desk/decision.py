"""Decision layer: a position state machine CASH / TQQQ / SQQQ.

Invariants (tested):
  * exactly one state at a time -> never TQQQ and SQQQ together,
  * a TQQQ<->SQQQ flip always passes through CASH for one session,
  * SQQQ is held at most `sqqq_max_days` consecutive sessions, then a cooldown applies,
  * missing inputs -> CASH."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .regime import BEAR, BULL, NEUTRAL

CASH, LONG, SHORT = "CASH", "TQQQ", "SQQQ"


@dataclass
class PosState:
    position: str = CASH
    days_held: int = 0
    sqqq_cooldown: int = 0

    def to_dict(self):
        return asdict(self)


def _fmt(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"


def decision_step(st: PosState, regime: str | None, p_up: float | None, p_hivol: float | None,
                  d, fallback, mode: str) -> tuple[PosState, str, str]:
    """Return (new_state, desired_before_flip_rule, plain-language reason)."""
    cooldown = max(0, st.sqqq_cooldown - 1)
    reasons = []
    if regime is None:
        desired, reasons = CASH, ["regime unavailable (insufficient history)"]
    elif mode == "ML":
        if p_up is None or p_hivol is None or not np.isfinite(p_up) or not np.isfinite(p_hivol):
            desired, reasons = CASH, ["model probabilities unavailable"]
        else:
            p_dn = 1 - p_up
            up_thr = d["up_exit"] if st.position == LONG else d["up_enter"]
            dn_thr = d["down_exit"] if st.position == SHORT else d["down_enter"]
            long_ok = regime in (BULL, NEUTRAL) and p_up >= up_thr and p_hivol < d["vol_cap"]
            short_ok = regime == BEAR and p_dn >= dn_thr
            if long_ok:
                desired = LONG
                reasons.append(f"regime {regime}, P(up)={_fmt(p_up)} >= {up_thr:.2f}, "
                               f"P(high vol)={_fmt(p_hivol)} < cap {d['vol_cap']:.2f}")
            elif short_ok:
                desired = SHORT
                reasons.append(f"regime BEAR, P(down)={_fmt(p_dn)} >= {dn_thr:.2f}")
            else:
                desired = CASH
                why = []
                if regime not in (BULL, NEUTRAL, BEAR):
                    why.append(f"regime {regime} allows no position")
                if regime in (BULL, NEUTRAL):
                    if p_up < up_thr:
                        why.append(f"P(up)={_fmt(p_up)} below {up_thr:.2f}")
                    if p_hivol >= d["vol_cap"]:
                        why.append(f"P(high vol)={_fmt(p_hivol)} at/above cap {d['vol_cap']:.2f}")
                if regime == BEAR:
                    why.append(f"P(down)={_fmt(p_dn)} below {dn_thr:.2f}")
                reasons.append("cash: " + "; ".join(why))
    else:  # RULE fallback
        if regime in fallback["long_regimes"]:
            desired = LONG
            reasons.append(f"rule-based: regime {regime} -> TQQQ")
        elif fallback["allow_sqqq"] and regime == BEAR:
            desired = SHORT
            reasons.append("rule-based: regime BEAR -> SQQQ")
        else:
            desired = CASH
            reasons.append(f"rule-based: regime {regime} -> cash")

    pre_flip = desired
    # SQQQ holding cap + cooldown
    if desired == SHORT:
        if st.position == SHORT and st.days_held >= d["sqqq_max_days"]:
            desired = CASH
            cooldown = d["sqqq_cooldown_days"]
            reasons.append(f"SQQQ holding cap of {d['sqqq_max_days']} days reached -> cash, "
                           f"{cooldown}-day cooldown")
        elif st.position != SHORT and cooldown > 0:
            desired = CASH
            reasons.append(f"SQQQ cooldown active ({cooldown} days left)")
    # never jump directly between TQQQ and SQQQ
    if {st.position, desired} == {LONG, SHORT}:
        reasons.append(f"flip {st.position}->{desired} goes via one session in cash")
        desired = CASH
    days = st.days_held + 1 if desired == st.position and desired != CASH else (1 if desired != CASH else 0)
    return PosState(desired, days, cooldown), pre_flip, " | ".join(reasons)
