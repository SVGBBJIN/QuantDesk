import numpy as np
import pytest

from tqqq_desk.decision import CASH, LONG, SHORT, PosState, decision_step
from tqqq_desk.regime import BEAR, BULL, HIGH_VOL, NEUTRAL


def step(st, cfg, regime, p_up, p_hv=0.1, mode="ML"):
    return decision_step(st, regime, p_up, p_hv, cfg["decision"], cfg["fallback"], mode)[0]


def test_tqqq_entry_requires_regime_prob_and_vol(cfg):
    assert step(PosState(), cfg, BULL, 0.60).position == LONG
    assert step(PosState(), cfg, NEUTRAL, 0.60).position == LONG
    assert step(PosState(), cfg, BULL, 0.53).position == CASH          # below up_enter
    assert step(PosState(), cfg, BULL, 0.60, p_hv=0.7).position == CASH  # vol cap
    assert step(PosState(), cfg, HIGH_VOL, 0.70).position == CASH
    assert step(PosState(), cfg, BEAR, 0.70).position == CASH


def test_tqqq_hysteresis(cfg):
    st = step(PosState(), cfg, BULL, 0.60)
    st = step(st, cfg, BULL, 0.52)            # between exit 0.50 and entry 0.55 -> hold
    assert st.position == LONG and st.days_held == 2
    st = step(st, cfg, BULL, 0.49)
    assert st.position == CASH


def test_sqqq_needs_bear_and_higher_threshold(cfg):
    assert step(PosState(), cfg, BEAR, 0.38).position == SHORT          # P(down)=0.62
    assert step(PosState(), cfg, BEAR, 0.42).position == CASH           # P(down)=0.58 < 0.60
    assert step(PosState(), cfg, NEUTRAL, 0.20).position == CASH        # not BEAR


def test_sqqq_holding_cap_and_cooldown(cfg):
    n = cfg["decision"]["sqqq_max_days"]
    cd = cfg["decision"]["sqqq_cooldown_days"]
    st = PosState()
    held = 0
    for _ in range(n + 3):
        st = step(st, cfg, BEAR, 0.2)
        if st.position == SHORT:
            held += 1
        else:
            break
    assert held == n and st.position == CASH and st.sqqq_cooldown == cd
    for k in range(cd - 1):
        st = step(st, cfg, BEAR, 0.2)
        assert st.position == CASH, k
    st = step(st, cfg, BEAR, 0.2)
    assert st.position == SHORT                                       # cooldown expired


def test_flip_goes_through_cash(cfg):
    st = step(PosState(), cfg, BULL, 0.7)
    assert st.position == LONG
    st = step(st, cfg, BEAR, 0.2)                                     # wants SQQQ
    assert st.position == CASH
    st = step(st, cfg, BEAR, 0.2)
    assert st.position == SHORT
    st = step(st, cfg, BULL, 0.8)                                     # wants TQQQ
    assert st.position == CASH


def test_missing_inputs_give_cash(cfg):
    st = PosState(LONG, 3, 0)
    assert step(st, cfg, None, 0.7).position == CASH
    assert step(st, cfg, BULL, np.nan).position == CASH
    assert step(st, cfg, BULL, None).position == CASH


def test_rule_mode(cfg):
    assert step(PosState(), cfg, BULL, None, mode="RULE").position == LONG
    assert step(PosState(), cfg, NEUTRAL, None, mode="RULE").position == CASH
    assert step(PosState(), cfg, BEAR, None, mode="RULE").position == CASH  # allow_sqqq false


@pytest.mark.parametrize("seed", range(5))
def test_never_both_and_invariants_random(cfg, seed):
    rng = np.random.default_rng(seed)
    st = PosState()
    prev = CASH
    run = 0
    for _ in range(3000):
        reg = rng.choice([BULL, NEUTRAL, BEAR, HIGH_VOL, None])
        st = step(st, cfg, reg, float(rng.random()), float(rng.random()))
        assert st.position in (CASH, LONG, SHORT)
        assert {prev, st.position} != {LONG, SHORT}                    # never direct flip
        run = run + 1 if st.position == SHORT else 0
        assert run <= cfg["decision"]["sqqq_max_days"]
        prev = st.position


def test_reason_is_plain_language(cfg):
    _, _, why = decision_step(PosState(), BULL, 0.52, 0.2, cfg["decision"], cfg["fallback"], "ML")
    assert "P(up)=0.52 below 0.55" in why
