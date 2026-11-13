import numpy as np
import pandas as pd

from tqqq_desk.regime import BEAR, BULL, HIGH_VOL, NEUTRAL, RegimeState, classify_regimes, regime_step


def _run(r, seq):
    st, out = RegimeState(), []
    for d, s, rv in seq:
        st = regime_step(st, d, s, rv, r)
        out.append(st.regime)
    return out


def test_entry_and_hysteresis(cfg):
    r = dict(cfg["regime"], min_hold_days=1)
    # enter BULL above +2%; stay BULL between -1% and +2%; leave below -1%
    seq = [(0.03, 0.01, 0.2), (0.0, 0.01, 0.2), (-0.005, 0.01, 0.2), (-0.02, 0.01, 0.2)]
    assert _run(r, seq) == [BULL, BULL, BULL, NEUTRAL]
    # BEAR needs both conditions; exits only above bear_exit (0.0)
    seq = [(-0.04, -0.01, 0.2), (-0.01, -0.01, 0.2), (0.01, -0.01, 0.2)]
    assert _run(r, seq) == [BEAR, BEAR, NEUTRAL]
    # d below -3% but SMA50 > SMA200 -> not BEAR
    assert _run(r, [(-0.04, 0.01, 0.2)]) == [NEUTRAL]


def test_high_vol_hysteresis(cfg):
    r = dict(cfg["regime"], min_hold_days=1)
    seq = [(0.05, 0.02, 0.2), (0.05, 0.02, 0.45), (0.05, 0.02, 0.35), (0.05, 0.02, 0.29)]
    assert _run(r, seq) == [BULL, HIGH_VOL, HIGH_VOL, BULL]


def test_min_hold_blocks_whipsaw(cfg):
    r = dict(cfg["regime"], min_hold_days=5)
    seq = [(0.03, 0.01, 0.2)] + [(-0.02, 0.01, 0.2)] * 6
    out = _run(r, seq)
    assert out[:5] == [BULL] * 5 and out[5] == NEUTRAL      # switch only once held 5 days
    # but HIGH_VOL overrides the minimum hold
    out = _run(r, [(0.03, 0.01, 0.2), (0.03, 0.01, 0.5)])
    assert out == [BULL, HIGH_VOL]
    r2 = dict(r, highvol_overrides_min_hold=False)
    assert _run(r2, [(0.03, 0.01, 0.2), (0.03, 0.01, 0.5)]) == [BULL, BULL]


def test_nan_gives_none_and_series(cfg, bars):
    assert _run(cfg["regime"], [(np.nan, 0.0, 0.2)]) == [None]
    from tqqq_desk.features import build_features
    reg = classify_regimes(build_features(bars, cfg), cfg["regime"])
    assert reg.iloc[:150].isna().all() and reg.iloc[260:].notna().all()
    assert set(reg.dropna()) <= {BULL, NEUTRAL, BEAR, HIGH_VOL}
