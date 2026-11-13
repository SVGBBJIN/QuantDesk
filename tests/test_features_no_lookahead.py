"""No-lookahead checks: features at t must be identical whether or not data after t exists,
and must not change when future prices are perturbed."""
import numpy as np
import pandas as pd
import pytest

from tqqq_desk.features import build_features, make_labels


def _truncate(bars, end):
    return {s: df.loc[:end] for s, df in bars.items()}


@pytest.mark.parametrize("cut_pos", [400, 700, 1100, -2])
def test_truncation_invariance(bars, cfg, cut_pos):
    full = build_features(bars, cfg)
    cut = full.index[cut_pos]
    trunc = build_features(_truncate(bars, cut), cfg)
    a, b = full.loc[:cut], trunc
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_frame_equal(a, b, check_freq=False, rtol=1e-10, atol=1e-12)


def test_future_perturbation_does_not_change_past(bars, cfg):
    base = build_features(bars, cfg)
    t = base.index[900]
    shocked = {s: df.copy() for s, df in bars.items()}
    for df in shocked.values():
        future = df.index > t
        df.loc[future, ["open", "high", "low", "close"]] *= np.linspace(0.5, 2.0, future.sum())[:, None]
    after = build_features(shocked, cfg)
    pd.testing.assert_frame_equal(base.loc[:t], after.loc[:t], check_freq=False)
    assert not base.loc[t:].iloc[1:].equals(after.loc[t:].iloc[1:])  # sanity: future did change


def test_features_have_expected_groups(bars, cfg):
    f = build_features(bars, cfg)
    for col in ["px_vs_sma200", "sma50_slope", "mom120", "dist_52w_high", "rv10", "vol_of_vol",
                "atr14_pct", "vol_ratio_10_60", "dd_depth", "dd_duration", "qqq_spy_rs20",
                "tlt_mom20", "vixy_vs_sma20", "dow", "days_to_month_end"]:
        assert col in f.columns, col
    # after warm-up there should be no NaNs
    assert f.iloc[300:].isna().sum().sum() == 0


def test_no_calendar_option(bars, cfg):
    f = build_features(bars, cfg, include_calendar=False)
    assert "dow" not in f.columns


def test_labels_look_forward_exactly_h(bars, cfg):
    h = cfg["labels"]["horizon"]
    c = bars["QQQ"]["close"]
    y = make_labels(c, h, 0.3)
    i = 500
    assert y["fwd_ret"].iloc[i] == pytest.approx(c.iloc[i + h] / c.iloc[i] - 1)
    lr = np.log(c).diff()
    assert y["fwd_vol"].iloc[i] == pytest.approx(lr.iloc[i + 1:i + h + 1].std() * np.sqrt(252))
    assert y["y_up"].iloc[-h:].isna().all() and y["y_hivol"].iloc[-h:].isna().all()
    assert y["y_up"].iloc[:-h].notna().all()


def test_context_symbol_dropped_when_missing(bars, cfg):
    b = {s: df for s, df in bars.items() if s != "VIXY"}
    f = build_features(b, cfg)
    assert "vixy_vs_sma20" not in f.columns and "tlt_mom20" in f.columns
