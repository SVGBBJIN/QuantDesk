"""Strategy family: causality, shared position invariants, live/backtest parity, selection logic."""
import numpy as np
import pandas as pd
import pytest

from tqqq_desk.config import ConfigError
from tqqq_desk.data import synthetic_bars
from tqqq_desk.selection import FALLBACK, evaluate, select
from tqqq_desk.state import StateStore
from tqqq_desk.strategies import (CASH, LONG, SHORT, PathState, apply_rules, make_inputs, perturbations,
                                  position_rules, run_path, to_target_weights)

NAMES = ["sentinel", "accelerator", "anchor", "sma200"]


@pytest.fixture(scope="module")
def big():
    return synthetic_bars(1600, seed=21)


@pytest.fixture(scope="module")
def paths(cfg, big):
    inp = make_inputs(big, cfg)
    return {n: run_path(n, inp, cfg) for n in NAMES}


@pytest.mark.parametrize("name", NAMES)
def test_no_lookahead_truncation_invariance(cfg, big, paths, name):
    """Decisions up to t must not change when bars after t are removed."""
    cut = 1100
    trunc = {s: df.iloc[:cut] for s, df in big.items()}
    short = run_path(name, make_inputs(trunc, cfg), cfg)
    full = paths[name].iloc[:cut]
    pd.testing.assert_frame_equal(short[["state", "w_TQQQ", "w_SQQQ", "days_held"]],
                                  full[["state", "w_TQQQ", "w_SQQQ", "days_held"]])


@pytest.mark.parametrize("name", NAMES)
def test_future_perturbation_invariance(cfg, big, paths, name):
    cut = 1200
    pert = {s: df.copy() for s, df in big.items()}
    for df in pert.values():
        df.iloc[cut:, :4] *= 1.3
    alt = run_path(name, make_inputs(pert, cfg), cfg)
    assert (alt.iloc[:cut]["state"] == paths[name].iloc[:cut]["state"]).all()


@pytest.mark.parametrize("name", NAMES)
def test_position_invariants(cfg, paths, name):
    p = paths[name]
    d = cfg["decision"]
    assert not ((p["w_TQQQ"] > 0) & (p["w_SQQQ"] > 0)).any()                    # never both
    assert ((p["w_TQQQ"] <= 1) & (p["w_SQQQ"] <= 0.5) & (p[["w_TQQQ", "w_SQQQ"]] >= 0).all(axis=1)).all()
    prev = p["state"].shift()
    assert not ((prev == LONG) & (p["state"] == SHORT)).any()                    # flips go via cash
    assert not ((prev == SHORT) & (p["state"] == LONG)).any()
    assert (p.loc[p["state"] == SHORT, "days_held"] <= d["sqqq_max_days"]).all()
    assert (p.loc[p["state"] == CASH, ["w_TQQQ", "w_SQQQ"]] == 0).all().all()
    inv = p["state"] != CASH
    assert (p.loc[inv, ["w_TQQQ", "w_SQQQ"]].sum(axis=1) >= cfg["sizing"]["min_weight"]).all()


def test_strategies_actually_trade_and_differ(paths):
    for n in NAMES:
        assert (paths[n]["state"] == LONG).mean() > 0.2, n
    assert not paths["sentinel"]["w_TQQQ"].equals(paths["anchor"]["w_TQQQ"])


def test_apply_rules_cap_cooldown_and_flip(cfg):
    r = position_rules(cfg)
    st = PathState(SHORT, r["sqqq_max_days"], 0, 0.2)
    st = apply_rules(st, SHORT, 0.2, r, notes := [])
    assert st.position == CASH and st.sqqq_cooldown == r["sqqq_cooldown_days"] and "cap" in notes[0]
    st2 = apply_rules(st, SHORT, 0.2, r, [])
    assert st2.position == CASH                                                   # cooldown blocks re-entry
    st3 = apply_rules(PathState(LONG, 3, 0, 0.8), SHORT, 0.2, r, [])
    assert st3.position == CASH                                                   # flip via cash
    st4 = apply_rules(PathState(), LONG, r["min_weight"] / 2, r, [])
    assert st4.position == CASH and st4.weight == 0                               # tiny size -> cash
    st5 = apply_rules(PathState(), LONG, float("nan"), r, [])
    assert st5.position == CASH


def test_live_equals_backtest_last_row(cfg, big):
    """The live runner takes the last row of the same path: the path on data up to t must equal
    row t of the path on longer data (parity between live and backtest)."""
    inp_full = make_inputs(big, cfg)
    full = run_path("accelerator", inp_full, cfg)
    for cut in (900, 1250, 1599):
        live = run_path("accelerator", make_inputs({s: df.iloc[:cut + 1] for s, df in big.items()}, cfg), cfg)
        assert live.iloc[-1]["state"] == full.iloc[cut]["state"]
        assert np.isclose(live.iloc[-1]["w_TQQQ"], full.iloc[cut]["w_TQQQ"])


def test_accelerator_brake_engages_in_drawdowns(cfg, big):
    tight = dict(cfg["strategies"]["accelerator"], dd_cap=0.10, dd_soft=0.2)
    a = run_path("accelerator", make_inputs(big, cfg), cfg, tight)
    braked = a["reason"].str.contains("drawdown brake")
    assert braked.any()
    loose = run_path("accelerator", make_inputs(big, cfg), cfg,
                     dict(tight, dd_cap=0.99, dd_soft=0.99, dd_floor_mult=1.0))
    assert (a.loc[braked, "w_TQQQ"] <= loose.loc[braked, "w_TQQQ"] + 1e-12).all()


def test_sentinel_vol_targeting(cfg, paths):
    s = paths["sentinel"]
    # vol targeting: sentinel holds less TQQQ than the all-in benchmark on average when invested
    assert s.loc[s["state"] == LONG, "w_TQQQ"].mean() < 0.95
    assert to_target_weights(s, cfg).columns.tolist() == ["TQQQ", "SQQQ"]


@pytest.mark.parametrize("name", ["sentinel", "accelerator", "anchor"])
def test_perturbations_are_3x3_and_valid(cfg, name):
    ps = perturbations(name, cfg)
    assert len(ps) == 9 and len({lab for lab, _ in ps}) == 9


def _sel_cfg(cfg):
    return cfg.copy_with({"validation.bootstrap_samples": 300})


def test_selection_prefers_eligible_and_falls_back(cfg):
    c = _sel_cfg(cfg)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    bench = pd.Series(rng.normal(0.0006, 0.03, 1500), idx)
    good = bench * 0.5 + 0.0007 + pd.Series(rng.normal(0, 0.004, 1500), idx)
    bad = bench - 0.0008 + pd.Series(rng.normal(0, 0.01, 1500), idx)

    def summ(r):
        eq = (1 + r).cumprod()
        return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "MaxDD": float((eq / eq.cummax() - 1).min())}

    ev = {"good": evaluate("good", good, summ(good), bench, summ(bench), [1.0, 1.1], c),
          "bad": evaluate("bad", bad, summ(bad), bench, summ(bench), [0.1, 0.2], c)}
    assert ev["good"]["risk_adjusted"]["eligible"] and not ev["bad"]["risk_adjusted"]["eligible"]
    assert select(ev, "risk_adjusted")["active"] == "good"
    assert select({"bad": ev["bad"]}, "risk_adjusted")["active"] == FALLBACK
    no_nb = evaluate("x", good, summ(good), bench, summ(bench), None, c)
    assert not no_nb["robust"]["eligible"]                                       # robustness needs evidence
    with pytest.raises(ValueError):
        select(ev, "yolo")


def test_config_rejects_bad_strategy_settings(cfg):
    with pytest.raises(ConfigError):
        cfg.copy_with({"selection.objective": "moon"})
    with pytest.raises(ConfigError):
        cfg.copy_with({"selection.candidates": ["sentinel", "nope"]})
    with pytest.raises(ConfigError):
        cfg.copy_with({"strategies.accelerator": dict(cfg["strategies"]["accelerator"], max_weight=2.0)})
    with pytest.raises(ConfigError):
        cfg.copy_with({"strategies.sentinel": dict(cfg["strategies"]["sentinel"], short_cap=0.9)})


def test_csv_schema_migration(tmp_path):
    p = tmp_path / "signals.csv"
    p.write_text("signal_date,regime,p_up,p_hivol,model_mode,state,w_TQQQ,w_SQQQ,target_symbol,target_qty,halted,reason\n"
                 "2024-01-02,BULL,,,RULE,TQQQ,1.0,0.0,TQQQ,10,False,old row\n")
    StateStore(tmp_path).append("signals.csv", {"signal_date": "2024-01-03", "strategy": "anchor", "reason": "new"})
    df = pd.read_csv(p)
    assert list(df["reason"]) == ["old row", "new"] and df["strategy"].iloc[1] == "anchor"
