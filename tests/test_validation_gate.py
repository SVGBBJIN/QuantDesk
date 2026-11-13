import numpy as np
import pandas as pd
import pytest

from tqqq_desk.data import synthetic_bars
from tqqq_desk.features import build_features, make_labels
from tqqq_desk.gate import evaluate_gate
from tqqq_desk.validation import oos_scores, walk_forward_predict, walk_forward_splits


@pytest.mark.parametrize("h,emb", [(5, 0), (5, 5), (10, 3)])
def test_purged_embargoed_splits(h, emb):
    n = 1000
    prev_test_end = None
    for tr, te in walk_forward_splits(n, 300, 63, h, emb):
        assert tr[0] == 0 and tr.max() < te.min()                      # expanding, strictly before
        # label window of last train row [i+1, i+h] ends before test start minus embargo
        assert tr.max() + h < te.min() - emb
        assert te.min() - tr.max() == h + emb + 1
        if prev_test_end is not None:
            assert te.min() == prev_test_end                           # contiguous OOS coverage
        prev_test_end = te.max() + 1
    assert prev_test_end == n


def _small_cfg(cfg):
    return cfg.copy_with({"validation.min_train": 300, "validation.test_block": 126,
                          "model.lgbm": dict(cfg["model"]["lgbm"], n_estimators=60)})


def test_leakage_canary_random_walk(cfg):
    """On a driftless i.i.d. random walk the OOS AUC for direction must be ~0.5.
    Any feature/label leakage would push it well above."""
    c = _small_cfg(cfg)
    b = synthetic_bars(1300, seed=5, regime_switching=False)
    f = build_features(b, c, include_calendar=False)
    y = make_labels(b["QQQ"]["close"], 5, 0.3)
    s = oos_scores(walk_forward_predict(f, y, c), y)
    assert abs(s["p_up_auc"] - 0.5) < 0.06, s


def test_positive_control_detects_real_signal(cfg):
    """A feature genuinely (causally) related to the label must be picked up."""
    c = _small_cfg(cfg)
    rng = np.random.default_rng(1)
    n = 1300
    idx = pd.bdate_range("2015-01-01", periods=n)
    f = pd.DataFrame(rng.normal(size=(n, 4)), idx, columns=["a", "b", "c", "sig"])
    y = pd.DataFrame(index=idx)
    y["y_up"] = (f["sig"] + 0.8 * rng.normal(size=n) > 0).astype(float)
    y["y_hivol"] = (rng.random(n) < 0.2).astype(float)
    p = walk_forward_predict(f, y, c)
    s = oos_scores(p, y)
    assert s["p_up_auc"] > 0.75
    # calibration: mean predicted prob close to realised frequency
    assert abs(p["p_up"].mean() - y["y_up"].reindex(p.index).mean()) < 0.05


def test_gate_pass_and_fail(cfg):
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.0004, 0.02, 1500))
    good = bench * 0.6 + 0.0008 + rng.normal(0, 0.003, 1500)
    bad = bench - 0.0006 + rng.normal(0, 0.01, 1500)                  # same risk, less return
    g1 = evaluate_gate(good, bench, cfg)
    assert g1["passed"] and g1["mode"] == "ML"
    g2 = evaluate_gate(bad, bench, cfg)
    assert not g2["passed"] and g2["mode"] == "RULE" and "RULE-BASED" in g2["message"]


def test_plateau_prefers_stable_interior_over_spike():
    from tqqq_desk.sensitivity import plateau
    g = pd.DataFrame(np.full((5, 5), 0.5), index=pd.Index([1, 2, 3, 4, 5], name="a"),
                     columns=pd.Index([1, 2, 3, 4, 5], name="b"))
    g.iloc[0, 4] = 2.0                       # isolated spike on the edge
    g.iloc[2:5, 0:3] = 0.9                   # broad plateau
    g.iloc[4, 0] = 0.95
    pl = plateau(g)
    assert pl["best_point"]["a"] == 1 and pl["best_point"]["b"] == 5
    assert (pl["plateau_center"]["a"], pl["plateau_center"]["b"]) == (4, 2)
