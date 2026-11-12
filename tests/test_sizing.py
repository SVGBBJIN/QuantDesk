import numpy as np
import pandas as pd
import pytest

from tqqq_desk.config import ConfigError, load_config
from tqqq_desk.sizing import etf_risk_inputs, final_weight, kelly_weight, vol_target_weight


def test_vol_target():
    assert vol_target_weight(0.90, 0.45) == pytest.approx(0.5)
    assert vol_target_weight(0.0, 0.45) == 0.0 and vol_target_weight(np.nan, 0.45) == 0.0


def test_kelly_math():
    # mu = (2*0.6-1)*0.06 = 0.012, var = 0.0064 -> f* = 1.875, quarter Kelly = 0.46875
    assert kelly_weight(0.6, 0.06, 0.0064, 0.25) == pytest.approx(0.46875)
    assert kelly_weight(0.5, 0.06, 0.0064, 0.25) == 0.0
    assert kelly_weight(0.4, 0.06, 0.0064, 0.25) == 0.0              # no negative bets
    with pytest.raises(ValueError):
        kelly_weight(0.6, 0.06, 0.0064, 0.5)                          # > 0.25 Kelly refused


def test_final_weight_takes_min_and_caps(cfg):
    s = cfg["sizing"]
    # vol target 0.45/0.6=0.75, kelly 0.46875 -> 0.46875
    assert final_weight("TQQQ", 0.6, 0.6, 0.06, 0.0064, s) == pytest.approx(0.46875)
    # strong edge -> kelly huge, vol target 0.45/0.3=1.5 -> cap 1.0
    assert final_weight("TQQQ", 0.9, 0.3, 0.06, 0.0064, s) == pytest.approx(1.0)
    # SQQQ capped at 30%
    assert final_weight("SQQQ", 0.9, 0.3, 0.06, 0.0064, s) == pytest.approx(0.30)
    # tiny -> below min weight -> 0
    assert final_weight("TQQQ", 0.505, 0.6, 0.06, 0.0064, s) == 0.0
    # rule mode: no probability -> vol target + cap
    assert final_weight("TQQQ", None, 0.9, np.nan, np.nan, s) == pytest.approx(0.5)


def test_config_rejects_kelly_above_quarter(tmp_path):
    import yaml
    from tqqq_desk.config import DEFAULT_PATH
    c = yaml.safe_load(open(DEFAULT_PATH))
    c["sizing"]["kelly_fraction"] = 0.5
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ConfigError):
        load_config(p)


def test_risk_inputs_causal():
    c = pd.Series(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.03, 400))) * 50)
    a = etf_risk_inputs(c, 5, 20, 252)
    b = etf_risk_inputs(c.iloc[:300], 5, 20, 252)
    pd.testing.assert_frame_equal(a.iloc[:300], b)
