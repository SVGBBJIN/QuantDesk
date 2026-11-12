"""Config loading + validation. Config is a nested dict with attribute access."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
MAX_KELLY = 0.25


class Cfg(dict):
    """dict with recursive attribute access: cfg.regime.bull_enter."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def copy_with(self, overrides: dict[str, Any]) -> "Cfg":
        """Return a deep copy with dotted-key overrides, e.g. {"decision.up_enter": 0.6}."""
        new = Cfg(copy.deepcopy(dict(self)))
        for dotted, val in overrides.items():
            node = new
            *parents, leaf = dotted.split(".")
            for p in parents:
                node = node[p]
            if leaf not in node:
                raise KeyError(f"unknown config key {dotted}")
            node[leaf] = val
        validate(new)
        return new


class ConfigError(ValueError):
    pass


def validate(cfg: Cfg) -> None:
    r, d, s = cfg["regime"], cfg["decision"], cfg["sizing"]
    if not 0 < s["kelly_fraction"] <= MAX_KELLY:
        raise ConfigError(f"sizing.kelly_fraction must be in (0, {MAX_KELLY}]")
    for sym, w in s["max_weight"].items():
        if not 0 <= w <= 1:
            raise ConfigError(f"sizing.max_weight.{sym} must be in [0, 1]")
    if r["sma_long"] not in (50, 100, 200) or r["sma_short"] != 50:
        raise ConfigError("regime.sma_long must be one of 50/100/200 and sma_short 50 (features computed)")
    if r["bull_exit"] > r["bull_enter"]:
        raise ConfigError("regime.bull_exit must be <= bull_enter (hysteresis)")
    if r["bear_exit"] < r["bear_enter"]:
        raise ConfigError("regime.bear_exit must be >= bear_enter (hysteresis)")
    if r["hv_exit"] > r["hv_enter"]:
        raise ConfigError("regime.hv_exit must be <= hv_enter (hysteresis)")
    if d["up_exit"] > d["up_enter"] or d["down_exit"] > d["down_enter"]:
        raise ConfigError("decision exit thresholds must be <= entry thresholds")
    if d["down_enter"] < d["up_enter"]:
        raise ConfigError("decision.down_enter must be >= up_enter (SQQQ needs a higher bar)")
    if d["sqqq_max_days"] < 1:
        raise ConfigError("decision.sqqq_max_days must be >= 1")
    if cfg["validation"]["embargo"] < 0:
        raise ConfigError("validation.embargo must be >= 0")
    if cfg["risk"]["kill_action"] not in ("flatten", "freeze"):
        raise ConfigError("risk.kill_action must be flatten or freeze")


def load_config(path: str | Path | None = None) -> Cfg:
    p = Path(path) if path else DEFAULT_PATH
    with open(p) as f:
        cfg = Cfg(yaml.safe_load(f))
    validate(cfg)
    return cfg
