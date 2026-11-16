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
    validate_strategies(cfg)
    if cfg["risk"]["kill_action"] not in ("flatten", "freeze"):
        raise ConfigError("risk.kill_action must be flatten or freeze")


def validate_strategies(cfg: Cfg) -> None:
    from .selection import OBJECTIVES
    from .strategies import REGISTRY
    sel = cfg["selection"]
    if sel["objective"] not in OBJECTIVES:
        raise ConfigError(f"selection.objective must be one of {OBJECTIVES}")
    for c in sel["candidates"]:
        if c != "ml" and c not in REGISTRY:
            raise ConfigError(f"selection.candidates: unknown strategy {c!r} (known: ml, {', '.join(REGISTRY)})")
    if not 0 < sel["max_drawdown_cap"] < 1 or not 0.5 <= sel["min_bootstrap_prob"] < 1:
        raise ConfigError("selection.max_drawdown_cap must be in (0,1) and min_bootstrap_prob in [0.5,1)")
    for name, p in cfg["strategies"].items():
        if name not in REGISTRY:
            raise ConfigError(f"strategies.{name}: unknown strategy")
        if not 0 < p.get("max_weight", 1.0) <= 1:
            raise ConfigError(f"strategies.{name}.max_weight must be in (0, 1] (no leverage on top of 3x)")
        if p.get("short_cap", 0) > 0.5:
            raise ConfigError(f"strategies.{name}.short_cap must be <= 0.5 (SQQQ decays)")
        if p.get("target_vol", 0.5) <= 0:
            raise ConfigError(f"strategies.{name}.target_vol must be > 0")
    s = cfg["strategies"]
    if "sentinel" in s and s["sentinel"]["exit_score"] > s["sentinel"]["enter_score"]:
        raise ConfigError("strategies.sentinel.exit_score must be <= enter_score (hysteresis)")
    if "accelerator" in s:
        a = s["accelerator"]
        if not 0 < a["dd_soft"] < 1 or not 0 < a["dd_cap"] < 1 or not 0 <= a["dd_floor_mult"] <= 1:
            raise ConfigError("strategies.accelerator: dd_cap, dd_soft in (0,1), dd_floor_mult in [0,1]")
        if a["band_out"] < -a["band_in"]:
            raise ConfigError("strategies.accelerator.band_out must be >= -band_in (hysteresis)")


ENV_OVERRIDES = {"QD_OBJECTIVE": ("selection", "objective")}


def apply_env_overrides(cfg: Cfg, env=None) -> None:
    """A few choices can be made from environment variables (e.g. Vercel project settings) without a
    commit. They become part of the config (and its hash), so validation and trading must agree."""
    import os
    env = os.environ if env is None else env
    for var, (sec, key) in ENV_OVERRIDES.items():
        if env.get(var):
            cfg[sec][key] = env[var].strip()


def load_config(path: str | Path | None = None) -> Cfg:
    p = Path(path) if path else DEFAULT_PATH
    with open(p) as f:
        cfg = Cfg(yaml.safe_load(f))
    apply_env_overrides(cfg)
    validate(cfg)
    return cfg
