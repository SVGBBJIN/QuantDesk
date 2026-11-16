"""Rule-based strategy family. Each strategy is designed for one objective:

  sentinel     risk-adjusted   (Sharpe / Calmar): trend-vote score x volatility targeting, crash guard,
                                small capped SQQQ sleeve only when every trend vote is bearish
  accelerator  max return under a drawdown cap: full-size trend following with a fast re-entry path,
                                volatility-scaled leverage and an in-trade drawdown brake; larger SQQQ
                                sleeve in confirmed bear markets
  anchor       robustly beat "TQQQ if QQQ > SMA200": the same idea, made less fragile: an ensemble of
                                SMA lengths (150/200/250) with entry/exit bands, mild vol scaling and a
                                crash cut. Long/cash only
  sma200       the benchmark itself; used as the safe fallback when nothing passes validation

All strategies are causal: the decision for row t uses only bars dated <= t, and is executed at the
open of t+1 by the backtest engine (and by market-on-open orders live). They run as a single loop
over history so path state (holding days, SQQQ cooldown, hysteresis, drawdown brake) is identical in
backtest and live: the live runner simply takes the last row of the same path.

Shared position rules (tested): never TQQQ and SQQQ together, a TQQQ<->SQQQ flip spends one session
in cash, SQQQ is held at most `sqqq_max_days` then a cooldown applies, weights below `min_weight`
mean cash, missing inputs mean cash."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import qqq_features

CASH, LONG, SHORT = "CASH", "TQQQ", "SQQQ"
COLUMNS = ["state", "w_TQQQ", "w_SQQQ", "days_held", "sqqq_cooldown", "score", "reason"]


# ----------------------------------------------------------------------------- inputs
@dataclass
class Inputs:
    idx: pd.DatetimeIndex
    cols: dict[str, np.ndarray]          # QQQ-derived feature columns + ETF stats, aligned to idx

    def __getitem__(self, k: str) -> np.ndarray:
        return self.cols[k]


def make_inputs(bars: dict[str, pd.DataFrame], cfg) -> Inputs:
    sig = cfg["data"]["signal_symbol"]
    long_sym, short_sym = cfg["data"]["trade_symbols"]["long"], cfg["data"]["trade_symbols"]["short"]
    q = bars[sig]
    c = q["close"]
    f = qqq_features(q)
    for n in (150, 250):
        f[f"px_vs_sma{n}"] = c / c.rolling(n, min_periods=n).mean() - 1
    f["close"] = c
    lb = cfg["sizing"]["vol_lookback"]
    for name, sym in (("T", long_sym), ("S", short_sym)):
        ec = bars[sym]["close"].reindex(q.index)
        lr = np.log(ec).diff()
        f[f"vol_{name}"] = lr.rolling(lb, min_periods=lb).std() * np.sqrt(252)
        f[f"ret_{name}"] = ec.pct_change().fillna(0.0)   # close[t]/close[t-1]-1, for the shadow equity
    f = f.replace([np.inf, -np.inf], np.nan)
    return Inputs(f.index, {k: f[k].to_numpy(float) for k in f.columns})


# ----------------------------------------------------------------------------- shared position rules
@dataclass
class PathState:
    position: str = CASH
    days_held: int = 0
    sqqq_cooldown: int = 0
    weight: float = 0.0
    extra: dict = field(default_factory=dict)   # strategy-specific memory (hysteresis, brake, ...)


def apply_rules(prev: PathState, desired: str, w: float, rules: dict, notes: list[str]) -> PathState:
    cooldown = max(0, prev.sqqq_cooldown - 1)
    if desired != CASH and (not np.isfinite(w) or w < rules["min_weight"]):
        notes.append(f"size {0 if not np.isfinite(w) else w:.2f} below min weight -> cash")
        desired, w = CASH, 0.0
    if desired == SHORT:
        if prev.position == SHORT and prev.days_held >= rules["sqqq_max_days"]:
            desired, w = CASH, 0.0
            cooldown = rules["sqqq_cooldown_days"]
            notes.append(f"SQQQ holding cap of {rules['sqqq_max_days']} days reached -> cash, {cooldown}-day cooldown")
        elif prev.position != SHORT and cooldown > 0:
            desired, w = CASH, 0.0
            notes.append(f"SQQQ cooldown active ({cooldown} days left)")
    if {prev.position, desired} == {LONG, SHORT}:
        notes.append(f"flip {prev.position}->{desired} goes via one session in cash")
        desired, w = CASH, 0.0
    if desired == CASH:
        w = 0.0
    days = prev.days_held + 1 if desired == prev.position and desired != CASH else (1 if desired != CASH else 0)
    return PathState(desired, days, cooldown, float(w), dict(prev.extra))


def _ok(*xs) -> bool:
    return all(np.isfinite(x) for x in xs)


def _vt(target: float, vol: float, cap: float) -> float:
    return float(min(cap, target / vol)) if np.isfinite(vol) and vol > 0 else 0.0


def _pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


# ----------------------------------------------------------------------------- sentinel
def sentinel_step(x: dict, prev: PathState, p: dict) -> tuple[str, float, float, list[str]]:
    need = (x["px_vs_sma50"], x["px_vs_sma100"], x["px_vs_sma200"], x["sma50_vs_sma200"], x["mom60"],
            x["mom120"], x["mom20"], x["rv10"], x["rv20"], x["rv60"], x["vol_T"])
    if not _ok(*need):
        return CASH, 0.0, np.nan, ["inputs unavailable (warm-up)"]
    votes = [x["px_vs_sma50"] > 0, x["px_vs_sma100"] > 0, x["px_vs_sma200"] > 0, x["sma50_vs_sma200"] > 0,
             x["mom60"] > 0, x["mom120"] > 0]
    score = float(np.mean(votes))
    thr = p["exit_score"] if prev.position == LONG else p["enter_score"]
    crash = x["rv10"] / x["rv60"] > p["crash_ratio"] and x["px_vs_sma50"] < 0
    notes = [f"trend score {sum(votes)}/6"]
    if score >= thr and not crash:
        w = _vt(p["target_vol"], x["vol_T"], p["max_weight"]) * score
        notes.append(f">= {thr:.2f}: TQQQ at vol target {p['target_vol']:.0%} x score")
        if x["rv20"] > p["hv_cut"]:
            w *= 0.5
            notes.append(f"20d vol {x['rv20']:.0%} > {p['hv_cut']:.0%}: half size")
        return LONG, w, score, notes
    if crash:
        notes.append(f"crash guard: 10d/60d vol ratio {x['rv10'] / x['rv60']:.1f} > {p['crash_ratio']} below SMA50")
    if (p["allow_sqqq"] and score == 0 and x["mom20"] < 0 and x["px_vs_sma200"] < p["short_band"]
            and x["rv20"] < p["short_max_vol"] and np.isfinite(x["vol_S"])):
        w = _vt(p["short_target_vol"], x["vol_S"], p["short_cap"])
        notes.append(f"all votes bearish, QQQ {_pct(x['px_vs_sma200'])} vs SMA200, 20d mom < 0: SQQQ")
        return SHORT, w, score, notes
    notes.append(f"< {thr:.2f} -> cash" if not crash else "-> cash")
    return CASH, 0.0, score, notes


# ----------------------------------------------------------------------------- accelerator
def accelerator_step(x: dict, prev: PathState, p: dict) -> tuple[str, float, float, list[str]]:
    need = (x["px_vs_sma200"], x["px_vs_sma50"], x["sma50_vs_sma200"], x["sma50_slope"], x["mom20"],
            x["mom60"], x["rv20"], x["vol_T"])
    if not _ok(*need):
        return CASH, 0.0, np.nan, ["inputs unavailable (warm-up)"]
    ex = prev.extra
    d200 = x["px_vs_sma200"]
    in_trend = ex.get("in_trend", False)
    in_trend = d200 > -p["band_out"] if in_trend else d200 > p["band_in"]
    ex["in_trend"] = in_trend
    fast = (p["fast_reentry"] and x["px_vs_sma50"] > 0 and x["sma50_slope"] > 0 and x["mom20"] > 0
            and x["rv20"] < p["fast_max_vol"])
    notes = []
    if in_trend or fast:
        w = _vt(p["target_vol"], x["vol_T"], p["max_weight"])
        notes.append(f"QQQ {_pct(d200)} vs SMA200 (trend)" if in_trend else
                     "fast re-entry: above rising SMA50 with positive 20d momentum")
        # portfolio drawdown brake: shadow equity of this strategy vs a slowly decaying high-water mark
        dd = ex.get("dd", 0.0)
        soft, cap = p["dd_soft"] * p["dd_cap"], p["dd_cap"]
        mult = float(np.clip(1 - max(0.0, dd - soft) / (cap - soft), p["dd_floor_mult"], 1.0))
        if mult < 1:
            notes.append(f"drawdown brake: strategy drawdown {dd:.0%} -> size x{mult:.2f}")
        return LONG, w * mult, 1.0, notes
    if (p["allow_sqqq"] and d200 < p["short_band"] and x["sma50_vs_sma200"] < 0 and x["mom20"] < 0
            and x["mom60"] < 0 and x["rv20"] < p["short_max_vol"] and np.isfinite(x["vol_S"])):
        w = _vt(p["short_target_vol"], x["vol_S"], p["short_cap"])
        notes.append(f"confirmed bear: QQQ {_pct(d200)} vs SMA200, SMA50<SMA200, 20d & 60d momentum < 0: SQQQ")
        return SHORT, w, -1.0, notes
    notes.append(f"QQQ {_pct(d200)} vs SMA200, no trend and no fast re-entry -> cash")
    return CASH, 0.0, 0.0, notes


def _accelerator_update(st: PathState, x: dict) -> None:
    """After today's close: update the shadow equity (what yesterday's position earned today) and the
    drawdown vs a high-water mark that decays with half-life `dd_halflife` sessions, so the brake
    releases over time instead of locking the strategy out for good."""
    ex = st.extra
    r = x["ret_T"] if np.isfinite(x["ret_T"]) else 0.0
    rs = x["ret_S"] if np.isfinite(x["ret_S"]) else 0.0
    eq = ex.get("eq", 1.0) * (1 + ex.get("held_w", 0.0) * r + ex.get("held_ws", 0.0) * rs)
    decay = 0.5 ** (1.0 / ex.get("halflife", 126))
    peak = max(eq, ex.get("peak", 1.0) * decay + eq * (1 - decay))
    ex["eq"], ex["peak"], ex["dd"] = eq, peak, 1 - eq / peak


# ----------------------------------------------------------------------------- anchor
def anchor_step(x: dict, prev: PathState, p: dict) -> tuple[str, float, float, list[str]]:
    lens = p["sma_lengths"]
    need = [x[f"px_vs_sma{n}"] for n in lens] + [x["px_vs_sma50"], x["rv10"], x["rv60"], x["vol_T"]]
    if not _ok(*need):
        return CASH, 0.0, np.nan, ["inputs unavailable (warm-up)"]
    ex = prev.extra
    votes = []
    for n in lens:
        k, d = f"on{n}", x[f"px_vs_sma{n}"]
        on = d > -p["band"] if ex.get(k, False) else d > p["band"]
        ex[k] = on
        votes.append(on)
    score = float(np.mean(votes))
    notes = [f"above SMA {'/'.join(str(n) for n, v in zip(lens, votes) if v) or 'none'} (bands ±{p['band']:.0%})"]
    if score == 0:
        return CASH, 0.0, score, notes + ["-> cash"]
    w = score * _vt(p["target_vol"], x["vol_T"], p["max_weight"])
    if x["rv10"] / x["rv60"] > p["crash_ratio"] and x["px_vs_sma50"] < 0:
        w *= 0.5
        notes.append("crash cut: vol spike below SMA50 -> half size")
    return LONG, w, score, notes


# ----------------------------------------------------------------------------- sma200 (benchmark)
def sma200_step(x: dict, prev: PathState, p: dict) -> tuple[str, float, float, list[str]]:
    d = x["px_vs_sma200"]
    if not np.isfinite(d):
        return CASH, 0.0, np.nan, ["SMA200 unavailable (warm-up)"]
    if d > 0:
        return LONG, 1.0, 1.0, [f"QQQ {_pct(d)} above SMA200 -> 100% TQQQ"]
    return CASH, 0.0, 0.0, [f"QQQ {_pct(d)} below SMA200 -> cash"]


@dataclass(frozen=True)
class Spec:
    name: str
    objective: str
    step: callable
    robust: tuple            # ((param, kind, delta), ...) perturbed for the robustness neighbourhood
    update: callable = None
    description: str = ""


REGISTRY: dict[str, Spec] = {
    "sentinel": Spec("sentinel", "risk_adjusted", sentinel_step,
                     (("target_vol", "mul", 0.2), ("enter_score", "add", 0.17)),
                     description="trend-vote score x vol targeting, crash guard, small SQQQ sleeve"),
    "accelerator": Spec("accelerator", "max_return", accelerator_step,
                        (("target_vol", "mul", 0.2), ("band_in", "add", 0.01)), update=_accelerator_update,
                        description="full-size trend + fast re-entry, drawdown brake, SQQQ in confirmed bears"),
    "anchor": Spec("anchor", "robust", anchor_step,
                   (("band", "add", 0.01), ("target_vol", "mul", 0.2)),
                   description="SMA150/200/250 ensemble with bands, mild vol scaling, crash cut"),
    "sma200": Spec("sma200", "benchmark", sma200_step, (), description="TQQQ if QQQ > SMA200 else cash"),
}
RULE_STRATEGIES = tuple(REGISTRY)


def strategy_params(cfg, name: str) -> dict:
    p = dict(cfg["strategies"].get(name, {})) if name != "sma200" else {}
    return p


def position_rules(cfg) -> dict:
    d = cfg["decision"]
    return {"sqqq_max_days": d["sqqq_max_days"], "sqqq_cooldown_days": d["sqqq_cooldown_days"],
            "min_weight": cfg["sizing"]["min_weight"]}


def run_path(name: str, inp: Inputs, cfg, params: dict | None = None,
             idx: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Run strategy `name` over all of `inp` (so path state warms up on full history) and return
    rows for `idx` (default: all)."""
    spec = REGISTRY[name]
    p = params if params is not None else strategy_params(cfg, name)
    rules = position_rules(cfg)
    keys = list(inp.cols)
    arr = np.column_stack([inp.cols[k] for k in keys])
    st = PathState()
    out = []
    for i in range(len(inp.idx)):
        x = dict(zip(keys, arr[i]))
        if spec.update is not None:
            st.extra["held_w"] = st.weight if st.position == LONG else 0.0
            st.extra["held_ws"] = st.weight if st.position == SHORT else 0.0
            st.extra["halflife"] = p.get("dd_halflife", 126)
        desired, w, score, notes = spec.step(x, st, p)
        st = apply_rules(st, desired, w, rules, notes)
        if spec.update is not None:
            spec.update(st, x)
        out.append((st.position, st.weight if st.position == LONG else 0.0,
                    st.weight if st.position == SHORT else 0.0, st.days_held, st.sqqq_cooldown, score,
                    f"{name}: " + " | ".join(notes)))
    df = pd.DataFrame(out, index=inp.idx, columns=COLUMNS)
    return df if idx is None else df.reindex(idx)


def perturbations(name: str, cfg) -> list[tuple[str, dict]]:
    """3x3 neighbourhood of the configured parameters along the strategy's two robustness axes."""
    spec = REGISTRY[name]
    base = strategy_params(cfg, name)
    if len(spec.robust) < 2:
        return [("base", base)]
    (a, ka, da), (b, kb, db) = spec.robust[:2]

    def vals(v, kind, dlt):
        return [v * (1 - dlt), v, v * (1 + dlt)] if kind == "mul" else [v - dlt, v, v + dlt]

    out = []
    for va in vals(base[a], ka, da):
        for vb in vals(base[b], kb, db):
            q = dict(base)
            q[a], q[b] = va, vb
            if "exit_score" in q and q.get("enter_score", 1) < q["exit_score"]:
                q["exit_score"] = q["enter_score"]
            out.append((f"{a}={va:.3g}, {b}={vb:.3g}", q))
    return out


def to_target_weights(sig: pd.DataFrame, cfg) -> pd.DataFrame:
    ts = cfg["data"]["trade_symbols"]
    return pd.DataFrame({ts["long"]: sig["w_TQQQ"].fillna(0.0), ts["short"]: sig["w_SQQQ"].fillna(0.0)},
                        index=sig.index)
