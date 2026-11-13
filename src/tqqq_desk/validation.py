"""Expanding-window walk-forward with purging + embargo. No random k-fold anywhere.

For a test block starting at row s, a training row i is allowed only if its label window
[i+1, i+h] ends strictly before s - embargo, i.e. i < s - h - embargo. Because the window is
expanding and always precedes the test block, no training label can overlap test data."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .features import CALENDAR_COLS
from .models import TARGETS, Ensemble

log = logging.getLogger(__name__)


def walk_forward_splits(n: int, min_train: int, test_block: int, horizon: int, embargo: int):
    """Yield (train_idx, test_idx) as integer position arrays."""
    s = min_train + horizon + embargo
    while s < n:
        e = min(s + test_block, n)
        train_end = s - horizon - embargo
        yield np.arange(0, train_end), np.arange(s, e)
        s = e


def usable_rows(feats: pd.DataFrame) -> pd.Index:
    """Rows after warm-up: all core QQQ features present (context NaNs tolerated)."""
    core = [c for c in feats.columns if not c.startswith(("qqq_spy", "tlt", "uup", "vixy"))]
    return feats.index[feats[core].notna().all(axis=1)]


def walk_forward_predict(feats: pd.DataFrame, labels: pd.DataFrame, cfg) -> pd.DataFrame:
    v, h = cfg["validation"], cfg["labels"]["horizon"]
    rows = usable_rows(feats)
    X, Y = feats.loc[rows], labels.loc[rows]
    preds = []
    for k, (tr, te) in enumerate(walk_forward_splits(len(X), v["min_train"], v["test_block"], h, v["embargo"])):
        ens = Ensemble(cfg).fit(X.iloc[tr], Y.iloc[tr])
        p = ens.predict(X.iloc[te])
        p["fold"] = k
        p["train_end"] = X.index[tr[-1]]
        preds.append(p)
    if not preds:
        raise ValueError("not enough history for a single walk-forward fold")
    return pd.concat(preds)


def oos_scores(preds: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    out = {}
    for p, y in TARGETS.items():
        yy = labels[y].reindex(preds.index)
        m = yy.notna()
        if m.sum() < 30 or yy[m].nunique() < 2:
            continue
        pp = preds.loc[m, p].clip(1e-6, 1 - 1e-6)
        out[f"{p}_logloss"] = float(log_loss(yy[m], pp))
        out[f"{p}_auc"] = float(roc_auc_score(yy[m], pp))
        base = float(yy[m].mean())
        out[f"{p}_logloss_base"] = float(log_loss(yy[m], np.full(m.sum(), base)))
        out[f"{p}_brier"] = float(np.mean((pp - yy[m]) ** 2))
    return out


def select_calendar(feats: pd.DataFrame, labels: pd.DataFrame, cfg) -> tuple[bool, dict, pd.DataFrame]:
    """Calendar features are kept only if they lower the mean OOS log-loss of both targets.
    Returns (keep, info, OOS predictions of the chosen feature set)."""
    mode = cfg["features"]["calendar"]
    if mode in ("on", "off"):
        f = feats if mode == "on" else feats.drop(columns=CALENDAR_COLS)
        return mode == "on", {"mode": mode}, walk_forward_predict(f, labels, cfg)
    with_cal = walk_forward_predict(feats, labels, cfg)
    without = walk_forward_predict(feats.drop(columns=CALENDAR_COLS), labels, cfg)
    s1, s0 = oos_scores(with_cal, labels), oos_scores(without, labels)
    ll = lambda s: np.mean([s.get("p_up_logloss", np.nan), s.get("p_hivol_logloss", np.nan)])
    keep = bool(ll(s1) < ll(s0))
    info = {"mode": "auto", "logloss_with": ll(s1), "logloss_without": ll(s0), "kept": keep}
    log.info("calendar features %s (OOS log-loss with=%.4f without=%.4f)", "KEPT" if keep else "DROPPED", ll(s1), ll(s0))
    return keep, info, (with_cal if keep else without)
