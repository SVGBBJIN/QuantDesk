"""Predictive layer: regularized shallow LightGBM + logistic-regression baseline, each calibrated
on a purged, time-ordered holdout at the end of its training window, then averaged."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {"p_up": "y_up", "p_hivol": "y_hivol"}


def make_lgbm(cfg):
    m = cfg["model"]
    return LGBMClassifier(**m["lgbm"], random_state=m["seed"], verbose=-1, n_jobs=1)


def make_logreg(cfg):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=cfg["model"]["logreg_C"], max_iter=2000))


class Calibrator:
    def __init__(self, method: str):
        self.method, self.f = method, None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if len(np.unique(y)) < 2 or len(y) < 20:
            return self  # identity: not enough information to calibrate
        if self.method == "isotonic":
            self.f = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        else:
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            self.f = LogisticRegression(C=1e4).fit(z.reshape(-1, 1), y)
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self.f is None:
            return p
        if self.method == "isotonic":
            return self.f.predict(p)
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return self.f.predict_proba(z.reshape(-1, 1))[:, 1]


class CalibratedModel:
    """Fit on the first part of the window, calibrate on the last part, with a purge gap of
    `gap` rows so calibration labels never overlap fitting labels."""

    def __init__(self, factory, cfg, gap: int):
        self.factory, self.cfg, self.gap = factory, cfg, gap

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CalibratedModel":
        n = len(X)
        n_cal = int(n * self.cfg["model"]["calib_fraction"])
        fit_end = n - n_cal - self.gap
        self.const = None
        if fit_end < 50 or len(np.unique(y.iloc[:fit_end])) < 2:
            self.const = float(y.mean())
            return self
        self.model = self.factory(self.cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X.iloc[:fit_end], y.iloc[:fit_end])
            p_cal = self.model.predict_proba(X.iloc[n - n_cal:])[:, 1]
        self.cal = Calibrator(self.cfg["model"]["calibration"]).fit(p_cal, y.iloc[n - n_cal:].to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.const is not None:
            return np.full(len(X), self.const)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.cal(self.model.predict_proba(X)[:, 1])


class Ensemble:
    """For each target: calibrated LGBM + calibrated logistic, probability-averaged."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.h = cfg["labels"]["horizon"]

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "Ensemble":
        self.cols = list(X.columns)
        self.models = {}
        for out, tgt in TARGETS.items():
            mask = Y[tgt].notna().to_numpy()
            Xt, yt = X[mask], Y[tgt][mask].astype(int)
            self.models[out] = {
                "lgbm": CalibratedModel(make_lgbm, self.cfg, self.h).fit(Xt, yt),
                "logreg": CalibratedModel(make_logreg, self.cfg, self.h).fit(Xt, yt),
            }
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.cols]
        out = {}
        for o, ms in self.models.items():
            parts = {k: m.predict(X) for k, m in ms.items()}
            for k, v in parts.items():
                out[f"{o}_{k}"] = v
            out[o] = np.mean(list(parts.values()), axis=0)
        return pd.DataFrame(out, index=X.index)

    def feature_importance(self) -> pd.Series:
        m = self.models["p_up"]["lgbm"]
        if m.const is not None:
            return pd.Series(dtype=float)
        return pd.Series(m.model.feature_importances_, index=self.cols).sort_values(ascending=False)
