"""Build a LONG-HISTORY PROXY dataset (1999-2018) for stress-testing strategies.

Alpaca daily data only starts around 2016, which contains very few bear markets. This script builds
a 20-year proxy from the real NASDAQ Composite daily OHLC bundled with the `arch` package
(pip install arch), which covers the dot-com crash (-78%), 2008, 2011, 2015-16 and 2018 Q4:

  QQQ  := NASDAQ Composite (price index, no dividends; close enough for timing research)
  SPY  := S&P 500 index
  TQQQ := daily-reset +3x of the index, applied separately to the overnight gap and the intraday
          move, minus 0.95%/yr expense ratio, minus 2x the risk-free rate (financing the leverage)
          and a 0.5%/yr swap spread
  SQQQ := daily-reset -3x, minus 0.95%/yr expense ratio and 0.5%/yr swap spread, plus 2x the
          risk-free rate (collateral + short-swap financing, conservatively under-credited)
Risk-free rate: monthly Fama-French RF (also bundled with `arch`).

THIS IS A PROXY. Real TQQQ/SQQQ track their index with small extra errors; the Composite is not the
Nasdaq-100. Use it to see how a rule behaves through crashes, never as the validation that unlocks
trading (the report marks it as non-Alpaca data and the gate cannot enable anything from it).

Usage:
    pip install arch
    python scripts/make_proxy_history.py --out data/proxy
    python -m tqqq_desk.report --csv-dir data/proxy --out reports/proxy --artifacts /tmp/proxy_art
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ER, SPREAD = 0.0095, 0.005


def _arch_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    spec = importlib.util.find_spec("arch")  # locate without importing (no scipy/statsmodels needed)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("the `arch` package is required: pip install arch (or pass --arch-dir)")
    return Path(list(spec.submodule_search_locations)[0]) / "data"


def _ohlc(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.index = pd.DatetimeIndex(pd.to_datetime(d["Date"], format="%m/%d/%Y"), name="date")
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype(float)
    return d[~d.index.duplicated()].sort_index()


def _rf_daily(path: Path, idx: pd.DatetimeIndex) -> pd.Series:
    ff = pd.read_csv(path)
    ff.index = pd.PeriodIndex(ff["Date"].astype(str), freq="M")
    rf_month = ff["RF"] / 100.0  # monthly simple rate
    per = idx.to_period("M")
    rf = rf_month.reindex(per).ffill().to_numpy()
    days = pd.Series(1, idx).groupby(per).transform("count").to_numpy()
    return pd.Series(rf / days, idx)


def leveraged(q: pd.DataFrame, k: float, daily_drag: np.ndarray) -> pd.DataFrame:
    c_prev = q["close"].shift().fillna(q["open"]).to_numpy()
    gap = q["open"].to_numpy() / c_prev - 1
    intra = q["close"].to_numpy() / q["open"].to_numpy() - 1
    hi_rel = q["high"].to_numpy() / q["open"].to_numpy() - 1
    lo_rel = q["low"].to_numpy() / q["open"].to_numpy() - 1
    close = 100 * np.cumprod((1 + k * gap) * (1 + k * intra) * (1 - daily_drag))
    open_ = np.r_[100.0, close[:-1]] * (1 + k * gap)
    a, b = open_ * (1 + k * hi_rel), open_ * (1 + k * lo_rel)
    high = np.maximum.reduce([a, b, open_, close])
    low = np.minimum.reduce([a, b, open_, close])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1e7}, index=q.index)


def build(arch_dir: Path) -> dict[str, pd.DataFrame]:
    q = _ohlc(arch_dir / "nasdaq" / "nasdaq.csv.gz")
    spx = _ohlc(arch_dir / "sp500" / "sp500.csv.gz").reindex(q.index).ffill()
    rf = _rf_daily(arch_dir / "frenchdata" / "frenchdata.csv.gz", q.index).to_numpy()
    fee = (ER + SPREAD) / 252
    return {"QQQ": q, "SPY": spx,
            "TQQQ": leveraged(q, 3.0, fee + 2 * rf),
            "SQQQ": leveraged(q, -3.0, fee - 2 * rf)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/proxy")
    ap.add_argument("--arch-dir", default=None, help="path to arch/data (default: locate installed arch)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sym, df in build(_arch_data_dir(a.arch_dir)).items():
        df.to_csv(out / f"{sym}.csv")
        print(f"{sym}: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} bars), "
              f"total {df['close'].iloc[-1] / df['close'].iloc[0] - 1:+.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
