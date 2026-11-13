"""NYSE market calendar helpers (pandas_market_calendars). Handles holidays + half-days."""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

ET = "America/New_York"


@lru_cache(maxsize=1)
def _nyse():
    return mcal.get_calendar("NYSE")


@lru_cache(maxsize=64)
def schedule(start: str, end: str) -> pd.DataFrame:
    """Session schedule with tz-aware UTC market_open / market_close, index = session date."""
    s = _nyse().schedule(start_date=start, end_date=end)
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s


def _sched_around(d: dt.date, back: int = 15, fwd: int = 15) -> pd.DataFrame:
    return schedule(str(d - dt.timedelta(days=back)), str(d + dt.timedelta(days=fwd)))


def is_session(d: dt.date) -> bool:
    return pd.Timestamp(d) in _sched_around(d, 3, 3).index


def is_half_day(d: dt.date) -> bool:
    s = _sched_around(d, 3, 3)
    ts = pd.Timestamp(d)
    if ts not in s.index:
        return False
    close_et = s.loc[ts, "market_close"].tz_convert(ET)
    return (close_et.hour, close_et.minute) < (16, 0)


def last_completed_session(now: pd.Timestamp) -> dt.date:
    """Most recent session whose close is <= now (now must be tz-aware)."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must be tz-aware")
    s = _sched_around(now.date(), 20, 2)
    done = s[s["market_close"] <= now.tz_convert("UTC")]
    return done.index[-1].date()


def next_session(d: dt.date) -> dt.date:
    s = _sched_around(d, 1, 20)
    after = s.index[s.index > pd.Timestamp(d)]
    return after[0].date()


def session_open_utc(d: dt.date) -> pd.Timestamp:
    return _sched_around(d, 3, 3).loc[pd.Timestamp(d), "market_open"]


def trading_days_to_month_end(dates: pd.DatetimeIndex) -> pd.Series:
    """Number of NYSE sessions remaining in the month after each date (0 = last session).
    Uses the published exchange calendar, which is known in advance -> not lookahead."""
    if len(dates) == 0:
        return pd.Series(dtype=float)
    end = (dates.max() + pd.offsets.MonthEnd(1)).date()
    sess = schedule(str(dates.min().date()), str(end)).index
    month = sess.to_period("M")
    rank_from_end = pd.Series(1, index=sess).groupby(month).cumsum()
    count = pd.Series(1, index=sess).groupby(month).transform("sum")
    remaining = (count - rank_from_end).reindex(dates)
    return remaining.astype(float)
