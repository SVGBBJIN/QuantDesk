import datetime as dt

import pandas as pd

from tqqq_desk import mktcal
from tqqq_desk.data import drop_incomplete, staleness_report, synthetic_bars


def test_holiday_and_half_day():
    assert not mktcal.is_session(dt.date(2024, 12, 25))
    assert mktcal.is_session(dt.date(2024, 12, 24)) and mktcal.is_half_day(dt.date(2024, 12, 24))
    assert mktcal.is_half_day(dt.date(2024, 11, 29))       # day after Thanksgiving
    assert not mktcal.is_half_day(dt.date(2024, 11, 27))


def test_last_completed_session():
    # Fri 2024-07-05 20:00 ET -> Friday is complete
    now = pd.Timestamp("2024-07-05 20:00", tz="America/New_York")
    assert mktcal.last_completed_session(now) == dt.date(2024, 7, 5)
    # Fri 15:00 ET -> Thursday Jul 4 is a holiday -> Wed Jul 3 (a half day, closes 13:00)
    now = pd.Timestamp("2024-07-05 15:00", tz="America/New_York")
    assert mktcal.last_completed_session(now) == dt.date(2024, 7, 3)
    # half day: Jul 3 2024 closes 13:00 ET; at 13:30 it's complete
    assert mktcal.last_completed_session(pd.Timestamp("2024-07-03 13:30", tz="America/New_York")) == dt.date(2024, 7, 3)
    assert mktcal.next_session(dt.date(2024, 7, 3)) == dt.date(2024, 7, 5)


def test_days_to_month_end():
    idx = pd.DatetimeIndex(["2024-05-29", "2024-05-30", "2024-05-31", "2024-06-03"])
    assert list(mktcal.trading_days_to_month_end(idx)) == [2.0, 1.0, 0.0, 18.0]  # June 2024 has 19 sessions (Juneteenth)


def test_staleness_and_incomplete():
    b = synthetic_bars(300, seed=1)
    last = b["QQQ"].index[-1].date()
    assert staleness_report(b, ["QQQ", "TQQQ"], last) == {}
    trimmed = drop_incomplete(b, b["QQQ"].index[-2].date())
    probs = staleness_report(trimmed, ["QQQ", "NOPE"], last)
    assert "QQQ" in probs and probs["NOPE"] == "missing"


def test_synthetic_leveraged_decay_is_real():
    b = synthetic_bars(1500, seed=2)
    q = b["QQQ"]["close"].pct_change().dropna()
    t = b["TQQQ"]["close"].pct_change().dropna()
    # daily-reset leverage: daily returns ~3x, but long-run growth < 3x compounded growth
    assert (t / q).median() > 2.5
    naive = (1 + 3 * q).prod()
    assert t.add(1).prod() < naive
