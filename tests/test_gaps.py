"""Tests for calendar-aware gap filling."""

import pandas as pd
import pytest

from src.market_data.cleaning import clean_ohlcv, expected_timestamps, forward_fill_gaps
from tests.factories import make_ohlcv


def nyse_daily(start='2024-01-02', end='2024-02-29'):
    """Daily equity bars stamped like Alpaca: midnight New York, in UTC."""
    ts = expected_timestamps(start, end, '1d', 'equity')
    return make_ohlcv(timestamps=ts).assign(symbol='AAPL')


def test_nyse_calendar_skips_weekends_and_holidays():
    ts = expected_timestamps('2024-01-01', '2024-01-31', '1d', 'equity')
    days = set(ts.tz_convert('America/New_York').date)

    assert len(ts) == 21  # 23 weekdays minus New Year's Day and MLK Day
    assert pd.Timestamp('2024-01-01').date() not in days  # New Year's Day
    assert pd.Timestamp('2024-01-15').date() not in days  # MLK Day
    assert pd.Timestamp('2024-01-06').date() not in days  # Saturday
    assert ts[0] == pd.Timestamp('2024-01-02T05:00Z')     # midnight New York


def test_complete_equity_series_is_unchanged():
    df = nyse_daily()
    out = forward_fill_gaps(df)
    assert len(out) == len(df)
    assert not out['was_forward_filled'].any()


def test_missing_session_is_filled_with_flat_zero_volume_bar():
    df = nyse_daily()
    missing = df.index[10]
    previous_close = df.loc[missing - 1, 'close']
    out = forward_fill_gaps(df.drop(index=missing))

    filled = out[out['was_forward_filled']]
    assert len(out) == len(df) and len(filled) == 1
    bar = filled.iloc[0]
    assert bar['timestamp'] == df.loc[missing, 'timestamp']
    assert bar[['open', 'high', 'low', 'close']].tolist() == [previous_close] * 4
    assert bar['volume'] == 0.0
    assert bar['symbol'] == 'AAPL'


def test_gap_longer_than_limit_is_left_empty():
    df = nyse_daily()
    out = forward_fill_gaps(df.drop(index=range(10, 16)), max_gap_bars=5)

    assert len(out) == len(df)
    assert not out['was_forward_filled'].any()
    assert out['close'].isna().sum() == 6


def test_crypto_series_is_continuous():
    df = make_ohlcv(48, freq='h')
    out = forward_fill_gaps(df.drop(index=[10, 11]), interval='1h', market='crypto')
    assert len(out) == 48
    assert out['was_forward_filled'].sum() == 2


def test_bars_off_the_timeline_are_kept():
    df = nyse_daily()
    saturday = pd.DataFrame([{**df.iloc[3].to_dict(), 'timestamp': pd.Timestamp('2024-01-06T05:00Z')}])
    out = forward_fill_gaps(pd.concat([df, saturday], ignore_index=True))
    assert len(out) == len(df) + 1


def test_intraday_equity_needs_explicit_timeline():
    with pytest.raises(NotImplementedError):
        forward_fill_gaps(make_ohlcv(10, freq='5min'), interval='5m', market='equity')


def test_filled_bars_are_flagged_stale_by_cleaning():
    df = nyse_daily()
    out = clean_ohlcv(forward_fill_gaps(df.drop(index=30)))
    assert out.loc[out['was_forward_filled'], 'is_stale'].all()
