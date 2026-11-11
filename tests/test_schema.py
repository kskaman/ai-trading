"""Tests for the canonical OHLCV schema."""

import numpy as np
import pandas as pd
import pytest

from src.market_data.schema import (
    bars_to_frame, estimate_periods_per_year, normalize_interval, ohlc_valid_mask,
    periods_per_year, require_single_symbol,
)


def _bar(ts, close=100.0, volume=1.5):
    return {'timestamp': ts, 'open': close, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': volume}


def test_bars_to_frame_sorts_dedupes_and_uses_utc_float_columns():
    bars = [
        _bar('2024-01-03T05:00:00+00:00', close=103),
        _bar('2024-01-02T05:00:00+00:00', close=101),
        _bar('2024-01-02T05:00:00+00:00', close=102),  # duplicate: last wins
    ]
    df = bars_to_frame(bars, symbol='AAPL')

    assert list(df['timestamp']) == [
        pd.Timestamp('2024-01-02T05:00Z'), pd.Timestamp('2024-01-03T05:00Z'),
    ]
    assert list(df['close']) == [102.0, 103.0]
    assert str(df['timestamp'].dtype) == 'datetime64[ns, UTC]'
    assert df['volume'].dtype == np.float64
    assert (df['symbol'] == 'AAPL').all()


@pytest.mark.parametrize('values', [
    ['2024-01-02T00:00:00-05:00', '2024-01-03T00:00:00-05:00'],  # offsets
    ['2024-01-02T05:00:00.000Z', '2024-01-03T05:00:00.000Z'],    # CCXT style
    ['2024-01-02 05:00:00', '2024-01-03 05:00:00'],              # naive = UTC
])
def test_timestamps_are_converted_to_utc(values):
    df = bars_to_frame([_bar(v) for v in values])
    assert list(df['timestamp']) == [
        pd.Timestamp('2024-01-02T05:00Z'), pd.Timestamp('2024-01-03T05:00Z'),
    ]


def test_mixed_naive_and_offset_strings_are_rejected():
    with pytest.raises(ValueError, match='mix'):
        bars_to_frame([_bar('2024-01-02T00:00:00-05:00'), _bar('2024-01-03 05:00:00')])


def test_fractional_volume_is_preserved():
    assert bars_to_frame([_bar('2024-01-02', volume=0.4)])['volume'].iloc[0] == 0.4


def test_empty_bars_give_empty_frame_with_schema_columns():
    df = bars_to_frame([])
    assert df.empty
    assert {'timestamp', 'open', 'high', 'low', 'close', 'volume'} <= set(df.columns)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match='volume'):
        bars_to_frame([{'timestamp': '2024-01-02', 'open': 1, 'high': 1, 'low': 1, 'close': 1}])


def test_bars_without_timestamp_are_dropped():
    df = bars_to_frame([_bar(None), _bar('2024-01-02')])
    assert len(df) == 1


@pytest.mark.parametrize('o,h,l,c,v,valid', [
    (100, 101, 99, 100, 10, True),
    (100, 99, 101, 100, 10, False),    # high < low
    (100, 101, 99, 102, 10, False),    # close above high
    (98, 101, 99, 100, 10, False),     # open below low
    (0, 0, 0, 0, 10, False),           # non-positive prices
    (100, 101, 99, 100, -1, False),    # negative volume
    (100, 101, 99, np.nan, 10, False), # missing close
    (np.inf, np.inf, np.inf, np.inf, 10, False),  # all +inf passes every comparison
    (100, np.inf, 99, 100, 10, False), # infinite high
    (100, 101, 99, 100, np.inf, False), # infinite volume
    (100, 101, -np.inf, 100, 10, False), # -inf low
])
def test_ohlc_valid_mask(o, h, l, c, v, valid):
    df = pd.DataFrame({'open': [o], 'high': [h], 'low': [l], 'close': [c], 'volume': [v]})
    assert bool(ohlc_valid_mask(df).iloc[0]) is valid


def test_normalize_interval_accepts_alpaca_names():
    assert normalize_interval('1Day') == '1d'
    assert normalize_interval(' 5Min ') == '5m'
    with pytest.raises(ValueError):
        normalize_interval('2d')


@pytest.mark.parametrize('interval,market,expected', [
    ('1d', 'equity', 252),
    ('1d', 'crypto', 365),
    ('1h', 'crypto', 365 * 24),
    ('5m', 'equity', 252 * 78),
    ('1h', 'equity', 252 * 7),   # 09:30 partial bar + 10:00..15:00
    ('4h', 'equity', 252 * 2),   # whole bars, never 1.625 per session
])
def test_periods_per_year(interval, market, expected):
    assert periods_per_year(interval, market) == pytest.approx(expected)


def _session_bars(day, times):
    """UTC timestamps for New York wall-clock `times` on `day`."""
    return [pd.Timestamp(f'{day} {t}', tz='America/New_York').tz_convert('UTC') for t in times]


def test_estimate_periods_per_year_follows_the_data():
    regular = ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00']
    extended = [f'{h:02d}:00' for h in range(4, 20)]  # 04:00..19:00, 16 bars
    days = pd.bdate_range('2024-03-04', periods=10).strftime('%Y-%m-%d')

    regular_ts = [t for d in days for t in _session_bars(d, regular)]
    regular_ts += _session_bars('2024-03-18', regular[:4])  # a half day
    assert estimate_periods_per_year(regular_ts, 'equity') == 252 * 7

    extended_ts = [t for d in days for t in _session_bars(d, extended)]
    assert estimate_periods_per_year(extended_ts, 'equity') == 252 * 16


def test_estimate_periods_per_year_daily_and_crypto():
    daily = pd.date_range('2024-01-02', periods=20, freq='B', tz='America/New_York')
    assert estimate_periods_per_year(daily, 'equity') == 252
    hourly = pd.date_range('2024-01-01', periods=24 * 5, freq='h', tz='UTC')
    assert estimate_periods_per_year(hourly, 'crypto') == 365 * 24
    with pytest.raises(ValueError):
        estimate_periods_per_year([], 'equity')


def test_unknown_market_raises():
    with pytest.raises(ValueError):
        periods_per_year('1d', 'forex')


def test_require_single_symbol():
    require_single_symbol(pd.DataFrame({'symbol': ['A', 'A']}), 'x')
    with pytest.raises(ValueError, match='one symbol'):
        require_single_symbol(pd.DataFrame({'symbol': ['A', 'B']}), 'x')
