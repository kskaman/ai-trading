"""Tests for cleaning flags and the clean_ohlcv pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.market_data.cleaning import (
    CleaningReport, clean_ohlcv, flag_return_outliers, flag_stale_volume,
)
from tests.factories import make_ohlcv


def _scale_prices(df, rows, factor):
    df.loc[rows, ['open', 'high', 'low', 'close']] *= factor


def test_price_jump_is_flagged():
    df = make_ohlcv(300)
    _scale_prices(df, df.index >= 200, 3.0)  # e.g. an unadjusted split

    flags = flag_return_outliers(df['close'])
    assert flags.iloc[200]
    assert flags.sum() == 1


def test_window_excludes_the_scored_bar():
    # With the bar inside its own window the z-score is capped at
    # (n-1)/sqrt(n) = 6.93 for n=50, so a 7-sigma threshold could never fire.
    df = make_ohlcv(300)
    _scale_prices(df, df.index >= 200, 3.0)
    assert flag_return_outliers(df['close'], max_z=7.0).iloc[200]


def test_single_bad_tick_flags_spike_and_reversal():
    df = make_ohlcv(300)
    _scale_prices(df, [200], 3.0)  # one bad print, then back to normal

    flags = flag_return_outliers(df['close'])
    assert flags.iloc[200] and flags.iloc[201]


def test_normal_random_walk_has_no_outliers():
    assert not flag_return_outliers(make_ohlcv(1000, seed=3)['close']).any()


def test_low_and_zero_volume_bars_are_stale():
    volume = pd.Series(np.full(100, 1_000.0))
    volume[60] = 50.0   # 5% of median
    volume[70] = 0.0
    volume[80] = 500.0  # 50% of median: fine

    stale = flag_stale_volume(volume)
    assert list(stale[stale].index) == [60, 70]


def test_clean_ohlcv_adds_flags_and_reports():
    df = make_ohlcv(300)
    _scale_prices(df, df.index >= 200, 3.0)
    df.loc[100, 'volume'] = 0.0
    df.loc[150, 'high'] = df.loc[150, 'low'] - 1

    out = clean_ohlcv(df)

    assert {'log_return', 'is_outlier', 'is_stale', 'ohlc_valid'} <= set(out.columns)
    assert CleaningReport.from_frame(out) == CleaningReport(
        n_bars=300, n_outliers=1, n_stale=1, n_invalid_ohlc=1,
    )


def test_clean_ohlcv_sorts_dedupes_and_does_not_mutate_input():
    df = make_ohlcv(20)
    shuffled = pd.concat([df.iloc[::-1], df.iloc[[5]]], ignore_index=True)
    before = shuffled.copy()

    out = clean_ohlcv(shuffled)

    assert out['timestamp'].is_monotonic_increasing
    assert len(out) == 20
    pd.testing.assert_frame_equal(shuffled, before)


def test_clean_ohlcv_refuses_mixed_symbols():
    a = make_ohlcv(5).assign(symbol='AAPL')
    b = make_ohlcv(5).assign(symbol='MSFT')
    with pytest.raises(ValueError, match='one symbol'):
        clean_ohlcv(pd.concat([a, b]))
