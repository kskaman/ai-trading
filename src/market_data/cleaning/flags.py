"""
Bar-level quality rules.

Each function is pure: it reads OHLCV columns and returns a boolean Series
aligned to the input without modifying anything, so rules can be tested and
combined independently. OHLC consistency lives in src.market_data.schema
(ohlc_valid_mask) because the store and providers share it.
"""

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """Bar-to-bar log returns; non-positive prices give NaN instead of -inf."""
    positive = close.where(close > 0)
    return np.log(positive / positive.shift(1))


def flag_return_outliers(
    close: pd.Series,
    window: int = 50,
    min_periods: int = 10,
    max_z: float = 5.0,
) -> pd.Series:
    """
    True where a bar's log return is more than `max_z` standard deviations
    away from the mean of the previous `window` returns.

    The window deliberately excludes the bar being scored. If it included
    it, the bar would inflate its own standard deviation and the z-score
    could never exceed (n-1)/sqrt(n), which is 6.93 for n=50, so any
    threshold above that would never fire.
    """
    returns = log_returns(close)
    prior = returns.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std()
    z_scores = (returns - mean).abs() / std.replace(0, np.nan)
    return (z_scores > max_z).rename('is_outlier')


def flag_stale_volume(
    volume: pd.Series,
    window: int = 50,
    min_periods: int = 10,
    min_fraction: float = 0.1,
) -> pd.Series:
    """
    True where volume is below `min_fraction` of the median volume of the
    previous `window` bars (a likely stale or partial bar). Zero-volume bars
    are always flagged once enough history exists.
    """
    median = volume.shift(1).rolling(window, min_periods=min_periods).median()
    return (volume < median * min_fraction).rename('is_stale')
