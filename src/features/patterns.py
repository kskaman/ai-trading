"""Candle shape and band-position features."""

import numpy as np
import pandas as pd

BOLLINGER_PERIOD = 20


def pattern_features(df: pd.DataFrame, bb_period: int = BOLLINGER_PERIOD) -> pd.DataFrame:
    """
    range_pct: (high - low) / close.
    close_position: where the close sits in the bar's range (0 = low, 1 = high).
    body_size: |close - open| / open.
    upper_wick, lower_wick: wick lengths as a fraction of close.
    bb_position_{p}: position within 2-sigma Bollinger Bands (0 = lower, 1 = upper).
    """
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    bar_range = (h - l).replace(0, np.nan)

    ma = c.rolling(bb_period).mean()
    band_width = (4 * c.rolling(bb_period).std()).replace(0, np.nan)

    return pd.DataFrame({
        'range_pct': (h - l) / c,
        'close_position': (c - l) / bar_range,
        'body_size': (c - o).abs() / o,
        'upper_wick': (h - np.maximum(o, c)) / c,
        'lower_wick': (np.minimum(o, c) - l) / c,
        f'bb_position_{bb_period}': (c - (ma - band_width / 2)) / band_width,
    }, index=df.index)
