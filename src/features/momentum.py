"""Momentum and trend features."""

import numpy as np
import pandas as pd

MA_CROSSOVERS = ((5, 20), (10, 50), (20, 60))
PRICE_VS_MA_PERIODS = (20, 50, 200)
RSI_PERIODS = (14, 21)


def rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI (exponential smoothing with alpha = 1/period).

    100 when the window had gains but no losses, 50 when prices were flat,
    NaN until `period` price changes are available.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    value = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    value = value.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return value.mask((avg_loss == 0) & (avg_gain == 0), 50.0)


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ma_crossover_{fast}_{slow}: fast SMA / slow SMA - 1.
    price_vs_ma_{p}: close / SMA(p) - 1.
    rsi_{p}: Wilder's RSI.
    macd, macd_signal, macd_histogram: 12/26/9 MACD in price units.
    """
    close = df['close']
    out = {}

    for fast, slow in MA_CROSSOVERS:
        out[f'ma_crossover_{fast}_{slow}'] = (
            close.rolling(fast).mean() / close.rolling(slow).mean() - 1
        )

    for period in PRICE_VS_MA_PERIODS:
        out[f'price_vs_ma_{period}'] = close / close.rolling(period).mean() - 1

    for period in RSI_PERIODS:
        out[f'rsi_{period}'] = rsi(close, period)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out['macd'] = ema_12 - ema_26
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['macd_histogram'] = out['macd'] - out['macd_signal']

    return pd.DataFrame(out, index=df.index)
