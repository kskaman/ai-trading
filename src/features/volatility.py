"""Volatility features, annualized so values are comparable across intervals."""

from typing import Sequence

import numpy as np
import pandas as pd

VOLATILITY_WINDOWS = (5, 10, 20, 60)
GARMAN_KLASS_WINDOWS = (5, 20)


def garman_klass_variance(df: pd.DataFrame) -> pd.Series:
    """
    Per-bar Garman-Klass variance:

        0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2

    Non-negative for any valid bar (|ln(C/O)| <= ln(H/L)).

    Args:
        df: Bars with 'open', 'high', 'low' and 'close' columns.

    Returns:
        Variance of one bar's log return on `df.index`, not annualized.
    """
    hl = np.log(df['high'] / df['low']) ** 2
    co = np.log(df['close'] / df['open']) ** 2
    return 0.5 * hl - (2 * np.log(2) - 1) * co


def volatility_features(
    df: pd.DataFrame,
    periods_per_year: float,
    windows: Sequence[int] = VOLATILITY_WINDOWS,
    gk_windows: Sequence[int] = GARMAN_KLASS_WINDOWS,
) -> pd.DataFrame:
    """
    Compute annualized volatility features.

    Args:
        df: Bars for one symbol with OHLC columns, sorted by time.
        periods_per_year: Bars in a year for this interval and market,
            from `schema.periods_per_year` (e.g. 252 for daily stocks).
        windows: Look-back lengths in bars for close-to-close volatility.
        gk_windows: Look-back lengths in bars for Garman-Klass volatility.

    Returns:
        DataFrame on `df.index` with these columns:

        - volatility_{w}: annualized std of log returns over w bars.
        - vol_ratio_{fast}_{slow}: shortest-window volatility divided by
            longest-window volatility (above 1 = volatility rising).
        - garman_klass_var: per-bar Garman-Klass variance (not annualized).
        - gk_volatility_{w}: annualized Garman-Klass volatility over w bars.
    """
    close = df['close']
    log_ret = np.log(close / close.shift(1))
    annualize = np.sqrt(periods_per_year)

    out = {
        f'volatility_{w}': log_ret.rolling(w).std() * annualize for w in windows
    }
    fast, slow = min(windows), max(windows)
    out[f'vol_ratio_{fast}_{slow}'] = (
        out[f'volatility_{fast}'] / out[f'volatility_{slow}'].replace(0, np.nan)
    )

    gk_var = garman_klass_variance(df)
    out['garman_klass_var'] = gk_var
    for w in gk_windows:
        out[f'gk_volatility_{w}'] = np.sqrt(gk_var.rolling(w).mean() * periods_per_year)

    return pd.DataFrame(out, index=df.index)
