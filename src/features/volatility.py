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
    volatility_{w}: annualized std of log returns over w bars.
    vol_ratio_{fast}_{slow}: shortest-window volatility over longest-window volatility.
    garman_klass_var: per-bar Garman-Klass variance (not annualized).
    gk_volatility_{w}: annualized Garman-Klass volatility over w bars.
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
