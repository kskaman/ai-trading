"""Past-return features. Windows are in bars, so names work for any interval."""

from typing import Sequence

import numpy as np
import pandas as pd

RETURN_HORIZONS = (1, 3, 5, 10, 20, 60)


def return_features(df: pd.DataFrame, horizons: Sequence[int] = RETURN_HORIZONS) -> pd.DataFrame:
    """
    Compute past returns over several look-back windows.

    Args:
        df: Bars for one symbol with a 'close' column, sorted by time.
        horizons: Look-back lengths in bars, one output column each.

    Returns:
        DataFrame on `df.index` with these columns:

        - return_{h}: close / close h bars ago - 1 (NaN for the first h rows).
        - log_return_1: one-bar log return.
    """
    close = df['close']
    out = {f'return_{h}': close / close.shift(h) - 1 for h in horizons}
    out['log_return_1'] = np.log(close / close.shift(1))
    return pd.DataFrame(out, index=df.index)
