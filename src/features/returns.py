"""Past-return features. Windows are in bars, so names work for any interval."""

from typing import Sequence

import numpy as np
import pandas as pd

RETURN_HORIZONS = (1, 3, 5, 10, 20, 60)


def return_features(df: pd.DataFrame, horizons: Sequence[int] = RETURN_HORIZONS) -> pd.DataFrame:
    """
    return_{h}: simple return over the last h bars (close / close h bars ago - 1).
    log_return_1: one-bar log return.
    """
    close = df['close']
    out = {f'return_{h}': close / close.shift(h) - 1 for h in horizons}
    out['log_return_1'] = np.log(close / close.shift(1))
    return pd.DataFrame(out, index=df.index)
