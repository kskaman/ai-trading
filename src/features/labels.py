"""
Prediction targets.

Kept apart from src.features on purpose: these columns look into the future,
so they must never be used as model inputs. Build features and labels as
separate frames and join them only when assembling a training set.
"""

from typing import Sequence

import pandas as pd

FORWARD_HORIZONS = (1, 5, 10)


def forward_returns(df: pd.DataFrame, horizons: Sequence[int] = FORWARD_HORIZONS) -> pd.DataFrame:
    """
    Compute future returns to use as prediction targets.

    These look ahead in time, so never feed them to a model as inputs.

    Args:
        df: Bars for one symbol with a 'close' column, sorted by time.
        horizons: How many bars ahead to look, one output column each.

    Returns:
        DataFrame on `df.index` with one column per horizon:
        forward_return_{h} = close h bars ahead / close now - 1.
        The last h rows are NaN because their future is not known yet.
    """
    close = df['close']
    return pd.DataFrame(
        {f'forward_return_{h}': close.shift(-h) / close - 1 for h in horizons},
        index=df.index,
    )
