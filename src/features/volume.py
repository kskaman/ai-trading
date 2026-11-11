"""
Volume features.

All features use rolling windows. Cumulative versions (classic OBV, VWAP
anchored at the first bar) depend on where the data happens to start, so the
same day gets different values in a backtest and in live trading.
"""

from typing import Sequence

import numpy as np
import pandas as pd

VOLUME_RATIO_WINDOWS = (10, 20, 60)
FLOW_WINDOW = 20
MFI_PERIOD = 14


def volume_features(
    df: pd.DataFrame,
    ratio_windows: Sequence[int] = VOLUME_RATIO_WINDOWS,
    flow_window: int = FLOW_WINDOW,
    mfi_period: int = MFI_PERIOD,
) -> pd.DataFrame:
    """
    Compute volume and money-flow features over rolling windows.

    Args:
        df: Bars for one symbol with 'high', 'low', 'close' and 'volume'
            columns, sorted by time.
        ratio_windows: Look-back lengths in bars for `volume_ratio_{w}`.
        flow_window: Look-back length in bars for OBV ratio and VWAP.
        mfi_period: Look-back length in bars for the money flow index.

    Returns:
        DataFrame on `df.index` with these columns:

        - volume_ratio_{w}: volume / mean volume of the last w bars.
        - obv_ratio_{w}: rolling on-balance volume / total volume, in
            [-1, 1] (+1 = all volume on up bars, -1 = all on down bars).
        - vwap_deviation_{w}: close / rolling w-bar VWAP - 1.
        - mfi_{p}: money flow index, 0 to 100.
    """
    close, volume = df['close'], df['volume']
    out = {}

    for w in ratio_windows:
        out[f'volume_ratio_{w}'] = volume / volume.rolling(w).mean().replace(0, np.nan)

    total_volume = volume.rolling(flow_window).sum().replace(0, np.nan)
    signed_volume = np.sign(close.diff()) * volume
    out[f'obv_ratio_{flow_window}'] = signed_volume.rolling(flow_window).sum() / total_volume

    typical_price = (df['high'] + df['low'] + close) / 3
    vwap = (typical_price * volume).rolling(flow_window).sum() / total_volume
    out[f'vwap_deviation_{flow_window}'] = close / vwap - 1

    money_flow = typical_price * volume
    tp_change = typical_price.diff()
    positive = money_flow.where(tp_change > 0, 0.0).rolling(mfi_period).sum()
    negative = money_flow.where(tp_change < 0, 0.0).rolling(mfi_period).sum()
    out[f'mfi_{mfi_period}'] = 100 - 100 / (1 + positive / negative.replace(0, np.nan))

    return pd.DataFrame(out, index=df.index)
