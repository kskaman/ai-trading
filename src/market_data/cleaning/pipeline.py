"""
OHLCV cleaning pipeline: normalize, then attach quality flags.

clean_ohlcv never drops or edits bars (beyond normalizing the schema). It
adds flag columns so downstream code decides what to exclude:

    log_return   bar-to-bar log return
    is_outlier   return far outside the recent distribution
    is_stale     volume far below the recent median
    ohlc_valid   prices internally consistent (see schema.ohlc_valid_mask)
"""

import logging
from dataclasses import dataclass

import pandas as pd

from src.market_data.cleaning.flags import flag_return_outliers, flag_stale_volume, log_returns
from src.market_data.schema import normalize_ohlcv, ohlc_valid_mask, require_single_symbol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningReport:
    """Counts of flagged bars in a cleaned frame."""

    n_bars: int
    n_outliers: int
    n_stale: int
    n_invalid_ohlc: int

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> 'CleaningReport':
        return cls(
            n_bars=len(df),
            n_outliers=int(df['is_outlier'].sum()),
            n_stale=int(df['is_stale'].sum()),
            n_invalid_ohlc=int((~df['ohlc_valid']).sum()),
        )

    def __str__(self) -> str:
        return (
            f"{self.n_bars} bars: {self.n_outliers} return outliers, "
            f"{self.n_stale} stale (low volume), {self.n_invalid_ohlc} invalid OHLC"
        )


def clean_ohlcv(
    df: pd.DataFrame,
    max_return_std: float = 5.0,
    min_volume_threshold: float = 0.1,
    window: int = 50,
) -> pd.DataFrame:
    """
    Normalize one symbol's OHLCV bars and add quality flags.

    Args:
        df: OHLCV frame for a single symbol (any order; duplicates allowed).
        max_return_std: Flag returns more than this many standard deviations
            from the mean of the previous `window` returns.
        min_volume_threshold: Flag bars whose volume is below this fraction
            of the previous `window` bars' median volume.
        window: Look-back length, in bars, for both rules.

    Returns:
        Sorted, de-duplicated copy with log_return, is_outlier, is_stale and
        ohlc_valid columns. Use CleaningReport.from_frame() for the counts.
    """
    require_single_symbol(df, 'clean_ohlcv')
    df = normalize_ohlcv(df)

    df['log_return'] = log_returns(df['close'])
    df['is_outlier'] = flag_return_outliers(df['close'], window=window, max_z=max_return_std)
    df['is_stale'] = flag_stale_volume(
        df['volume'], window=window, min_fraction=min_volume_threshold
    )
    df['ohlc_valid'] = ohlc_valid_mask(df)

    logger.info("Cleaning summary: %s", CleaningReport.from_frame(df))
    return df
