"""
Feature engineering: composes the feature groups for one symbol's bars.

    returns      return_{h}, log_return_1
    volatility   volatility_{w}, vol_ratio_*, garman_klass_var, gk_volatility_{w}
    momentum     ma_crossover_*, price_vs_ma_*, rsi_*, macd*
    volume       volume_ratio_*, obv_ratio_*, vwap_deviation_*, mfi_*
    patterns     range_pct, close_position, body_size, wicks, bb_position_*

Windows are measured in bars. Targets (forward returns) are deliberately not
produced here; see src.features.labels.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.market_data.schema import (
    NUMERIC_COLUMNS, check_market, normalize_interval, require_single_symbol,
)
from src.market_data.schema import periods_per_year as conventional_periods_per_year
from src.features.momentum import momentum_features
from src.features.patterns import pattern_features
from src.features.returns import return_features
from src.features.volatility import volatility_features
from src.features.volume import volume_features


class FeatureEngineer:
    """
    Feature engineer for one symbol's OHLCV time series.

    Args:
        interval: Bar interval of the input ('1d', '5m', ...).
        market: 'equity' or 'crypto'; with the interval it sets the
            annualization factor (252 sessions vs 365 days, bars per day).
        periods_per_year: Bars per year to annualize volatility with,
            overriding the convention from `schema.periods_per_year`. Pass
            `schema.estimate_periods_per_year(df['timestamp'], market)` for
            intraday equity data, whose real bar count depends on extended
            hours and how bars are aligned.

    Raises:
        ValueError: If the interval or market is unsupported, or
            `periods_per_year` is not positive.
    """

    def __init__(
        self,
        interval: str = '1d',
        market: str = 'equity',
        periods_per_year: Optional[float] = None,
    ):
        """Validate the interval and market and work out bars per year."""
        self.interval = normalize_interval(interval)
        self.market = check_market(market)
        if periods_per_year is None:
            periods_per_year = conventional_periods_per_year(self.interval, self.market)
        elif not periods_per_year > 0:
            raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
        self.periods_per_year = float(periods_per_year)
        self.feature_names: List[str] = []

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the full feature set.

        Args:
            df: One symbol's bars with open, high, low, close and volume,
                sorted by timestamp ascending.

        Returns:
            Copy of `df` with every feature column appended (existing columns
            with the same names are replaced). The new column names are
            stored in `self.feature_names`.
        """
        missing = [c for c in NUMERIC_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"FeatureEngineer needs OHLCV columns; missing {missing}")
        require_single_symbol(df, 'FeatureEngineer')
        if 'timestamp' in df.columns and not df['timestamp'].is_monotonic_increasing:
            raise ValueError("Sort bars by timestamp before computing features")

        features = pd.concat(
            [
                return_features(df),
                volatility_features(df, self.periods_per_year),
                momentum_features(df),
                volume_features(df),
                pattern_features(df),
            ],
            axis=1,
        )
        self.feature_names = list(features.columns)
        base = df.drop(columns=features.columns.intersection(df.columns))
        return pd.concat([base, features], axis=1)


# Usage example
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range('2023-01-01', periods=n, freq='B', tz='UTC')
    prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))

    sample = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + rng.normal(0, 0.002, n)),
        'high': prices * (1 + abs(rng.normal(0.005, 0.003, n))),
        'low': prices * (1 - abs(rng.normal(0.005, 0.003, n))),
        'close': prices,
        'volume': rng.lognormal(15, 1.5, n),
    })
    sample['high'] = sample[['open', 'high', 'close']].max(axis=1)
    sample['low'] = sample[['open', 'low', 'close']].min(axis=1)

    engineer = FeatureEngineer(interval='1d', market='equity')
    features_df = engineer.compute_all_features(sample)

    print(f"Generated {len(engineer.feature_names)} features from 6 raw columns\n")
    for col in engineer.feature_names:
        print(f" {col:24s}: {features_df[col].notna().sum()}/{len(features_df)} non-null")
