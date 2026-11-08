"""
Multi-factor long/short model (Intelligent Markets, chapter 4).

Each factor is a panel of scores, standardized across symbols on every bar
(cross-sectional z-score). A weighted sum of the factors ranks the
universe; the top names are bought and the bottom names sold short.

Built-in factors use prices only:

    momentum         return from lookback + skip bars ago to skip bars ago
    low_volatility   minus the rolling volatility of daily log returns

Changes from the book's MultiFactorModel:
    * Factors are computed for every bar, not just the latest one, so the
      model can be backtested.
    * Momentum skips the most recent month, as the momentum strategy does.
    * There is no built-in value factor: it needs earnings data, which the
      data layer does not collect. Pass any extra factor panel (value,
      quality, a model's predictions) through `extra_factors`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from src.market_data.panel import to_panel
from src.strategies.base import BaseStrategy
from src.strategies.weights import (
    cross_sectional_zscore,
    long_short_weights,
    weights_to_signals,
)

DEFAULT_FACTOR_WEIGHTS = {'momentum': 0.5, 'low_volatility': 0.5}


def combine_factors(
    factors: Mapping[str, pd.DataFrame], weights: Mapping[str, float]
) -> pd.DataFrame:
    """
    Weighted sum of factor panels.

    A symbol gets a composite score on a bar only if every weighted factor
    has a value for it there.

    Args:
        factors: Factor name -> panel (same index and columns).
        weights: Factor name -> weight. Factors not listed are ignored.

    Returns:
        Composite score panel.

    Raises:
        ValueError: If weights is empty or names a factor that is missing.
    """
    if not weights:
        raise ValueError("At least one factor weight is required")
    missing = sorted(set(weights) - set(factors))
    if missing:
        raise ValueError(f"No factor panel for {missing}")
    composite = None
    for name, weight in weights.items():
        term = factors[name] * weight
        composite = term if composite is None else composite + term
    return composite


class MultiFactorModel(BaseStrategy):
    """Rank symbols on a weighted mix of z-scored factors."""

    name = 'multi_factor'
    version = '1.0.0'

    def __init__(
        self,
        factor_weights: Optional[Mapping[str, float]] = None,
        momentum_lookback: int = 120,
        momentum_skip: int = 21,
        volatility_window: int = 60,
        n_long: int = 20,
        n_short: int = 20,
        rebalance_every: int = 21,
    ):
        """
        Args:
            factor_weights: Factor name -> weight. Defaults to equal weights
                on momentum and low_volatility.
            momentum_lookback: Bars of return in the momentum factor.
            momentum_skip: Most recent bars left out of momentum.
            volatility_window: Bars in the rolling volatility.
            n_long: Symbols bought at each rebalance.
            n_short: Symbols sold short at each rebalance.
            rebalance_every: Bars between rebalances.

        Raises:
            ValueError: If a window or count is out of range.
        """
        self.factor_weights = dict(factor_weights or DEFAULT_FACTOR_WEIGHTS)
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.volatility_window = volatility_window
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_every = rebalance_every
        self.validate()

    def get_parameters(self) -> Dict[str, Any]:
        """Parameters for experiment tracking."""
        return {
            'factor_weights': dict(self.factor_weights),
            'momentum_lookback': self.momentum_lookback,
            'momentum_skip': self.momentum_skip,
            'volatility_window': self.volatility_window,
            'n_long': self.n_long,
            'n_short': self.n_short,
            'rebalance_every': self.rebalance_every,
        }

    def validate(self) -> bool:
        """
        Check parameter ranges.

        Raises:
            ValueError: If there are no factor weights, a window is too
                short, or the counts are invalid.
        """
        if not self.factor_weights:
            raise ValueError("factor_weights must not be empty")
        if self.momentum_lookback < 1 or self.momentum_skip < 0:
            raise ValueError("Need momentum_lookback >= 1 and momentum_skip >= 0")
        if self.volatility_window < 2 or self.rebalance_every < 1:
            raise ValueError("Need volatility_window >= 2 and rebalance_every >= 1")
        if self.n_long < 0 or self.n_short < 0 or self.n_long + self.n_short == 0:
            raise ValueError("n_long and n_short must be >= 0 and not both 0")
        return True

    def momentum_factor(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score of skip-adjusted past return."""
        past = prices.shift(self.momentum_skip)
        raw = past / prices.shift(self.momentum_lookback + self.momentum_skip) - 1
        return cross_sectional_zscore(raw)

    def low_volatility_factor(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score of negative rolling volatility (calm is high)."""
        log_returns = np.log(prices).diff()
        volatility = log_returns.rolling(self.volatility_window).std()
        return -cross_sectional_zscore(volatility)

    def factors(
        self,
        prices: pd.DataFrame,
        extra_factors: Optional[Mapping[str, pd.DataFrame]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Every factor panel by name.

        Args:
            prices: Close-price panel.
            extra_factors: Additional panels, z-scored here so they are on
                the same scale as the built-in factors.
        """
        panels = {
            'momentum': self.momentum_factor(prices),
            'low_volatility': self.low_volatility_factor(prices),
        }
        for name, panel in (extra_factors or {}).items():
            aligned = panel.reindex(index=prices.index, columns=prices.columns)
            panels[name] = cross_sectional_zscore(aligned)
        return panels

    def composite_score(
        self,
        prices: pd.DataFrame,
        extra_factors: Optional[Mapping[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Weighted factor mix used for ranking (see `combine_factors`)."""
        return combine_factors(self.factors(prices, extra_factors), self.factor_weights)

    def target_weights(
        self,
        prices: pd.DataFrame,
        extra_factors: Optional[Mapping[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Portfolio weights for every bar (see src.strategies.weights).

        Args:
            prices: Close-price panel with at least n_long + n_short symbols.
            extra_factors: Additional factor panels named in factor_weights.
        """
        return long_short_weights(
            self.composite_score(prices, extra_factors),
            self.n_long, self.n_short, self.rebalance_every,
        )

    def generate_signals(
        self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Entry/exit events from the price-only factors.

        Args:
            data: Long OHLCV frame with several symbols.
            features: Unused.
        """
        return weights_to_signals(self.target_weights(to_panel(data)))
