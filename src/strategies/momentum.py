"""
Cross-sectional momentum (Intelligent Markets, chapter 4).

Rank symbols by their return over the past `lookback` bars, skipping the
most recent `skip` bars (the last month tends to mean-revert), then buy the
winners and short the losers. Rebalance every `rebalance_every` bars and hold
in between.

Changes from the book's MomentumStrategy:
    * Holdings carry between rebalances. The book filled the weight frame
      with 0.0 before forward-filling, so positions existed only on
      rebalance days.
    * Long and short sides are each half the capital (dollar neutral, gross
      exposure 1) rather than +1 and -1.
    * Windows are counted in bars, so the same class works on crypto.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.market_data.panel import to_panel
from src.strategies.base import BaseStrategy
from src.strategies.weights import long_short_weights, weights_to_signals


class CrossSectionalMomentum(BaseStrategy):
    """Long the strongest past performers, short the weakest."""

    name = 'cross_sectional_momentum'
    version = '1.0.0'

    def __init__(
        self,
        lookback: int = 120,
        skip: int = 21,
        rebalance_every: int = 21,
        n_long: int = 10,
        n_short: int = 10,
    ):
        """
        Args:
            lookback: Bars of return used to rank symbols.
            skip: Most recent bars left out of the ranking window.
            rebalance_every: Bars between rebalances.
            n_long: Symbols bought at each rebalance (0 for short-only).
            n_short: Symbols sold short at each rebalance (0 for long-only).

        Raises:
            ValueError: If a parameter is out of range (see `validate`).
        """
        self.lookback = lookback
        self.skip = skip
        self.rebalance_every = rebalance_every
        self.n_long = n_long
        self.n_short = n_short
        self.validate()

    def get_parameters(self) -> Dict[str, Any]:
        """Parameters for experiment tracking."""
        return {
            'lookback': self.lookback,
            'skip': self.skip,
            'rebalance_every': self.rebalance_every,
            'n_long': self.n_long,
            'n_short': self.n_short,
        }

    def validate(self) -> bool:
        """
        Check parameter ranges.

        Raises:
            ValueError: If lookback < 1, skip < 0, rebalance_every < 1, a
                count is negative, or both counts are 0.
        """
        if self.lookback < 1 or self.skip < 0 or self.rebalance_every < 1:
            raise ValueError("Need lookback >= 1, skip >= 0, rebalance_every >= 1")
        if self.n_long < 0 or self.n_short < 0 or self.n_long + self.n_short == 0:
            raise ValueError("n_long and n_short must be >= 0 and not both 0")
        return True

    def scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum score: price `skip` bars ago / price `lookback + skip` bars
        ago - 1. NaN until enough history exists.

        Args:
            prices: Close-price panel (see src.market_data.panel).
        """
        return prices.shift(self.skip) / prices.shift(self.lookback + self.skip) - 1

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Portfolio weights for every bar (see src.strategies.weights).

        Args:
            prices: Close-price panel with at least n_long + n_short symbols.

        Returns:
            Weights panel, all zero until the first rebalance.
        """
        return long_short_weights(
            self.scores(prices), self.n_long, self.n_short, self.rebalance_every
        )

    def generate_signals(
        self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Entry/exit events for a long multi-symbol OHLCV frame.

        Args:
            data: Bars with 'timestamp', 'symbol' and 'close' columns.
            features: Unused; momentum only needs prices.
        """
        return weights_to_signals(self.target_weights(to_panel(data)))
