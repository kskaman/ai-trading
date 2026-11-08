"""Interface every trading strategy implements."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Every strategy must implement:
        - generate_signals(): Core method that produces trading signals
        - get_parameters(): Returns strategy hyperparameters for tracking
        - validate(): Checks that parameters are within valid ranges

    The built-in strategies also expose `target_weights()`, a panel of
    portfolio weights (see src.strategies.weights), and derive their signals
    from it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier (e.g., 'momentum_ml', 'kalman_pairs')."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Strategy version for tracking and audit purposes."""
        pass

    @abstractmethod
    def generate_signals(
        self, 
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals from market data and features.

        Args:
            data: Market data in long format (OHLCV rows with a 'symbol'
                column; several symbols for cross-sectional strategies).
            features: Engineered features from the feature pipeline; None
                for strategies that only use prices.

        Returns:
            DataFrame of signals with columns:
            ['timestamp', 'symbol', 'signal_type', 'strength']
            where signal_type is one of:
            'LONG_ENTER', 'LONG_EXIT', 'SHORT_ENTER', 'SHORT_EXIT'
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy hyperparameters for tracking and reproducibility."""
        pass

    def validate(self) -> bool:
        """Validate that strategy parameters are within acceptable ranges."""
        return True