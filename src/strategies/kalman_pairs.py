"""
Pairs trading with a Kalman-filtered hedge ratio (Intelligent Markets, ch. 4).

Instead of re-fitting a rolling regression, a Kalman filter treats the
intercept and hedge ratio of

    log A_t = intercept_t + hedge_ratio_t * log B_t + noise

as slowly drifting random walks and updates them one bar at a time. The
trading signal is the filter's prediction error (innovation) divided by its
predicted standard deviation: a z-score built only from earlier bars.

Changes from the book's KalmanFilterPairs:
    * The state includes an intercept, so the spread can have a non-zero mean.
    * The observation noise is estimated from a warm-up regression instead
      of being fixed at 1.0, which only suited one price scale, and log B
      is centred on its warm-up mean so `delta` means the same for a $5
      stock as for a $500 one.
    * The z-score uses the innovation and its variance (x² P + R) from the
      prediction step. The book divided a spread built with the already
      updated hedge ratio, which had seen the current bar.
    * Missing bars advance time without an update instead of being skipped,
      so the output stays aligned with the input index.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.market_data.panel import to_panel
from src.strategies.base import BaseStrategy
from src.strategies.pairs_trading import hold_positions, log_prices, pair_weights
from src.strategies.weights import weights_to_signals


class KalmanHedgeRatio:
    """Online estimate of the intercept and hedge ratio between two prices."""

    def __init__(self, delta: float = 1e-5, warmup: int = 30):
        """
        Args:
            delta: How fast the hedge ratio may drift. The state noise is
                delta / (1 - delta) per bar; larger values adapt faster but
                give noisier estimates.
            warmup: Bars used for the initial regression. No output for them.

        Raises:
            ValueError: If delta is not in (0, 1) or warmup < 3.
        """
        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1")
        if warmup < 3:
            raise ValueError("warmup must be at least 3")
        self.delta = delta
        self.warmup = warmup

    def filter(self, price_a: pd.Series, price_b: pd.Series) -> pd.DataFrame:
        """
        Run the filter over two price series.

        Args:
            price_a: Prices of leg A.
            price_b: Prices of leg B.

        Returns:
            DataFrame on the union of both indexes with columns hedge_ratio
            and intercept (estimates after seeing the bar), spread (the
            innovation), spread_std (its predicted standard deviation) and
            z_score. NaN during warm-up and on bars missing either price.

        Raises:
            ValueError: If fewer than `warmup` bars have both prices, or a
                price is not positive.
        """
        index = price_a.index.union(price_b.index)
        log_a, log_b = log_prices(price_a, price_b)
        if len(log_a) < self.warmup:
            raise ValueError(f"Need at least {self.warmup} bars with both prices")
        y = log_a.reindex(index).to_numpy()
        x = log_b.reindex(index).to_numpy()
        out = np.full((len(index), 5), np.nan)

        # Initial state from an OLS fit on the first `warmup` complete bars.
        # log B is centred on its warm-up mean so that hedge-ratio drift
        # (delta) does not scale with the price level.
        complete = np.flatnonzero(~np.isnan(y) & ~np.isnan(x))
        first = complete[: self.warmup]
        centre = x[first].mean()
        x = x - centre
        design = np.column_stack([x[first], np.ones(len(first))])
        state, *_ = np.linalg.lstsq(design, y[first], rcond=None)
        residuals = y[first] - design @ state
        obs_var = max(float(residuals @ residuals) / (len(first) - 2), 1e-12)
        cov = obs_var * np.linalg.pinv(design.T @ design)
        state_noise = self.delta / (1 - self.delta) * np.eye(2)

        for t in range(first[-1] + 1, len(index)):
            cov = cov + state_noise
            if np.isnan(y[t]) or np.isnan(x[t]):
                continue
            h = np.array([x[t], 1.0])
            innovation = y[t] - h @ state
            innovation_var = h @ cov @ h + obs_var
            gain = cov @ h / innovation_var
            state = state + gain * innovation
            cov = cov - np.outer(gain, h) @ cov
            std = np.sqrt(innovation_var)
            out[t] = (state[0], state[1], innovation, std, innovation / std)

        result = pd.DataFrame(
            out, index=index,
            columns=['hedge_ratio', 'intercept', 'spread', 'spread_std', 'z_score'],
        )
        # Report the intercept for uncentred log prices:
        # log A = intercept + hedge_ratio * log B + spread.
        result['intercept'] -= result['hedge_ratio'] * centre
        return result


class KalmanPairsStrategy(BaseStrategy):
    """Pairs trading on the Kalman filter's innovation z-score."""

    name = 'kalman_pairs'
    version = '1.0.0'

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        delta: float = 1e-5,
        warmup: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
    ):
        """
        Args:
            symbol_a: Leg A (regressed on B).
            symbol_b: Leg B (the hedge).
            delta: Hedge-ratio drift speed (see KalmanHedgeRatio).
            warmup: Bars for the initial regression.
            entry_z: |z| that opens a trade.
            exit_z: |z| at or below which a trade closes.
            stop_z: |z| at or above which a trade closes at a loss.

        Raises:
            ValueError: If a parameter is out of range.
        """
        if symbol_a == symbol_b:
            raise ValueError("symbol_a and symbol_b must differ")
        if not 0 <= exit_z < entry_z < stop_z:
            raise ValueError("Thresholds must satisfy 0 <= exit_z < entry_z < stop_z")
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.kalman = KalmanHedgeRatio(delta=delta, warmup=warmup)

    def get_parameters(self) -> Dict[str, Any]:
        """Parameters for experiment tracking."""
        return {
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'delta': self.kalman.delta,
            'warmup': self.kalman.warmup,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'stop_z': self.stop_z,
        }

    def spread(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Filter output plus the held position for the pair.

        Args:
            prices: Close-price panel containing both symbols.
        """
        frame = self.kalman.filter(prices[self.symbol_a], prices[self.symbol_b])
        frame['position'] = hold_positions(
            frame['z_score'], self.entry_z, self.exit_z, self.stop_z
        )
        return frame

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Weights for the two legs, sized with the current hedge ratio.

        Args:
            prices: Close-price panel containing both symbols.
        """
        frame = self.spread(prices).reindex(prices.index)
        return pair_weights(
            frame['position'], frame['hedge_ratio'], self.symbol_a, self.symbol_b
        )

    def generate_signals(
        self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Entry/exit events for both legs.

        Args:
            data: Long OHLCV frame containing both symbols.
            features: Unused.
        """
        return weights_to_signals(self.target_weights(to_panel(data)))
