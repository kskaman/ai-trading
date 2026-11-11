"""
Pairs trading on a cointegrated spread (Intelligent Markets, chapter 4).

Two related prices A and B are modelled in logs:

    log A = intercept + hedge_ratio * log B + spread

When the spread is unusually low (z <= -entry) buy A and short B; when it is
unusually high do the opposite. Close when it is back near zero
(|z| <= exit) or when it keeps diverging (|z| >= stop).

Changes from the book's PairsTradingStrategy:
    * Cointegration uses statsmodels' Engle-Granger test. ADF p-values on
      regression residuals are too optimistic, because the regression has
      already made the residuals look as stationary as possible.
    * The hedge ratio comes from a rolling window that ends on the previous
      bar, not from the whole sample, so there is no look-ahead.
    * Positions are held between entry and exit. The book's signals were
      stateless, so a trade lasted only while |z| stayed above entry.
    * The stop-loss works (the book read `stop_loss_threshold` but stored
      `stop_loss_thresh`, raising AttributeError).
    * Log prices make the hedge ratio a dollar ratio and keep parameters
      independent of the price level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.market_data.panel import to_panel
from src.strategies.base import BaseStrategy
from src.strategies.weights import weights_to_signals

# Bars needed to run the Engle-Granger cointegration test once.
MIN_COINTEGRATION_OBS = 60

# Default smallest window for the rolling hedge-ratio regression. An
# engineering safeguard, not a statistical guarantee: 60 daily bars is about
# three months, but 60 one-minute bars is one hour. Pass `min_window` to go
# lower on purpose, and check any window out of sample.
MIN_ROLLING_WINDOW = 60

# A regression with one slope and one intercept needs at least 3 points to
# leave any residual variance (it divides by window - 2). Never negotiable.
_MATH_MIN_WINDOW = 3


def _check_window(window: int, min_window: int, name: str) -> None:
    """
    Reject a regression window that is too short to trust.

    Args:
        window: The window being checked, in bars.
        min_window: Smallest window allowed; usually `MIN_ROLLING_WINDOW`.
        name: Parameter name for the error message, e.g. 'lookback'.

    Raises:
        ValueError: If `min_window` is below 3, or `window` is below
            `min_window`.
    """
    if min_window < _MATH_MIN_WINDOW:
        raise ValueError(f"min_window must be at least {_MATH_MIN_WINDOW}, got {min_window}")
    if window < min_window:
        raise ValueError(
            f"{name} must be at least {min_window} bars for pairs trading, got {window}. "
            "Shorter windows give unstable hedge ratios; lower `min_window` only "
            "deliberately (e.g. for intraday research)."
        )


@dataclass(frozen=True)
class CointegrationResult:
    """Outcome of `check_cointegration`."""

    is_cointegrated: bool
    """p_value is below the chosen significance level."""
    p_value: float
    """Engle-Granger p-value (MacKinnon); small means cointegrated."""
    t_statistic: float
    """Test statistic; more negative means more mean-reverting."""
    critical_value_5pct: float
    """t_statistic must be below this to reject at 5%."""
    hedge_ratio: float
    """Slope of log A on log B: dollars of B to short per dollar of A."""
    intercept: float
    """Intercept of the regression."""
    r_squared: float
    """Share of log A's variance explained by log B."""
    half_life: float
    """Bars for a spread deviation to halve; inf if it does not revert."""
    n_obs: int
    """Bars used after aligning the two series."""


def log_prices(price_a: pd.Series, price_b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align two price series on their common timestamps and take logs.

    Rows where either price is missing are dropped.

    Args:
        price_a: Prices of leg A, indexed by timestamp.
        price_b: Prices of leg B, indexed by timestamp.

    Returns:
        (log_a, log_b): natural-log prices on their shared timestamps only.

    Raises:
        ValueError: If a price is zero or negative.
    """
    joined = pd.concat([price_a, price_b], axis=1, join='inner').dropna()
    if (joined <= 0).any().any():
        raise ValueError("Prices must be positive to take logs")
    logs = np.log(joined.astype('float64'))
    return logs.iloc[:, 0], logs.iloc[:, 1]


def _ols(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
    """
    Fit a straight line y = intercept + slope * x by least squares.

    Args:
        y: Dependent values (e.g. log prices of leg A).
        x: Explanatory values of the same length (e.g. log prices of leg B).

    Returns:
        (intercept, slope, residuals, r_squared). R² is NaN when y is
        constant.
    """
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)
    total = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - np.sum(residuals ** 2) / total if total > 0 else float('nan')
    return float(intercept), float(slope), residuals, float(r_squared)


def half_life(spread: np.ndarray) -> float:
    """
    Estimate how fast a spread reverts to its mean.

    Regresses the change in the spread on its previous level; a negative
    slope means deviations shrink over time.

    Args:
        spread: Spread values in time order (at least 3).

    Returns:
        Bars for a deviation to halve, e.g. 10.0 means half the gap closes
        in about 10 bars. `math.inf` when the spread does not revert.
    """
    spread = np.asarray(spread, dtype='float64')
    _, slope, _, _ = _ols(np.diff(spread), spread[:-1])
    return -math.log(2) / slope if slope < 0 else math.inf


def check_cointegration(
    price_a: pd.Series, price_b: pd.Series, significance: float = 0.05
) -> CointegrationResult:
    """
    Engle-Granger cointegration test on log prices.

    Run it on a formation period before trading the pair, and trade on data
    after that period; testing and trading on the same bars overstates the
    edge.

    Args:
        price_a: Prices of the leg that is regressed (A).
        price_b: Prices of the hedge leg (B).
        significance: p-value below which the pair counts as cointegrated.

    Returns:
        CointegrationResult with the test outcome and the static hedge ratio.

    Raises:
        ValueError: If fewer than 60 aligned bars remain or a price is not
            positive.
    """
    from statsmodels.tsa.stattools import coint

    log_a, log_b = log_prices(price_a, price_b)
    if len(log_a) < MIN_COINTEGRATION_OBS:
        raise ValueError(
            f"Need at least {MIN_COINTEGRATION_OBS} aligned bars, got {len(log_a)}"
        )
    y, x = log_a.to_numpy(), log_b.to_numpy()
    t_statistic, p_value, critical_values = coint(y, x, trend='c')
    intercept, hedge_ratio, residuals, r_squared = _ols(y, x)
    return CointegrationResult(
        is_cointegrated=bool(p_value < significance),
        p_value=float(p_value),
        t_statistic=float(t_statistic),
        critical_value_5pct=float(critical_values[1]),
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        r_squared=r_squared,
        half_life=half_life(residuals),
        n_obs=len(y),
    )


def rolling_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int,
    min_window: int = MIN_ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Out-of-sample spread and z-score from a rolling regression.

    The regression of log A on log B is fitted on the `window` bars ending
    at the previous bar, then applied to the current bar. The z-score is the
    current residual divided by the window's residual standard deviation.

    Args:
        price_a: Prices of leg A.
        price_b: Prices of leg B.
        window: Bars in each regression.
        min_window: Smallest `window` accepted (default 60, never below 3).
            Short windows fit noise, so the hedge ratio jumps around.

    Returns:
        DataFrame on the aligned timestamps with columns hedge_ratio,
        intercept, spread and z_score; NaN for the first `window` bars.

    Raises:
        ValueError: If `window` is below `min_window`, `min_window` is
            below 3, or a price is not positive.
    """
    _check_window(window, min_window, 'window')
    y, x = log_prices(price_a, price_b)
    var_x = x.rolling(window).var()
    cov = x.rolling(window).cov(y)
    beta = cov / var_x.where(var_x > 0)
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    resid_var = (y.rolling(window).var() - beta * cov) * (window - 1) / (window - 2)
    resid_std = np.sqrt(resid_var.clip(lower=0))

    beta, alpha, resid_std = beta.shift(1), alpha.shift(1), resid_std.shift(1)
    spread = y - alpha - beta * x
    return pd.DataFrame({
        'hedge_ratio': beta,
        'intercept': alpha,
        'spread': spread,
        'z_score': spread / resid_std.where(resid_std > 0),
    })


def _next_position(position: int, z: float, entry: float, exit: float, stop: float) -> int:
    """
    One step of the pairs state machine.

    Args:
        position: Current position: +1 long the spread (long A, short B),
            -1 short the spread, 0 flat.
        z: This bar's spread z-score (never NaN here).
        entry: Open when |z| reaches this.
        exit: Close a trade once the spread has come back to within `exit`
            of the mean or gone past it: a long closes when z >= -exit, a
            short when z <= exit. So exit=0 means "close on crossing the
            mean", and a one-bar jump across the mean also closes.
        stop: Close a trade (and never open one) when |z| reaches this.

    Returns:
        The position to hold after this bar: -1, 0 or +1.
    """
    # 1. Stop-loss: close any existing position.
    # Do not open a new trade beyond the stop threshold.
    if abs(z) >= stop:
        return 0
    
    # 2. Directional exit for existing positions
    if position == 1 and z >= -exit:
        return 0

    if position == -1 and z <= exit:
        return 0
    
    # 3. Continue holding an existing position
    if position != 0:
        return position

    # 4. Enter a new position if currently flat
    if z <= -entry:
        return 1

    if z >= entry:
        return -1

    return 0

def hold_positions(
    z_score: pd.Series, entry: float = 2.0, exit: float = 0.5, stop: float = 3.5
) -> pd.Series:
    """
    Walk the z-score bar by bar and keep the position between entry and exit.

    Bars with a NaN z-score keep the current position (flat at the start).

    Args:
        z_score: Spread z-score per bar.
        entry: |z| that opens a trade.
        exit: |z| at or below which an open trade closes.
        stop: |z| at or above which an open trade closes at a loss.

    Returns:
        Integer Series (-1, 0, +1) on the same index, named 'position'.

    Raises:
        ValueError: Unless 0 <= exit < entry < stop.
    """
    if not 0 <= exit < entry < stop:
        raise ValueError("Thresholds must satisfy 0 <= exit < entry < stop")
    positions = np.zeros(len(z_score), dtype='int64')
    position = 0
    for i, z in enumerate(z_score.to_numpy(dtype='float64')):
        if not np.isnan(z):
            position = _next_position(position, float(z), entry, exit, stop)
        positions[i] = position
    return pd.Series(positions, index=z_score.index, name='position')


def pair_weights(
    position: pd.Series, hedge_ratio: pd.Series, symbol_a: str, symbol_b: str
) -> pd.DataFrame:
    """
    Dollar weights for a spread position, scaled to gross exposure 1.

    Long the spread holds 1 / (1 + |b|) in A and -b / (1 + |b|) in B, where
    b is the hedge ratio on that bar. Bars without a position or hedge ratio
    (missing prices) keep the previous values.

    Args:
        position: Spread position per bar: +1 long, -1 short, 0 flat.
        hedge_ratio: Units of log B per unit of log A on each bar.
        symbol_a: Column name for leg A in the output.
        symbol_b: Column name for leg B in the output.

    Returns:
        Weights panel with columns `symbol_a` and `symbol_b`; |w_a| + |w_b|
        is 1 when in a trade and 0 when flat (or before the first hedge
        ratio).
    """
    position = position.ffill().fillna(0.0)
    hedge_ratio = hedge_ratio.ffill()
    gross = 1.0 + hedge_ratio.abs()
    return pd.DataFrame({
        symbol_a: position / gross,
        symbol_b: -position * hedge_ratio / gross,
    }).fillna(0.0)


class PairsTradingStrategy(BaseStrategy):
    """Mean reversion on the rolling-regression spread of two symbols."""

    name = 'pairs_trading'
    version = '1.0.0'

    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        lookback: int = 100,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
        min_window: int = MIN_ROLLING_WINDOW,
    ):
        """
        Store and validate the pair and its trading thresholds.

        Args:
            symbol_a: Leg A (regressed on B).
            symbol_b: Leg B (the hedge).
            lookback: Bars in the rolling hedge-ratio regression. 100-120
                daily bars are worth testing; longer windows are steadier but
                react more slowly when the relationship changes.
            entry_z: |z| that opens a trade.
            exit_z: |z| at or below which a trade closes.
            stop_z: |z| at or above which a trade closes at a loss.
            min_window: Smallest `lookback` accepted (default 60, never
                below 3). Lower it only on purpose, e.g. for intraday bars.

        Raises:
            ValueError: If a parameter is out of range (see `validate`).
        """
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.min_window = min_window
        self.validate()

    def get_parameters(self) -> Dict[str, Any]:
        """
        Report the parameters for experiment tracking.

        Returns:
            Dict of symbol_a, symbol_b, lookback, entry_z, exit_z, stop_z
            and min_window.
        """
        return {
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'lookback': self.lookback,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'stop_z': self.stop_z,
            'min_window': self.min_window,
        }

    def validate(self) -> bool:
        """
        Check parameter ranges.

        Returns:
            True when every parameter is valid.

        Raises:
            ValueError: If the symbols are equal, lookback is below
                min_window (or min_window below 3), or the thresholds are
                not 0 <= exit_z < entry_z < stop_z.
        """
        if self.symbol_a == self.symbol_b:
            raise ValueError("symbol_a and symbol_b must differ")
        _check_window(self.lookback, self.min_window, 'lookback')
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("Thresholds must satisfy 0 <= exit_z < entry_z < stop_z")
        return True

    def spread(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling hedge ratio, spread, z-score and position for the pair.

        Args:
            prices: Close-price panel containing both symbols.

        Returns:
            DataFrame on the timestamps where both symbols have prices,
            with columns hedge_ratio, intercept, spread, z_score (see
            `rolling_spread`) and position (-1, 0, +1).
        """
        frame = rolling_spread(
            prices[self.symbol_a], prices[self.symbol_b], self.lookback, self.min_window
        )
        frame['position'] = hold_positions(
            frame['z_score'], self.entry_z, self.exit_z, self.stop_z
        )
        return frame

    def target_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Weights for the two legs on every bar of the panel.

        Args:
            prices: Close-price panel containing both symbols.

        Returns:
            Weights panel on `prices.index` with columns symbol_a and
            symbol_b (see `pair_weights`).
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

        Returns:
            Signal rows with columns timestamp, symbol, signal_type and
            strength (see `weights_to_signals`).
        """
        return weights_to_signals(self.target_weights(to_panel(data)))
