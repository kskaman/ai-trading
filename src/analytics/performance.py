"""
Performance metrics for periodic strategy returns (Intelligent Markets, ch. 4).

    analyzer = PerformanceAnalyzer.for_interval(daily_returns, '1d', 'equity')
    analyzer.summary(n_trials=20)

Returns are simple per-bar returns (0.01 = +1%). NaN values, such as the
first bar of a backtest, are dropped.

Changes from the book's PerformanceAnalyzer:
    * Sortino divides by the downside deviation over every period (losses
      below the risk-free rate, gains counted as zero). The book mixed an
      annual return with a per-period risk-free rate and used the standard
      deviation of the negative periods only.
    * The deflated Sharpe ratio is computed in per-period units and returned
      as a probability, following Bailey and Lopez de Prado (2014). The book
      subtracted a per-observation standard error from an annualized Sharpe.
    * Max drawdown counts a loss in the first period (the equity curve starts
      at 1.0).
    * Win rate, average win/loss and profit factor ignore flat periods
      (exactly zero return), which would otherwise count as losses.
    * Normal-distribution functions come from the standard library, not scipy.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.market_data.schema import periods_per_year as bars_per_year

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def _clean(returns: pd.Series) -> pd.Series:
    series = pd.Series(returns, dtype='float64').dropna()
    if len(series) < 2:
        raise ValueError("Need at least 2 non-NaN returns")
    return series


def _std(series: pd.Series) -> float:
    """Sample std; exactly 0 for a constant series (no rounding residue)."""
    return 0.0 if series.min() == series.max() else float(series.std(ddof=1))


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator; inf for x/0 with x > 0, NaN for other x/0."""
    if denominator == 0 or np.isnan(denominator):
        return math.inf if numerator > 0 else math.nan
    return float(numerator / denominator)


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """
    Expected highest Sharpe ratio among `n_trials` strategies whose true
    Sharpe is zero, given the spread of their estimated Sharpe ratios.

    This is the bar a strategy picked as the best of n_trials has to clear
    before its Sharpe means anything.

    Args:
        n_trials: Number of strategy variants tried (at least 1).
        sharpe_std: Standard deviation of the per-period Sharpe ratios across
            those variants.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    first = _NORMAL.inv_cdf(1 - 1 / n_trials)
    second = _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    return sharpe_std * ((1 - _EULER_GAMMA) * first + _EULER_GAMMA * second)


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """
    Probability that the true Sharpe ratio exceeds `benchmark_sharpe`.

    Accounts for track-record length, skewness and fat tails. Both Sharpe
    ratios are per period (not annualized).

    Args:
        returns: Per-period returns (excess returns if a risk-free rate
            applies).
        benchmark_sharpe: Per-period Sharpe ratio to beat.

    Returns:
        Probability between 0 and 1; NaN if returns have no variance.
    """
    r = _clean(returns)
    std = _std(r)
    if std == 0:
        return math.nan
    sharpe = r.mean() / std
    kurtosis = r.kurt() + 3.0  # pandas reports excess kurtosis
    variance = 1 - r.skew() * sharpe + (kurtosis - 1) / 4 * sharpe ** 2
    if not variance > 0:
        return math.nan
    z = (sharpe - benchmark_sharpe) * math.sqrt(len(r) - 1) / math.sqrt(variance)
    return _NORMAL.cdf(z)


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, trials_sharpe_std: Optional[float] = None
) -> float:
    """
    Probabilistic Sharpe ratio against the best Sharpe expected by luck.

    Use it when `returns` belong to the best of several backtested variants
    (parameter sweeps count). Above 0.95 is the usual bar for "probably not
    a fluke".

    Args:
        returns: Per-period returns of the selected strategy.
        n_trials: How many variants were tried, including this one.
        trials_sharpe_std: Standard deviation of the per-period Sharpe ratios
            of all variants. If unknown, the sampling error of a zero Sharpe,
            1 / sqrt(T - 1), is used; that is the mildest correction, so
            pass the real figure when you have it.

    Returns:
        Probability between 0 and 1.
    """
    r = _clean(returns)
    if trials_sharpe_std is None:
        trials_sharpe_std = 1 / math.sqrt(len(r) - 1)
    return probabilistic_sharpe_ratio(r, expected_max_sharpe(n_trials, trials_sharpe_std))


def drawdowns(returns: pd.Series) -> pd.Series:
    """Fall from the running peak of the equity curve, as a fraction (<= 0)."""
    r = _clean(returns)
    equity = (1 + r).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1


class PerformanceAnalyzer:
    """Return, risk and robustness metrics for one strategy's returns."""

    def __init__(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252.0,
    ):
        """
        Args:
            returns: Per-period simple returns.
            risk_free_rate: Annual risk-free rate (0.045 = 4.5%). Leave at 0
                for dollar-neutral long/short books, whose cash already earns
                the short rebate.
            periods_per_year: Bars in a year (see `for_interval`).

        Raises:
            ValueError: If fewer than 2 returns remain after dropping NaN.
        """
        self.returns = _clean(returns)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = float(periods_per_year)
        self._rf_period = (1 + risk_free_rate) ** (1 / self.periods_per_year) - 1
        self.excess_returns = self.returns - self._rf_period

    @classmethod
    def for_interval(
        cls,
        returns: pd.Series,
        interval: str = '1d',
        market: str = 'equity',
        risk_free_rate: float = 0.0,
    ) -> 'PerformanceAnalyzer':
        """Build an analyzer whose annualization matches the bar interval."""
        return cls(returns, risk_free_rate, bars_per_year(interval, market))

    @property
    def n_periods(self) -> int:
        """Number of returns analyzed."""
        return len(self.returns)

    @property
    def years(self) -> float:
        """Length of the track record in years."""
        return self.n_periods / self.periods_per_year

    def total_return(self) -> float:
        """Compounded return over the whole period."""
        return float((1 + self.returns).prod() - 1)

    def annual_return(self) -> float:
        """Compound annual growth rate."""
        total = self.total_return()
        return -1.0 if total <= -1 else (1 + total) ** (1 / self.years) - 1

    def annual_volatility(self) -> float:
        """Annualized standard deviation of returns."""
        return float(self.returns.std(ddof=1) * math.sqrt(self.periods_per_year))

    def sharpe_ratio(self) -> float:
        """Annualized mean excess return over volatility."""
        ratio = _ratio(self.excess_returns.mean(), _std(self.returns))
        if math.isinf(ratio):
            return math.nan  # constant returns: Sharpe is undefined
        return ratio * math.sqrt(self.periods_per_year)

    def downside_deviation(self) -> float:
        """Annualized root-mean-square of returns below the risk-free rate."""
        shortfall = self.excess_returns.clip(upper=0)
        return float(np.sqrt((shortfall ** 2).mean()) * math.sqrt(self.periods_per_year))

    def sortino_ratio(self) -> float:
        """Annualized mean excess return over downside deviation."""
        annual_excess = self.excess_returns.mean() * self.periods_per_year
        return _ratio(annual_excess, self.downside_deviation())

    def max_drawdown(self) -> float:
        """Deepest peak-to-trough fall, as a negative fraction."""
        return float(drawdowns(self.returns).min())

    def calmar_ratio(self) -> float:
        """Annual return over the absolute max drawdown."""
        return _ratio(self.annual_return(), abs(self.max_drawdown()))

    def _active(self) -> pd.Series:
        return self.returns[self.returns != 0]

    def win_rate(self) -> float:
        """Share of non-flat periods with a positive return."""
        active = self._active()
        return float((active > 0).mean()) if len(active) else math.nan

    def average_win(self) -> float:
        """Mean of the positive returns."""
        wins = self.returns[self.returns > 0]
        return float(wins.mean()) if len(wins) else math.nan

    def average_loss(self) -> float:
        """Mean of the negative returns (a negative number)."""
        losses = self.returns[self.returns < 0]
        return float(losses.mean()) if len(losses) else math.nan

    def profit_factor(self) -> float:
        """Sum of gains over the absolute sum of losses."""
        gains = self.returns[self.returns > 0].sum()
        losses = -self.returns[self.returns < 0].sum()
        return _ratio(gains, losses)

    def omega_ratio(self, threshold: Optional[float] = None) -> float:
        """
        Gains above `threshold` over losses below it, per period.

        Args:
            threshold: Per-period return threshold; defaults to the
                per-period risk-free rate.
        """
        threshold = self._rf_period if threshold is None else threshold
        gains = (self.returns - threshold).clip(lower=0).sum()
        losses = (threshold - self.returns).clip(lower=0).sum()
        return _ratio(gains, losses)

    def probabilistic_sharpe(self, benchmark_sharpe: float = 0.0) -> float:
        """
        Probability the true Sharpe beats an annualized benchmark.

        Args:
            benchmark_sharpe: Annualized Sharpe ratio to beat.
        """
        per_period = benchmark_sharpe / math.sqrt(self.periods_per_year)
        return probabilistic_sharpe_ratio(self.excess_returns, per_period)

    def deflated_sharpe(
        self, n_trials: int = 1, trials_sharpe_std: Optional[float] = None
    ) -> float:
        """
        Deflated Sharpe ratio (see `deflated_sharpe_ratio`).

        Args:
            n_trials: Variants tried before choosing this one.
            trials_sharpe_std: Std of the variants' annualized Sharpe ratios.
        """
        if trials_sharpe_std is not None:
            trials_sharpe_std /= math.sqrt(self.periods_per_year)
        return deflated_sharpe_ratio(self.excess_returns, n_trials, trials_sharpe_std)

    def summary(
        self, n_trials: int = 1, trials_sharpe_std: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Every metric in one dictionary.

        Args:
            n_trials: Variants tried, for the deflated Sharpe ratio.
            trials_sharpe_std: Std of the variants' annualized Sharpe ratios.
        """
        return {
            'total_return': self.total_return(),
            'annual_return': self.annual_return(),
            'annual_volatility': self.annual_volatility(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'max_drawdown': self.max_drawdown(),
            'calmar_ratio': self.calmar_ratio(),
            'win_rate': self.win_rate(),
            'average_win': self.average_win(),
            'average_loss': self.average_loss(),
            'profit_factor': self.profit_factor(),
            'omega_ratio': self.omega_ratio(),
            'probabilistic_sharpe': self.probabilistic_sharpe(),
            'deflated_sharpe': self.deflated_sharpe(n_trials, trials_sharpe_std),
            'n_periods': float(self.n_periods),
            'years': self.years,
        }
