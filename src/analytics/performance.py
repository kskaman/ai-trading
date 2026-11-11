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
    """
    Prepare returns for the metrics: float64, NaN dropped.

    Args:
        returns: Per-period simple returns (a Series or anything
            `pd.Series` accepts).

    Returns:
        The non-NaN returns as a float64 Series, original index kept.

    Raises:
        ValueError: If fewer than 2 returns remain.
    """
    series = pd.Series(returns, dtype='float64').dropna()
    if len(series) < 2:
        raise ValueError("Need at least 2 non-NaN returns")
    return series


def _std(series: pd.Series) -> float:
    """
    Sample standard deviation that is exactly 0 for a constant series.

    pandas can return a tiny non-zero value for constant input because of
    rounding, which would turn an undefined ratio into a huge number.

    Args:
        series: At least 2 values.

    Returns:
        The ddof=1 standard deviation, or 0.0 when all values are equal.
    """
    return 0.0 if series.min() == series.max() else float(series.std(ddof=1))


def _ratio(numerator: float, denominator: float) -> float:
    """
    Divide, with defined results when the denominator is 0 or NaN.

    Args:
        numerator: Top of the ratio, e.g. total gains.
        denominator: Bottom of the ratio, e.g. total losses.

    Returns:
        numerator / denominator. When the denominator is 0 or NaN: inf if
        the numerator is positive (e.g. gains but no losses), NaN otherwise.
    """
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

    Returns:
        Expected maximum per-period Sharpe ratio from luck alone; 0.0 when
        only one variant was tried.

    Raises:
        ValueError: If n_trials < 1.
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
    """
    Measure how far the equity curve sits below its running peak.

    The curve starts at 1.0, so a loss in the very first period already
    counts as a drawdown.

    Args:
        returns: Per-period simple returns; NaN values are dropped.

    Returns:
        Drawdown per period as a fraction <= 0, e.g. -0.25 means 25% below
        the peak; 0 at new highs.

    Raises:
        ValueError: If fewer than 2 non-NaN returns are given.
    """
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
        Store the returns and convert the risk-free rate to per period.

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
        """
        Build an analyzer whose annualization matches the bar interval.

        Args:
            returns: Per-period simple returns.
            interval: Bar size of the returns, e.g. '1d' or '1h'.
            market: 'equity' (252 sessions a year) or 'crypto' (365 days).
            risk_free_rate: Annual risk-free rate (0.045 = 4.5%).

        Returns:
            An analyzer with `periods_per_year` from
            `schema.periods_per_year(interval, market)`.

        Raises:
            ValueError: If the interval or market is unsupported, or fewer
                than 2 returns remain.
        """
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
        """
        Compounded return over the whole period.

        Returns:
            Growth of 1 unit minus 1, e.g. 0.30 = +30%, -1.0 = wiped out.
        """
        return float((1 + self.returns).prod() - 1)

    def annual_return(self) -> float:
        """
        Compound annual growth rate.

        Returns:
            The yearly rate that compounds to `total_return` over `years`,
            e.g. 0.12 = +12% a year; -1.0 if the strategy lost everything.
        """
        total = self.total_return()
        return -1.0 if total <= -1 else (1 + total) ** (1 / self.years) - 1

    def annual_volatility(self) -> float:
        """
        Annualized standard deviation of returns.

        Returns:
            Per-period std times sqrt(periods_per_year), e.g. 0.20 = 20%.
        """
        return float(self.returns.std(ddof=1) * math.sqrt(self.periods_per_year))

    def sharpe_ratio(self) -> float:
        """
        Annualized mean excess return over volatility.

        Returns:
            The annualized Sharpe ratio (above 1 is good, above 2 rare);
            NaN when returns are constant.
        """
        ratio = _ratio(self.excess_returns.mean(), _std(self.returns))
        if math.isinf(ratio):
            return math.nan  # constant returns: Sharpe is undefined
        return ratio * math.sqrt(self.periods_per_year)

    def downside_deviation(self) -> float:
        """
        Annualized root-mean-square of returns below the risk-free rate.

        Gains count as zero rather than being left out, so every period is
        in the average.

        Returns:
            Annualized downside deviation, e.g. 0.10 = 10%; 0.0 if no
            period fell below the risk-free rate.
        """
        shortfall = self.excess_returns.clip(upper=0)
        return float(np.sqrt((shortfall ** 2).mean()) * math.sqrt(self.periods_per_year))

    def sortino_ratio(self) -> float:
        """
        Annualized mean excess return over downside deviation.

        Returns:
            The Sortino ratio; inf if there were gains but no shortfall,
            NaN if neither.
        """
        annual_excess = self.excess_returns.mean() * self.periods_per_year
        return _ratio(annual_excess, self.downside_deviation())

    def max_drawdown(self) -> float:
        """
        Deepest peak-to-trough fall.

        Returns:
            A fraction <= 0, e.g. -0.35 = the worst fall was 35%.
        """
        return float(drawdowns(self.returns).min())

    def calmar_ratio(self) -> float:
        """
        Annual return over the absolute max drawdown.

        Returns:
            Calmar ratio, e.g. 0.5 = a year's return is half the worst
            fall; inf if there was no drawdown and the return is positive.
        """
        return _ratio(self.annual_return(), abs(self.max_drawdown()))

    def _active(self) -> pd.Series:
        """
        Drop flat periods, which would otherwise count as losses.

        Returns:
            The returns that are not exactly zero.
        """
        return self.returns[self.returns != 0]

    def win_rate(self) -> float:
        """
        Share of non-flat periods with a positive return.

        Returns:
            A fraction between 0 and 1, e.g. 0.55 = 55% of trading periods
            made money; NaN if every period was flat.
        """
        active = self._active()
        return float((active > 0).mean()) if len(active) else math.nan

    def average_win(self) -> float:
        """
        Mean of the positive returns.

        Returns:
            A per-period return > 0, e.g. 0.012 = +1.2%; NaN if none.
        """
        wins = self.returns[self.returns > 0]
        return float(wins.mean()) if len(wins) else math.nan

    def average_loss(self) -> float:
        """
        Mean of the negative returns.

        Returns:
            A per-period return < 0, e.g. -0.009 = -0.9%; NaN if none.
        """
        losses = self.returns[self.returns < 0]
        return float(losses.mean()) if len(losses) else math.nan

    def profit_factor(self) -> float:
        """
        Sum of gains over the absolute sum of losses.

        Returns:
            Above 1 means gains outweigh losses, e.g. 1.5; inf if there
            were no losing periods.
        """
        gains = self.returns[self.returns > 0].sum()
        losses = -self.returns[self.returns < 0].sum()
        return _ratio(gains, losses)

    def omega_ratio(self, threshold: Optional[float] = None) -> float:
        """
        Gains above `threshold` over losses below it, per period.

        Args:
            threshold: Per-period return threshold; defaults to the
                per-period risk-free rate.

        Returns:
            Above 1 means more return above the threshold than below it;
            inf if no period fell below it.
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

        Returns:
            Probability between 0 and 1 (see `probabilistic_sharpe_ratio`);
            NaN if returns have no variance.
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

        Returns:
            Probability between 0 and 1 that the edge is real after
            allowing for the number of trials; above 0.95 is the usual bar.
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

        Returns:
            Metric name -> value, one entry per method above (total_return
            through deflated_sharpe) plus n_periods and years.
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
