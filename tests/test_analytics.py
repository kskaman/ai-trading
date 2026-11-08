"""Performance metrics and the vectorized backtest."""

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.performance import (
    PerformanceAnalyzer,
    deflated_sharpe_ratio,
    drawdowns,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from src.analytics.portfolio import portfolio_returns
from src.config.settings import RiskSettings
from tests.factories import make_panel


def noisy_returns(mean: float, n: int = 1000, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.01, n))


# --- performance ----------------------------------------------------------

def test_basic_return_metrics():
    analyzer = PerformanceAnalyzer(pd.Series([0.001] * 252))
    assert analyzer.total_return() == pytest.approx(1.001 ** 252 - 1)
    assert analyzer.annual_return() == pytest.approx(1.001 ** 252 - 1)
    assert analyzer.years == pytest.approx(1.0)
    assert analyzer.max_drawdown() == 0.0
    assert math.isnan(analyzer.sharpe_ratio())  # no variance


def test_sortino_uses_downside_deviation_over_all_periods():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    analyzer = PerformanceAnalyzer(returns, periods_per_year=252)

    downside = math.sqrt((0.01 ** 2 + 0.02 ** 2) / 4)
    assert analyzer.sortino_ratio() == pytest.approx(0.005 / downside * math.sqrt(252))


def test_max_drawdown_counts_a_first_period_loss():
    assert PerformanceAnalyzer(pd.Series([-0.1, 0.05])).max_drawdown() == pytest.approx(-0.1)
    dd = drawdowns(pd.Series([0.1, -0.2, 0.1]))
    assert dd.tolist() == pytest.approx([0.0, -0.2, -0.12])


def test_trade_statistics_ignore_flat_periods():
    analyzer = PerformanceAnalyzer(pd.Series([0.01, 0.0, 0.0, -0.01, 0.02]))
    assert analyzer.win_rate() == pytest.approx(2 / 3)
    assert analyzer.average_win() == pytest.approx(0.015)
    assert analyzer.average_loss() == pytest.approx(-0.01)
    assert analyzer.profit_factor() == pytest.approx(3.0)
    assert PerformanceAnalyzer(pd.Series([0.01, 0.02])).profit_factor() == math.inf


def test_omega_ratio_at_zero_threshold_equals_profit_factor():
    analyzer = PerformanceAnalyzer(noisy_returns(0.001))
    assert analyzer.omega_ratio(0.0) == pytest.approx(analyzer.profit_factor())


def test_risk_free_rate_lowers_sharpe():
    returns = noisy_returns(0.0005)
    assert (PerformanceAnalyzer(returns, risk_free_rate=0.05).sharpe_ratio()
            < PerformanceAnalyzer(returns).sharpe_ratio())


def test_for_interval_uses_bars_per_year():
    returns = noisy_returns(0.0, n=10)
    assert PerformanceAnalyzer.for_interval(returns, '1d', 'crypto').periods_per_year == 365
    assert PerformanceAnalyzer.for_interval(returns, '1h', 'crypto').periods_per_year == 8760


def test_probabilistic_sharpe_reflects_evidence():
    assert probabilistic_sharpe_ratio(noisy_returns(0.003)) > 0.99
    assert probabilistic_sharpe_ratio(noisy_returns(-0.003)) < 0.01
    zero_mean = noisy_returns(0.0)
    assert probabilistic_sharpe_ratio(zero_mean - zero_mean.mean()) == pytest.approx(0.5)


def test_deflated_sharpe_shrinks_with_more_trials():
    returns = noisy_returns(0.0008)
    one = deflated_sharpe_ratio(returns, n_trials=1)
    many = deflated_sharpe_ratio(returns, n_trials=100)
    assert one == pytest.approx(probabilistic_sharpe_ratio(returns))
    assert many < one


def test_expected_max_sharpe_approximates_the_maximum_of_normals():
    assert expected_max_sharpe(1, 1.0) == 0.0
    rng = np.random.default_rng(0)
    simulated = rng.normal(size=(20_000, 10)).max(axis=1).mean()
    assert expected_max_sharpe(10, 1.0) == pytest.approx(simulated, rel=0.05)
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)


def test_summary_has_every_metric_and_drops_nan():
    returns = noisy_returns(0.0005, n=300)
    returns.iloc[0] = np.nan
    summary = PerformanceAnalyzer(returns).summary(n_trials=5)
    assert summary['n_periods'] == 299
    assert {'sharpe_ratio', 'sortino_ratio', 'deflated_sharpe', 'calmar_ratio'} <= set(summary)


def test_analyzer_needs_two_returns():
    with pytest.raises(ValueError):
        PerformanceAnalyzer(pd.Series([0.01, np.nan]))


# --- vectorized backtest --------------------------------------------------

def test_weights_earn_the_next_bars_return():
    prices = make_panel(4, symbols=('AAA', 'BBB'))
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.iloc[1:, 0] = 1.0  # decide to hold AAA at bar 1's close

    result = portfolio_returns(weights, prices, cost_bps=0)

    asset = prices['AAA'].pct_change()
    assert result['gross_return'].iloc[:2].tolist() == [0.0, 0.0]
    assert result['gross_return'].iloc[2:].tolist() == pytest.approx(asset.iloc[2:].tolist())


def test_costs_scale_with_turnover():
    prices = make_panel(4, symbols=('AAA', 'BBB'))
    weights = pd.DataFrame({'AAA': [0.5, 0.5, -0.5, -0.5], 'BBB': [-0.5, -0.5, 0.5, 0.5]},
                           index=prices.index)

    result = portfolio_returns(weights, prices, cost_bps=10)

    assert result['turnover'].tolist() == pytest.approx([0.0, 1.0, 0.0, 2.0])
    assert result['cost'].tolist() == pytest.approx([0.0, 0.001, 0.0, 0.002])
    assert (result['net_return'] == result['gross_return'] - result['cost']).all()


def test_default_cost_comes_from_risk_settings(monkeypatch):
    prices = make_panel(3, symbols=('AAA',))
    weights = pd.DataFrame({'AAA': [1.0, 1.0, 1.0]}, index=prices.index)

    assert portfolio_returns(weights, prices, settings=RiskSettings(slippage_bps=20))[
        'cost'].iloc[1] == pytest.approx(0.002)
    monkeypatch.setenv('RISK_SLIPPAGE_BPS', '7')
    assert portfolio_returns(weights, prices)['cost'].iloc[1] == pytest.approx(0.0007)


def test_portfolio_returns_needs_prices_for_every_weight():
    prices = make_panel(3, symbols=('AAA',))
    with pytest.raises(ValueError, match='BBB'):
        portfolio_returns(pd.DataFrame({'BBB': [1.0] * 3}, index=prices.index), prices)
