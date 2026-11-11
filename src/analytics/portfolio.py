"""
Vectorized research backtest: weights + prices -> portfolio returns.

Good for comparing strategy ideas quickly. It trades at close prices with a
flat cost per unit of turnover and rebalances to the target weights every
bar; the event-driven backtester (fills, partial orders, cash) is a later
chapter.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config.settings import RiskSettings, get_settings


def portfolio_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: Optional[float] = None,
    settings: Optional[RiskSettings] = None,
) -> pd.DataFrame:
    """
    Per-bar returns of holding `weights`, traded one bar late.

    Weights decided at bar t's close earn bar t+1's return, so a strategy
    never trades on the price that produced its signal. Changing weights
    costs `cost_bps` per unit of weight traded (1.0 = the whole portfolio).
    A missing price counts as a zero return for that symbol and bar.

    Holdings are assumed to sit exactly at the target weights on every bar.
    Real holdings drift as prices move, and trading them back costs money;
    neither the drift nor that cost is modeled here. Tracking actual shares
    and cash belongs to an event-driven backtester.

    Args:
        weights: Weights panel (see src.strategies.weights).
        prices: Close-price panel covering every symbol in `weights`.
        cost_bps: Cost per unit of turnover in basis points; defaults to
            RISK_SLIPPAGE_BPS.
        settings: Risk settings used for the default cost.

    Returns:
        DataFrame on `prices.index` with columns gross_return, turnover,
        cost and net_return (net = gross - cost).

    Raises:
        ValueError: If `weights` has a symbol that `prices` does not.
    """
    missing = sorted(set(weights.columns) - set(prices.columns))
    if missing:
        raise ValueError(f"No prices for weighted symbols {missing}")
    if cost_bps is None:
        settings = settings if settings is not None else get_settings().risk
        cost_bps = settings.slippage_bps

    prices = prices[list(weights.columns)]
    weights = weights.reindex(prices.index).ffill().fillna(0.0)
    held = weights.shift(1, fill_value=0.0)
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)

    gross = (held * asset_returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)  # first row: held starts flat
    cost = turnover * cost_bps / 10_000
    return pd.DataFrame({
        'gross_return': gross,
        'turnover': turnover,
        'cost': cost,
        'net_return': gross - cost,
    })
