"""
Target weights: the common output of every strategy.

A strategy's main output is a weights frame shaped like a price panel (see
src.market_data.panel): one row per bar, one column per symbol.

    weight > 0   long that fraction of capital
    weight < 0   short that fraction of capital
    row total    sum of |weight| is 1 when invested, 0 when flat

Leverage and volatility targeting are the risk layer's job, not the
strategy's. Row t uses data up to and including bar t's close, so trade it on
the next bar; src.analytics.portfolio.portfolio_returns does that shift.

`weights_to_signals` turns weights into the event rows that BaseStrategy
promises (LONG_ENTER, LONG_EXIT, SHORT_ENTER, SHORT_EXIT).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = ('timestamp', 'symbol', 'signal_type', 'strength')


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each row across symbols: (value - row mean) / row std.

    Rows with fewer than two values, or no spread at all, become NaN.
    """
    std = frame.std(axis=1, ddof=1)
    return frame.sub(frame.mean(axis=1), axis=0).div(std.where(std > 0), axis=0)


def long_short_weights(
    scores: pd.DataFrame,
    n_long: int,
    n_short: int,
    rebalance_every: int,
) -> pd.DataFrame:
    """
    Equal-weight the highest scores long and the lowest short.

    Rebalances on the first bar where at least `n_long + n_short` symbols
    have a score, then every `rebalance_every` bars, and holds the same
    weights in between. A scheduled rebalance with too few scores keeps the
    previous holdings. With both sides the book is dollar neutral (+0.5 long,
    -0.5 short); with one side that side gets the full 1.0. Ties rank in
    column order.

    Args:
        scores: Panel of scores; higher means more attractive.
        n_long: Number of symbols to buy (0 for short-only).
        n_short: Number of symbols to sell short (0 for long-only).
        rebalance_every: Bars between rebalances.

    Returns:
        Weights panel with the same index and columns as `scores`.

    Raises:
        ValueError: If counts are negative, both are zero, `rebalance_every`
            is below 1, or there are fewer columns than `n_long + n_short`.
    """
    if n_long < 0 or n_short < 0 or n_long + n_short == 0:
        raise ValueError("n_long and n_short must be >= 0 and not both 0")
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be >= 1")
    needed = n_long + n_short
    if scores.shape[1] < needed:
        raise ValueError(
            f"Need at least n_long + n_short = {needed} symbols, "
            f"got {scores.shape[1]}"
        )

    long_weight = (0.5 if n_short else 1.0) / n_long if n_long else 0.0
    short_weight = (0.5 if n_long else 1.0) / n_short if n_short else 0.0

    weights = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)
    eligible = np.flatnonzero((scores.notna().sum(axis=1) >= needed).to_numpy())
    if eligible.size:
        for pos in range(eligible[0], len(scores), rebalance_every):
            ranked = scores.iloc[pos].dropna().sort_values(kind='stable')
            if len(ranked) < needed:
                continue
            target = pd.Series(0.0, index=scores.columns)
            if n_short:
                target[ranked.index[:n_short]] = -short_weight
            if n_long:
                target[ranked.index[len(ranked) - n_long:]] = long_weight
            weights.iloc[pos] = target.to_numpy()
    return weights.ffill().fillna(0.0)


def weights_to_signals(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a weights panel into entry/exit events.

    A signal is emitted when a symbol's side changes (flat, long or short).
    Going straight from long to short emits LONG_EXIT then SHORT_ENTER on the
    same bar. Resizing a position without changing side emits nothing; use
    the weights themselves when exact sizes matter.

    Args:
        weights: Weights panel indexed by timestamp. NaN counts as flat.

    Returns:
        DataFrame with columns timestamp, symbol, signal_type, strength,
        sorted by timestamp with exits before entries. `strength` is the
        absolute weight being opened or closed.
    """
    now = weights.fillna(0.0)
    before = now.shift(1, fill_value=0.0)
    side_now, side_before = np.sign(now), np.sign(before)
    changed = side_now != side_before

    events = [
        ('LONG_EXIT', changed & (side_before > 0), before, 0),
        ('SHORT_EXIT', changed & (side_before < 0), before, 0),
        ('LONG_ENTER', changed & (side_now > 0), now, 1),
        ('SHORT_ENTER', changed & (side_now < 0), now, 1),
    ]
    parts = []
    for signal_type, mask, source, order in events:
        strength = source.abs().where(mask).stack()
        if strength.empty:
            continue
        strength.index.names = ['timestamp', 'symbol']
        part = strength.rename('strength').reset_index()
        part['signal_type'] = signal_type
        part['_order'] = order
        parts.append(part)

    if not parts:
        empty = pd.DataFrame(columns=list(SIGNAL_COLUMNS))
        return empty.astype({'strength': 'float64'})
    signals = pd.concat(parts, ignore_index=True)
    signals = signals.sort_values(['timestamp', '_order', 'symbol'], kind='stable')
    return signals[list(SIGNAL_COLUMNS)].reset_index(drop=True)
