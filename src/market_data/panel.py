"""
Wide price panels for cross-sectional work.

Cleaning and features take one symbol at a time (long format). Strategies
that compare symbols, such as momentum ranking or pairs trading, need one
column per symbol instead:

                               AAPL    MSFT    NVDA
    timestamp
    2024-01-02 05:00:00+00:00  185.6   370.9   48.2
    2024-01-03 05:00:00+00:00  184.3   370.6   47.6

Timestamps where a symbol has no bar are NaN; nothing is filled here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd

from src.market_data.schema import NUMERIC_COLUMNS

if TYPE_CHECKING:
    from src.market_data.store import FinancialDataStore


def to_panel(data: pd.DataFrame, field: str = 'close') -> pd.DataFrame:
    """
    Pivot a long multi-symbol frame into one column per symbol.

    Args:
        data: Bars with 'timestamp', 'symbol' and `field` columns, e.g. several
            symbols' cleaned OHLCV concatenated together.
        field: Column to put in the panel ('close', 'volume', ...).

    Returns:
        DataFrame indexed by UTC timestamp (sorted), one float column per
        symbol (sorted by name). Missing bars are NaN.

    Raises:
        ValueError: If a required column is missing or a symbol has two bars
            with the same timestamp.
    """
    missing = [c for c in ('timestamp', 'symbol', field) if c not in data.columns]
    if missing:
        raise ValueError(f"to_panel needs columns {missing}")
    if data.duplicated(['timestamp', 'symbol']).any():
        raise ValueError(
            "Duplicate (timestamp, symbol) rows; run the data through "
            "normalize_ohlcv or the cleaning pipeline first."
        )
    panel = data.pivot(index='timestamp', columns='symbol', values=field)
    panel = panel.sort_index().sort_index(axis=1).astype('float64')
    panel.index = pd.DatetimeIndex(panel.index, name='timestamp')
    panel.columns.name = None
    return panel


def load_panel(
    store: 'FinancialDataStore',
    symbols: Iterable[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d',
    field: str = 'close',
) -> pd.DataFrame:
    """
    Read several symbols from the store and return them as a panel.

    Args:
        store: Open data store.
        symbols: Symbols to load. Symbols with no stored bars are left out
            (check the returned columns).
        start_date: Inclusive bound (naive means UTC).
        end_date: Exclusive bound (naive means UTC).
        interval: Bar interval.
        field: OHLCV column to use.

    Returns:
        Panel as described in `to_panel`.

    Raises:
        ValueError: If `field` is not an OHLCV column or no symbol has data.
    """
    if field not in NUMERIC_COLUMNS:
        raise ValueError(f"field must be one of {NUMERIC_COLUMNS}, got '{field}'")
    frames = [
        store.query_ohlcv(symbol, start_date, end_date, interval)
        for symbol in symbols
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No stored bars for any of the requested symbols")
    return to_panel(pd.concat(frames, ignore_index=True), field)
