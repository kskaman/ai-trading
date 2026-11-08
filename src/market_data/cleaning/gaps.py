"""
Gap handling: insert bars missing from the expected trading timeline.

The timeline depends on the market:

    crypto   trades 24/7, so every interval step is expected.
    equity   daily bars follow NYSE sessions (exchange_calendars), so
             weekends and market holidays are never treated as gaps. Daily
             equity bars are stamped at midnight New York time, which is
             Alpaca's convention. Intraday equity timelines need a decision
             about extended hours, so pass `expected` explicitly for those.
"""

import logging
from typing import Optional

import pandas as pd

from src.market_data.schema import (
    INTERVAL_TIMEDELTA, PRICE_COLUMNS, check_market, normalize_interval,
    normalize_ohlcv, require_single_symbol,
)

logger = logging.getLogger(__name__)

_NEW_YORK = 'America/New_York'


def _as_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def _new_york_date(value) -> pd.Timestamp:
    """Session date for a bound: naive values are already New York dates."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_NEW_YORK).tz_localize(None)
    return ts.normalize()


def expected_timestamps(start, end, interval: str, market: str) -> pd.DatetimeIndex:
    """
    Bar timestamps a complete series would contain between start and end
    (both inclusive), in UTC.

    Naive bounds mean UTC for crypto, and New York calendar dates for daily
    equities (so '2024-01-31' includes the January 31 session).
    """
    interval = normalize_interval(interval)
    check_market(market)

    if market == 'crypto':
        return pd.date_range(_as_utc(start), _as_utc(end), freq=INTERVAL_TIMEDELTA[interval])

    if interval != '1d':
        raise NotImplementedError(
            "Intraday equity gap detection needs session minutes and a choice "
            "about extended-hours bars; pass `expected` to forward_fill_gaps."
        )

    import exchange_calendars as xcals

    first, last = _new_york_date(start), _new_york_date(end)
    # Pad the calendar bounds: exchange_calendars rejects a range whose start
    # or end falls outside the sessions the calendar was built with.
    padding = pd.Timedelta(days=14)
    calendar = xcals.get_calendar(
        'XNYS', start=first - padding, end=max(last, first) + padding
    )
    sessions = calendar.sessions_in_range(first, last)
    return pd.DatetimeIndex(sessions).tz_localize(_NEW_YORK).tz_convert('UTC')


def forward_fill_gaps(
    df: pd.DataFrame,
    interval: str = '1d',
    market: str = 'equity',
    max_gap_bars: int = 5,
    expected: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Insert bars missing from the expected timeline.

    Gaps of up to `max_gap_bars` consecutive bars are filled with flat bars:
    open = high = low = close = previous close and volume 0, because no
    trading happened. Longer gaps are inserted as NaN rows so they stay
    visible and unfilled. Bars present in `df` are never dropped, even if
    they are not on the expected timeline.

    Run this before clean_ohlcv: filled bars then show up as stale.

    Args:
        df: OHLCV frame for a single symbol.
        interval: Bar interval, used to build the timeline.
        market: 'equity' or 'crypto', used to build the timeline.
        max_gap_bars: Longest gap (in bars) that is filled.
        expected: Explicit timeline to use instead of building one.

    Returns:
        Frame on the expected timeline with a 'was_forward_filled' column.
    """
    require_single_symbol(df, 'forward_fill_gaps')
    df = normalize_ohlcv(df)
    if df.empty:
        df['was_forward_filled'] = pd.Series(dtype=bool)
        return df

    original = pd.DatetimeIndex(df['timestamp'])
    if expected is None:
        expected = expected_timestamps(original[0], original[-1], interval, market)
    expected = pd.DatetimeIndex(expected)
    expected = expected.tz_localize('UTC') if expected.tz is None else expected.tz_convert('UTC')

    off_timeline = ~original.isin(expected)
    if off_timeline.any():
        logger.warning(
            "%d of %d bars are not on the expected %s %s timeline; kept as-is",
            int(off_timeline.sum()), len(original), market, interval,
        )

    in_span = expected[(expected >= original[0]) & (expected <= original[-1])]
    out = df.set_index('timestamp').reindex(in_span.union(original))
    out.index.name = 'timestamp'

    inserted = pd.Series(~out.index.isin(original), index=out.index)
    run_id = (inserted != inserted.shift()).cumsum()
    run_len = inserted.groupby(run_id).transform('sum')
    fill = inserted & (run_len <= max_gap_bars)

    previous_close = out['close'].ffill()
    for col in PRICE_COLUMNS:
        out.loc[fill, col] = previous_close[fill]
    out.loc[fill, 'volume'] = 0.0
    for label in ('symbol', 'interval'):  # identifiers carry over to inserted rows
        if label in out.columns:
            out[label] = out[label].ffill()
    out['was_forward_filled'] = fill

    gap_sizes = run_len[inserted].groupby(run_id[inserted]).first()
    logger.info(
        "Gap analysis: %d gaps (largest %d bars); %d bars forward-filled, %d left empty",
        len(gap_sizes), int(gap_sizes.max()) if len(gap_sizes) else 0,
        int(fill.sum()), int((inserted & ~fill).sum()),
    )
    return out.reset_index()
