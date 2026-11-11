"""
Canonical OHLCV schema shared by providers, the store, cleaning and features.

Every bar that enters the system goes through `bars_to_frame` or
`normalize_ohlcv`, so column names, dtypes and time zone rules live in one
place:

    timestamp            datetime64[ns, UTC]  bar open time, always UTC
    open/high/low/close  float64
    volume               float64              fractional for crypto

Interval names and the bars-per-year used for annualization live here too.
"""

from __future__ import annotations

import logging
import math
from datetime import timedelta
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ('timestamp', 'open', 'high', 'low', 'close', 'volume')
PRICE_COLUMNS = ('open', 'high', 'low', 'close')
NUMERIC_COLUMNS = PRICE_COLUMNS + ('volume',)

MARKETS = ('equity', 'crypto')

# Canonical intervals match the store ('1d'); aliases cover Alpaca-style names.
_INTERVAL_ALIASES = {
    '1m': '1m', '1min': '1m',
    '5m': '5m', '5min': '5m',
    '15m': '15m', '15min': '15m',
    '1h': '1h', '1hour': '1h',
    '4h': '4h', '4hour': '4h',
    '1d': '1d', '1day': '1d',
}

INTERVAL_TIMEDELTA = {
    '1m': timedelta(minutes=1),
    '5m': timedelta(minutes=5),
    '15m': timedelta(minutes=15),
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    '1d': timedelta(days=1),
}

_EQUITY_SESSION_MINUTES = 390  # 09:30-16:00 New York
_EQUITY_SESSIONS_PER_YEAR = 252
_NEW_YORK = 'America/New_York'


def normalize_interval(interval: str) -> str:
    """
    Map a provider-specific interval name to the canonical one.

    Args:
        interval: Bar size as a provider or user writes it, e.g. '1Day',
            '5min' or '1h'. Case and surrounding spaces are ignored.

    Returns:
        The canonical name used by the store and settings: one of
        '1m', '5m', '15m', '1h', '4h', '1d'.

    Raises:
        ValueError: If the interval is not supported.
    """
    try:
        return _INTERVAL_ALIASES[interval.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Supported: {sorted(set(_INTERVAL_ALIASES.values()))}"
        ) from None


def check_market(market: str) -> str:
    """
    Validate a market name.

    Args:
        market: 'equity' or 'crypto'.

    Returns:
        The same string, so it can be used inline: `m = check_market(m)`.

    Raises:
        ValueError: If `market` is not one of `MARKETS`.
    """
    if market not in MARKETS:
        raise ValueError(f"Unknown market '{market}'; use one of {MARKETS}")
    return market


def infer_market(symbol: str) -> str:
    """
    Guess the market from a symbol's format.

    Args:
        symbol: Ticker such as 'AAPL', or a pair such as 'BTC/USDT'.

    Returns:
        'crypto' if the symbol contains '/', otherwise 'equity'.
    """
    return 'crypto' if '/' in symbol else 'equity'


def periods_per_year(interval: str, market: str) -> float:
    """
    Conventional number of bars in a year, used to annualize volatility.

    Crypto trades 24/7/365, so every interval is exact. Equities use 252
    sessions a year, and intraday intervals count whole bars in one regular
    09:30-16:00 session: ceil(390 / minutes). A partial bar at the session
    edge is still a bar, so 1h gives 7 per session (not 6.5) and 4h gives
    2 (not 1.625).

    Intraday equity figures are still a convention, not a measurement:
    they assume regular hours only, a full session every day (no half
    days or halts) and no missing bars. Data that includes extended hours
    has more bars per day. When the exact figure matters, measure it with
    `estimate_periods_per_year` and pass that number on.

    Args:
        interval: Bar size in any form `normalize_interval` accepts,
            e.g. '1d', '1Day' or '5min'.
        market: 'equity' or 'crypto'.

    Returns:
        Bars per year, e.g. 252.0 for ('1d', 'equity'), 365.0 for
        ('1d', 'crypto'), 1764.0 for ('1h', 'equity') and 19656.0 for
        ('5m', 'equity'). Multiply a per-bar volatility by its square root
        to annualize it.

    Raises:
        ValueError: If the interval is not supported or the market is
            unknown.
    """
    interval = normalize_interval(interval)
    check_market(market)
    minutes = INTERVAL_TIMEDELTA[interval] / timedelta(minutes=1)
    if market == 'crypto':
        return 365 * 24 * 60 / minutes
    if interval == '1d':
        return float(_EQUITY_SESSIONS_PER_YEAR)
    bars_per_session = math.ceil(_EQUITY_SESSION_MINUTES / minutes)
    return float(_EQUITY_SESSIONS_PER_YEAR * bars_per_session)


def estimate_periods_per_year(timestamps: Iterable, market: str) -> float:
    """
    Measure bars per year from the bars you actually have.

    Counts the bars in each session (New York calendar date for equities,
    UTC date for crypto), takes the median, and scales it to a year. The
    median ignores half days, a partial first or last day, and the odd
    missing bar, but follows the data's real structure: regular-hours
    hourly bars give 7 * 252, bars that include extended hours give more.

    Args:
        timestamps: Bar timestamps for one symbol, e.g. `df['timestamp']`.
            Naive values are taken to be UTC.
        market: 'equity' (252 sessions a year) or 'crypto' (365 days).

    Returns:
        Median bars per session times sessions per year, e.g. 252.0 for
        daily equity bars or 1764.0 for regular-hours hourly bars.

    Raises:
        ValueError: If `timestamps` is empty or the market is unknown.
    """
    check_market(market)
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(timestamps)), utc=True)).dropna()
    if ts.empty:
        raise ValueError("Need at least one timestamp to estimate bars per year")
    if market == 'crypto':
        sessions, per_year = ts.normalize(), 365
    else:
        sessions, per_year = ts.tz_convert(_NEW_YORK).normalize(), _EQUITY_SESSIONS_PER_YEAR
    bars_per_session = pd.Series(1, index=sessions).groupby(level=0).size()
    return float(bars_per_session.median() * per_year)


_HAS_UTC_OFFSET = r'(?:Z|[+-]\d{2}:?\d{2})$'


def _parse_timestamps(ts: pd.Series) -> pd.Series:
    """
    Parse a timestamp column to datetime64[ns, UTC].

    Naive values are taken to be UTC. Values that already carry an offset
    are converted to UTC.

    Args:
        ts: Raw timestamps: ISO-8601 strings, datetimes, or integers read
            as epoch nanoseconds.

    Returns:
        The parsed timestamps on `ts.index`; missing values stay NaT.

    Raises:
        ValueError: If string values mix ones with and without a UTC
            offset, because pandas would silently apply the first offset
            to the naive ones.
    """
    if ts.dtype != object:
        return pd.to_datetime(ts, utc=True).dt.as_unit('ns')

    strings = ts.dropna()
    strings = strings[strings.map(lambda v: isinstance(v, str))]
    if len(strings):
        has_offset = strings.str.strip().str.contains(_HAS_UTC_OFFSET)
        if has_offset.any() and not has_offset.all():
            # pandas would apply the first value's offset to the naive ones.
            raise ValueError(
                "Timestamps mix values with and without a UTC offset; "
                "make the source consistent before loading."
            )
        parsed = pd.to_datetime(ts, utc=True, format='ISO8601')
    else:
        parsed = pd.to_datetime(ts, utc=True)
    return parsed.dt.as_unit('ns')


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `df` in the canonical schema.

    Timestamps become UTC (naive values are taken to be UTC), prices and
    volume become float64, rows without a timestamp are dropped, rows are
    sorted by time and duplicate bars keep the last occurrence. Extra
    columns such as 'symbol' are kept.

    Args:
        df: Bars with at least the columns in `OHLCV_COLUMNS`. Timestamps
            may be ISO-8601 strings or datetimes; integers are read as
            epoch nanoseconds, so convert epoch seconds first. Not modified.

    Returns:
        A new DataFrame with a fresh 0..n-1 index, sorted by
        (`symbol`, `timestamp`) when a `symbol` column exists, otherwise
        by `timestamp`.

    Raises:
        ValueError: If a required column is missing, or if string
            timestamps mix values with and without a UTC offset.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {missing}")

    out = df.copy()
    out['timestamp'] = _parse_timestamps(out['timestamp'])
    out[list(NUMERIC_COLUMNS)] = out[list(NUMERIC_COLUMNS)].astype('float64')

    no_ts = out['timestamp'].isna()
    if no_ts.any():
        logger.warning("Dropping %d bars without a timestamp", int(no_ts.sum()))
        out = out[~no_ts]

    key = ['symbol', 'timestamp'] if 'symbol' in out.columns else ['timestamp']
    out = out.sort_values(key, kind='stable')
    n_rows = len(out)
    out = out.drop_duplicates(key, keep='last')
    if len(out) < n_rows:
        logger.warning("Dropped %d duplicate bars", n_rows - len(out))

    return out.reset_index(drop=True)


def bars_to_frame(
    bars: Iterable[Mapping], symbol: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert provider bar dicts to a canonical DataFrame.

    Args:
        bars: One mapping per bar with keys 'timestamp', 'open', 'high',
            'low', 'close' and 'volume'. Any iterable works, including a
            generator; an empty one gives an empty frame.
        symbol: Ticker to add as a leading 'symbol' column, e.g. 'AAPL'.
            When None, no symbol column is added.

    Returns:
        Bars in the canonical schema (see `normalize_ohlcv`).

    Raises:
        ValueError: If the bars are missing a required key.
    """
    records = list(bars)
    if records:
        df = pd.DataFrame.from_records(records)
    else:
        df = pd.DataFrame(columns=list(OHLCV_COLUMNS))
    if symbol is not None:
        df.insert(0, 'symbol', symbol)
    return normalize_ohlcv(df)


def ohlc_valid_mask(df: pd.DataFrame) -> pd.Series:
    """
    Flag which bars are internally consistent.

    A bar is valid when every price and the volume are finite numbers, the
    prices are positive, open and close lie inside [low, high], and volume
    is non-negative. NaN and +/-inf fail.

    The finite check matters because +inf passes every comparison below
    (inf >= x is True), so a bar of all +inf would otherwise look valid.

    Args:
        df: Bars with the price and volume columns in `NUMERIC_COLUMNS`.

    Returns:
        Boolean Series aligned with `df.index`: True for a valid bar.
    """
    o, h, l, c, v = (df[col] for col in NUMERIC_COLUMNS)
    finite = np.isfinite(df[list(NUMERIC_COLUMNS)].astype('float64')).all(axis=1)
    return (
        finite
        & (l > 0)
        & (h >= l)
        & (h >= o) & (h >= c)
        & (l <= o) & (l <= c)
        & (v >= 0)
    )


def require_single_symbol(df: pd.DataFrame, what: str) -> None:
    """
    Refuse a frame that holds more than one symbol.

    Rolling windows must not run from one symbol into the next, so cleaning
    and feature functions call this first.

    Args:
        df: Bars to check. A frame without a 'symbol' column passes.
        what: Name of the calling step, used in the error message,
            e.g. 'clean_ohlcv'.

    Raises:
        ValueError: If `df` contains two or more distinct symbols.
    """
    if 'symbol' in df.columns and df['symbol'].nunique(dropna=True) > 1:
        raise ValueError(
            f"{what} works on one symbol at a time; got "
            f"{sorted(df['symbol'].dropna().unique())[:5]}. "
            "Split the frame with df.groupby('symbol')."
        )
