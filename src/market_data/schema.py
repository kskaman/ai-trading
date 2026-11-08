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
from datetime import timedelta
from typing import Iterable, Mapping, Optional

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


def normalize_interval(interval: str) -> str:
    """Map provider-specific interval names to the canonical form ('1d', '5m', ...)."""
    try:
        return _INTERVAL_ALIASES[interval.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Supported: {sorted(set(_INTERVAL_ALIASES.values()))}"
        ) from None


def check_market(market: str) -> str:
    if market not in MARKETS:
        raise ValueError(f"Unknown market '{market}'; use one of {MARKETS}")
    return market


def infer_market(symbol: str) -> str:
    """'crypto' for pair symbols such as 'BTC/USDT', otherwise 'equity'."""
    return 'crypto' if '/' in symbol else 'equity'


def periods_per_year(interval: str, market: str) -> float:
    """
    Number of bars in a year, used to annualize volatility.

    Crypto trades 24/7/365. Equities use 252 sessions of 6.5 regular hours;
    intraday equity figures are approximate because they ignore half days
    and extended-hours bars.
    """
    interval = normalize_interval(interval)
    check_market(market)
    minutes = INTERVAL_TIMEDELTA[interval] / timedelta(minutes=1)
    if market == 'crypto':
        return 365 * 24 * 60 / minutes
    if interval == '1d':
        return float(_EQUITY_SESSIONS_PER_YEAR)
    return _EQUITY_SESSIONS_PER_YEAR * _EQUITY_SESSION_MINUTES / minutes


_HAS_UTC_OFFSET = r'(?:Z|[+-]\d{2}:?\d{2})$'


def _parse_timestamps(ts: pd.Series) -> pd.Series:
    """Parse to datetime64[ns, UTC]; naive values are UTC."""
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
    """Convert provider bar dicts to a canonical DataFrame (see module docstring)."""
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
    True where a bar is internally consistent: positive prices, open and
    close inside [low, high], and non-negative volume. NaN fails every check.
    """
    o, h, l, c, v = (df[col] for col in NUMERIC_COLUMNS)
    return (
        (l > 0)
        & (h >= l)
        & (h >= o) & (h >= c)
        & (l <= o) & (l <= c)
        & (v >= 0)
    )


def require_single_symbol(df: pd.DataFrame, what: str) -> None:
    """Rolling windows must not run across symbols; refuse mixed frames."""
    if 'symbol' in df.columns and df['symbol'].nunique(dropna=True) > 1:
        raise ValueError(
            f"{what} works on one symbol at a time; got "
            f"{sorted(df['symbol'].dropna().unique())[:5]}. "
            "Split the frame with df.groupby('symbol')."
        )
