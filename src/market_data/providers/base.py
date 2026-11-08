"""
Provider contract for historical OHLCV data.

Subclasses implement `_fetch_bars`; the base class adds symbol and interval
normalization, retry/circuit breaking and a short-lived response cache so
every provider behaves the same way.

Every provider uses the same date bounds: start inclusive, end exclusive.
Bars are dicts with keys timestamp (ISO 8601, UTC), open, high, low, close
and volume; src.market_data.schema.bars_to_frame turns them into a DataFrame.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from src.config.settings import DataSettings, get_settings
from src.market_data.resilience import ResilientExecutor
from src.market_data.schema import normalize_interval

Bar = Dict[str, Any]
ExceptionTypes = Tuple[Type[Exception], ...]


def to_utc(value: str) -> datetime:
    """Parse a date/time bound as a tz-aware UTC datetime (naive means UTC)."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
    return ts.to_pydatetime()


def to_epoch_ms(value: str) -> int:
    return int(to_utc(value).timestamp() * 1000)


def to_float(value: Any) -> float:
    """float(value), with None mapped to NaN (some APIs omit fields)."""
    return float('nan') if value is None else float(value)


class MarketDataProvider(ABC):
    """Common contract for historical OHLCV providers."""

    name: str = 'provider'

    def __init__(
        self,
        settings: Optional[DataSettings] = None,
        non_retryable: ExceptionTypes = (ValueError,),
        retryable: ExceptionTypes = (ConnectionError, TimeoutError),
    ):
        """
        Args:
            settings: Data settings; defaults to the global settings.
            non_retryable: Errors raised immediately (bad input, auth).
            retryable: Transient errors that are retried and counted by the
                circuit breaker. Pass the exception types your client library
                actually raises.
        """
        self.settings = settings if settings is not None else get_settings().data
        self._executor = ResilientExecutor(
            name=self.name,
            non_retryable=non_retryable,
            retryable_exceptions=retryable,
            settings=self.settings,
        )
        self._cache: Dict[tuple, Tuple[float, List[Bar]]] = {}
        self._cache_ttl = self.settings.cache_ttl_seconds

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.strip().upper()

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d',
    ) -> Dict[str, Any]:
        """
        Fetch OHLCV bars in [start_date, end_date).

        Args:
            symbol: Ticker or trading pair (e.g., 'AAPL', 'BTC/USDT').
            start_date: Inclusive start (date or datetime; naive means UTC).
            end_date: Exclusive end (date or datetime; naive means UTC).
            interval: '1m', '5m', '15m', '1h', '4h' or '1d' (Alpaca-style
                names such as '1Min' and '1D' are accepted).

        Returns:
            Dict with symbol (normalized), interval (canonical), data (list
            of bars), n_bars, source (provider name) and from_cache.

        Raises:
            CircuitBreakerOpen: The provider is being blocked after repeated
                failures (see src.market_data.resilience).
        """
        symbol = self.normalize_symbol(symbol)
        interval = normalize_interval(interval)
        key = (symbol, interval, start_date, end_date)

        cached = self._cache_get(key)
        if cached is not None:
            return self._result(symbol, interval, cached, from_cache=True)

        data = self._executor.call(self._fetch_bars, symbol, start_date, end_date, interval)
        self._cache_put(key, data)
        return self._result(symbol, interval, data, from_cache=False)

    @abstractmethod
    def _fetch_bars(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str,
    ) -> List[Bar]:
        """Fetch bars from the upstream API. Symbol and interval are normalized."""

    def _cache_get(self, key: tuple) -> Optional[List[Bar]]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, data = entry
        if time.monotonic() - stored_at >= self._cache_ttl:
            del self._cache[key]
            return None
        return list(data)

    def _cache_put(self, key: tuple, data: List[Bar]) -> None:
        if self._cache_ttl <= 0:
            return
        now = time.monotonic()
        # Evict expired entries so the cache cannot grow without bound.
        for old_key in [k for k, (t, _) in self._cache.items() if now - t >= self._cache_ttl]:
            del self._cache[old_key]
        self._cache[key] = (now, list(data))

    def _result(self, symbol: str, interval: str, data: List[Bar], from_cache: bool) -> Dict[str, Any]:
        return {
            'symbol': symbol,
            'interval': interval,
            'data': data,
            'n_bars': len(data),
            'source': self.name,
            'from_cache': from_cache,
        }
