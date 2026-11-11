"""
Alpaca equities market data (REST) via the official alpaca-py SDK.

Credentials come from DataSettings (ALPACA_API_KEY, ALPACA_API_SECRET).
Account and position queries live in src.execution.alpaca_broker.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

from src.config.settings import DataSettings
from src.market_data.providers.base import Bar, MarketDataProvider, to_utc

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient


def alpaca_credentials(settings: DataSettings) -> Tuple[str, str]:
    """
    Read the Alpaca API key and secret from settings.

    Args:
        settings: Data settings holding ALPACA_API_KEY / ALPACA_API_SECRET.

    Returns:
        (key, secret) as plain strings.

    Raises:
        ValueError: If either value is missing or empty.
    """
    key = settings.alpaca_api_key.get_secret_value() if settings.alpaca_api_key else ''
    secret = settings.alpaca_api_secret.get_secret_value() if settings.alpaca_api_secret else ''
    if not key or not secret:
        raise ValueError(
            "Alpaca API credentials required. Set ALPACA_API_KEY and "
            "ALPACA_API_SECRET in the environment or a .env file."
        )
    return key, secret


def alpaca_retryable_errors() -> Tuple[type, ...]:
    """
    List the transient errors alpaca-py can raise, for the retry policy.

    alpaca-py uses `requests` under the hood, so its network errors are
    `requests` exceptions rather than Python's built-in ones.

    Returns:
        Exception types that should be retried and counted by the
        circuit breaker.
    """
    import requests

    return (
        ConnectionError,
        TimeoutError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )


def _alpaca_timeframe(interval: str):
    """
    Translate a canonical interval into alpaca-py's `TimeFrame`.

    Args:
        interval: Canonical interval, e.g. '1d' or '5m'.

    Returns:
        The matching `alpaca.data.timeframe.TimeFrame`.

    Raises:
        KeyError: If the interval is not canonical (callers normalize first).
    """
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    frames = {
        '1m': TimeFrame(1, TimeFrameUnit.Minute),
        '5m': TimeFrame(5, TimeFrameUnit.Minute),
        '15m': TimeFrame(15, TimeFrameUnit.Minute),
        '1h': TimeFrame(1, TimeFrameUnit.Hour),
        '4h': TimeFrame(4, TimeFrameUnit.Hour),
        '1d': TimeFrame(1, TimeFrameUnit.Day),
    }
    return frames[interval]


class AlpacaDataClient(MarketDataProvider):
    """
    Historical equity bars from Alpaca, split-adjusted.

    Daily bars are stamped at midnight New York time (Alpaca's convention),
    which src.market_data.cleaning.gaps relies on.
    """

    name = 'alpaca'

    def __init__(
        self,
        settings: Optional[DataSettings] = None,
        data_client: Optional['StockHistoricalDataClient'] = None,
    ):
        """
        Check credentials now; the SDK client is created lazily.

        Args:
            settings: Data settings; defaults to the global settings.
            data_client: Pre-built StockHistoricalDataClient (or a test
                double). Created from the credentials on first use if None.

        Raises:
            ValueError: If the Alpaca credentials are missing.
        """
        super().__init__(settings, retryable=alpaca_retryable_errors())
        self.api_key, self.api_secret = alpaca_credentials(self.settings)
        self._data_client = data_client

    def _get_data_client(self):
        """
        Return the SDK client, creating it on first use.

        Returns:
            The injected client, or a new `StockHistoricalDataClient` built
            from the stored credentials.
        """
        if self._data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        return self._data_client

    def _fetch_bars(self, symbol, start_date, end_date, interval) -> List[Bar]:
        """
        Request split-adjusted bars from Alpaca.

        Alpaca treats `end` as inclusive, so bars at or after `end_date`
        are dropped here to keep the end exclusive like every provider.

        Args:
            symbol: Upper-case ticker, e.g. 'AAPL'.
            start_date: Inclusive start, or None for Alpaca's default.
            end_date: Exclusive end, or None for "up to now".
            interval: Canonical interval, e.g. '1d'.

        Returns:
            Bars as dicts (see `MarketDataProvider._fetch_bars`); empty if
            Alpaca has no data for the symbol in that range.
        """
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest

        end = to_utc(end_date) if end_date else None
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_alpaca_timeframe(interval),
            start=to_utc(start_date) if start_date else None,
            end=end,
            adjustment=Adjustment.SPLIT,
        )
        bars = self._get_data_client().get_stock_bars(request)

        # Alpaca's `end` is inclusive; trim to keep end exclusive like every provider.
        return [
            {
                'timestamp': bar.timestamp.isoformat(),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume),
            }
            for bar in bars.data.get(symbol, [])
            if bar.timestamp is not None and (end is None or bar.timestamp < end)
        ]
