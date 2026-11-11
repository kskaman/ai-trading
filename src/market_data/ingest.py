"""
Market data ingestion orchestrator.

    route -> fetch (with failover) -> store raw -> [fill gaps] -> [clean] -> DataFrame

Where things live:
    src.market_data.providers   historical bar providers (Alpaca, CCXT)
    src.market_data.streaming   real-time WebSocket feed
    src.market_data.resilience  retry + circuit breaker used by every provider
    src.market_data.schema      the canonical OHLCV DataFrame
    src.market_data.cleaning    quality flags and gap filling
    src.execution        account, balance and order placement (not data)

Every tunable value (retries, delays, cache TTL, URLs, credentials, source
routing) comes from DataSettings in src.config.settings, i.e. environment
variables or a .env file.

Run the demo from the project root with: python -m src.market_data.ingest
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import pandas as pd

from src.config.settings import DataSettings, get_settings
from src.market_data.cleaning import clean_ohlcv, forward_fill_gaps
from src.market_data.providers import AlpacaDataClient, CryptoExchangeClient, MarketDataProvider
from src.market_data.schema import bars_to_frame, infer_market

if TYPE_CHECKING:
    from src.market_data.store import FinancialDataStore

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Entry point for acquiring market data.

    Routes each symbol to a provider (pair symbols such as 'BTC/USDT' go to
    the crypto source, everything else to the primary source), fails over
    from the primary to the backup source, then optionally persists the raw
    bars, fills gaps and adds cleaning flags. Feature engineering stays
    downstream (src.features) so ingestion knows nothing about models.
    """

    def __init__(
        self,
        providers: Dict[str, MarketDataProvider],
        store: Optional['FinancialDataStore'] = None,
        settings: Optional[DataSettings] = None,
    ):
        """
        Wire up providers and an optional store.

        Args:
            providers: Provider instances keyed by source name.
            store: Where raw bars are persisted; None disables persistence.
            settings: Routing configuration; defaults to the global settings.
        """
        self.providers = providers
        self.store = store
        self.settings = settings if settings is not None else get_settings().data

    @classmethod
    def from_settings(
        cls,
        settings: Optional[DataSettings] = None,
        store: Optional['FinancialDataStore'] = None,
    ) -> 'DataIngestionService':
        """
        Build the service with every provider the settings can support.

        A provider whose credentials, SDK or exchange configuration is missing
        or invalid is skipped with a warning rather than failing startup.

        Args:
            settings: Data settings with credentials and routing; defaults to
                the global settings.
            store: Where raw bars are persisted; None disables persistence.

        Returns:
            A service with 'alpaca' and/or 'ccxt' registered (possibly
            neither; `fetch` then fails with RuntimeError).
        """
        settings = settings if settings is not None else get_settings().data

        providers: Dict[str, MarketDataProvider] = {}
        if settings.alpaca_api_key and settings.alpaca_api_secret:
            try:
                providers['alpaca'] = AlpacaDataClient(settings)
            except (ImportError, ValueError) as exc:
                logger.warning("Alpaca provider disabled: %s", exc)
        else:
            logger.warning("Alpaca provider disabled: ALPACA_API_KEY/ALPACA_API_SECRET not set.")

        try:
            providers['ccxt'] = CryptoExchangeClient(settings)
        except (ImportError, ValueError) as exc:
            logger.warning("CCXT provider disabled: %s", exc)

        return cls(providers, store=store, settings=settings)

    def _candidates(self, symbol: str, source: Optional[str]) -> List[str]:
        """
        Choose which sources to try for a symbol, in order.

        Args:
            symbol: Ticker or pair; pairs ('BTC/USDT') go to the crypto
                source only, because the backup is an equities source.
            source: A pinned source name, or None to route automatically.

        Returns:
            Source names to try: just `source` when pinned, the crypto
            source for pairs, otherwise the primary then the backup (if set
            and different).
        """
        if source is not None:
            return [source]
        if infer_market(symbol) == 'crypto':
            # The backup source is an equities source; it cannot serve pairs.
            return [self.settings.crypto_source]

        candidates = [self.settings.primary_source]
        backup = self.settings.backup_source
        if backup and backup != candidates[0]:
            candidates.append(backup)
        return candidates

    def fetch(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d',
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch bars in [start_date, end_date) from the routed provider.

        Any provider error, including an open circuit breaker, moves on to the
        backup source. Passing `source` pins the provider and disables failover.

        Args:
            symbol: Ticker or pair, e.g. 'AAPL' or 'BTC/USDT'.
            start_date: Inclusive start (naive means UTC), or None.
            end_date: Exclusive end (naive means UTC), or None for now.
            interval: Bar size, e.g. '1d' or '5m'.
            source: Provider name to use exclusively, e.g. 'alpaca'.

        Returns:
            The provider's response dict (see
            `MarketDataProvider.fetch_ohlcv`): symbol, interval, data,
            n_bars, source and from_cache.

        Raises:
            RuntimeError: If every candidate source failed or none is
                registered; the last provider error is chained as the cause.
        """
        candidates = self._candidates(symbol, source)

        last_error: Optional[Exception] = None
        for name in candidates:
            provider = self.providers.get(name)
            if provider is None:
                logger.warning("Data source '%s' is not registered; skipping.", name)
                continue
            try:
                return provider.fetch_ohlcv(symbol, start_date, end_date, interval)
            except Exception as exc:
                last_error = exc
                logger.warning("%s failed for %s: %s", name, symbol, exc)

        raise RuntimeError(
            f"All data sources failed for {symbol} (tried {candidates})"
        ) from last_error

    def ingest(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d',
        source: Optional[str] = None,
        persist: bool = True,
        fill_gaps: bool = False,
        clean: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch bars, store them raw, then optionally fill gaps and clean.

        Raw bars are persisted before any processing so the raw zone holds
        exactly what the provider returned (in the canonical schema).

        Args:
            symbol: Ticker or pair, e.g. 'AAPL' or 'BTC/USDT'.
            start_date: Inclusive start (naive means UTC), or None.
            end_date: Exclusive end (naive means UTC), or None for now.
            interval: Bar size, e.g. '1d' or '5m'.
            source: Provider name to use exclusively; disables failover.
            persist: Save the raw bars to the store (ignored if the service
                has no store).
            fill_gaps: Insert bars missing from the trading calendar
                (src.market_data.cleaning.forward_fill_gaps).
            clean: Add is_outlier / is_stale / ohlc_valid flags
                (src.market_data.cleaning.clean_ohlcv).

        Returns:
            Canonical OHLCV DataFrame with a 'symbol' column, sorted by
            timestamp (empty if the source had no bars).

        Raises:
            RuntimeError: If every data source failed (see `fetch`).
        """
        result = self.fetch(symbol, start_date, end_date, interval, source)
        symbol, interval = result['symbol'], result['interval']
        df = bars_to_frame(result['data'], symbol=symbol)

        if df.empty:
            logger.warning("No bars returned for %s (%s).", symbol, interval)
            return df

        if persist and self.store is not None:
            self.store.ingest_ohlcv(symbol, interval, df)

        if fill_gaps:
            df = forward_fill_gaps(df, interval=interval, market=infer_market(symbol))
        if clean:
            df = clean_ohlcv(df)
        return df

    def ingest_many(
        self, symbols: Sequence[str], **kwargs: Any
    ) -> Dict[str, pd.DataFrame]:
        """
        Ingest several symbols, carrying on past failures.

        Args:
            symbols: Tickers and/or pairs, e.g. ['AAPL', 'BTC/USDT'].
            **kwargs: Passed to `ingest` for every symbol (start_date,
                interval, clean, ...).

        Returns:
            DataFrames keyed by symbol. A symbol that failed is logged and
            left out, so check the keys if you need all of them.
        """
        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.ingest(symbol, **kwargs)
            except Exception:
                logger.exception("Ingestion failed for %s", symbol)
        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    service = DataIngestionService.from_settings()
    frames = service.ingest_many(
        ['AAPL', 'BTC/USDT'], start_date='2024-01-01', end_date='2024-07-01',
        interval='1d', persist=False, fill_gaps=True, clean=True,
    )
    for sym, frame in frames.items():
        print(f"{sym}: {len(frame)} bars")
