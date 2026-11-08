"""
Crypto market data via CCXT (100+ exchanges, one API).

CCXT handles request signing, rate limiting (enableRateLimit) and data
normalization. OHLCV, order books and symbol lists are public endpoints, so
this client needs no API keys. Balances and orders live in
src.execution.ccxt_broker.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.config.settings import DataSettings
from src.market_data.providers.base import Bar, MarketDataProvider, to_epoch_ms, to_float
from src.market_data.schema import INTERVAL_TIMEDELTA

if TYPE_CHECKING:
    import ccxt

logger = logging.getLogger(__name__)

# Friendly names -> CCXT exchange ids. Any other CCXT id is accepted as-is.
EXCHANGE_MAP = {
    'binance': 'binance',
    'coinbase': 'coinbase',  # Coinbase Advanced Trade ('coinbasepro' was removed from CCXT)
    'kraken': 'kraken',
    'bybit': 'bybit',
    'okx': 'okx',
    'kucoin': 'kucoin',
}


def normalize_ccxt_symbol(symbol: str) -> str:
    """'btc-usdt' -> 'BTC/USDT'."""
    return symbol.strip().upper().replace('-', '/')


def ccxt_error_policy() -> Tuple[Tuple[type, ...], Tuple[type, ...]]:
    """(retryable, non_retryable) exception types for CCXT calls."""
    import ccxt

    retryable = (ccxt.NetworkError,)  # timeouts, rate limits, exchange down
    non_retryable = (ValueError, ccxt.AuthenticationError, ccxt.BadRequest)
    return retryable, non_retryable


def create_exchange(
    exchange_id: str,
    api_key: str = '',
    api_secret: str = '',
    sandbox: bool = False,
    require_sandbox: bool = False,
) -> 'ccxt.Exchange':
    """
    Build a CCXT exchange instance (no network calls happen here).

    Args:
        exchange_id: Friendly name from EXCHANGE_MAP or any CCXT id.
        api_key: Only needed for private endpoints (balances, orders).
        api_secret: Only needed for private endpoints (balances, orders).
        sandbox: Use the exchange testnet.
        require_sandbox: Raise instead of falling back to production when the
            exchange has no testnet (used for order placement).

    Raises:
        ValueError: Unknown exchange, or no testnet while `require_sandbox`.
    """
    import ccxt

    ccxt_id = EXCHANGE_MAP.get(exchange_id, exchange_id)
    if ccxt_id not in ccxt.exchanges:
        raise ValueError(
            f"Exchange '{exchange_id}' not found in CCXT. "
            f"Common choices: {sorted(EXCHANGE_MAP)}"
        )

    config: Dict[str, Any] = {
        'enableRateLimit': True,  # Critical: prevents API bans
        'timeout': 30000,
    }
    if api_key and api_secret:
        config['apiKey'] = api_key
        config['secret'] = api_secret

    exchange = getattr(ccxt, ccxt_id)(config)

    if sandbox:
        try:
            exchange.set_sandbox_mode(True)
        except ccxt.NotSupported:
            if require_sandbox:
                raise ValueError(
                    f"{exchange_id} has no testnet; refusing to use production "
                    "endpoints while paper trading is enabled."
                ) from None
            logger.warning("%s has no sandbox; using production endpoints.", exchange_id)

    return exchange


class CryptoExchangeClient(MarketDataProvider):
    """Historical candles, order books and symbol lists from one CCXT exchange."""

    name = 'ccxt'

    def __init__(
        self,
        settings: Optional[DataSettings] = None,
        exchange: Optional['ccxt.Exchange'] = None,
    ):
        """
        Args:
            settings: Data settings; defaults to the global settings.
            exchange: Pre-built CCXT exchange (or a test double). Built from
                settings (ccxt_exchange_id, ccxt_sandbox) if None.
        """
        retryable, non_retryable = ccxt_error_policy()
        super().__init__(settings, non_retryable=non_retryable, retryable=retryable)
        s = self.settings
        self.exchange_id = s.ccxt_exchange_id
        self.default_limit = s.ccxt_default_limit
        self.page_limit = s.ccxt_page_limit
        self.exchange = exchange if exchange is not None else create_exchange(
            s.ccxt_exchange_id, sandbox=s.ccxt_sandbox
        )
        self._markets: Optional[Dict[str, Any]] = None

    def normalize_symbol(self, symbol: str) -> str:
        return normalize_ccxt_symbol(symbol)

    def _ensure_markets(self) -> Dict[str, Any]:
        if self._markets is None:
            logger.info("Loading markets for %s...", self.exchange_id)
            self._markets = self._executor.call(self.exchange.load_markets)
        return self._markets

    def _fetch_bars(self, symbol, start_date, end_date, interval) -> List[Bar]:
        if symbol not in self._ensure_markets():
            raise ValueError(
                f"Symbol '{symbol}' not found on {self.exchange_id}. "
                f"Run list_symbols() to see available pairs."
            )

        step_ms = int(INTERVAL_TIMEDELTA[interval].total_seconds() * 1000)
        until = to_epoch_ms(end_date) if end_date else None

        if start_date:
            raw = self._fetch_range(symbol, interval, to_epoch_ms(start_date), until, step_ms)
        elif until is not None:
            # No start: the `default_limit` candles before the end date.
            since = until - self.default_limit * step_ms
            raw = self._fetch_range(symbol, interval, since, until, step_ms)
        else:
            raw = self.exchange.fetch_ohlcv(symbol, interval, limit=self.default_limit)

        # CCXT format: [timestamp_ms, open, high, low, close, volume]
        return [
            {
                'timestamp': self.exchange.iso8601(bar[0]),
                'open': to_float(bar[1]),
                'high': to_float(bar[2]),
                'low': to_float(bar[3]),
                'close': to_float(bar[4]),
                'volume': to_float(bar[5]),
            }
            for bar in raw
            if until is None or bar[0] < until
        ]

    def _fetch_range(
        self, symbol: str, interval: str, since: int, until: Optional[int], step_ms: int
    ) -> List[list]:
        """
        Page forward from `since` to `until` (or the latest candle).

        Stops on an empty page, never on a short one: exchanges cap page
        sizes below ccxt_page_limit (Kraken 720, Coinbase/OKX 300), so a
        short page does not mean the range is exhausted.
        """
        raw: List[list] = []
        while True:
            page = self.exchange.fetch_ohlcv(symbol, interval, since=since, limit=self.page_limit)
            page = [bar for bar in page if bar[0] >= since]  # guards against no progress
            if not page:
                break
            raw.extend(page)
            next_since = page[-1][0] + step_ms
            if until is not None and next_since >= until:
                break
            since = next_since
        return raw

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch the current order book (`limit` price levels per side)."""
        self._ensure_markets()
        order_book = self._executor.call(
            self.exchange.fetch_order_book, normalize_ccxt_symbol(symbol), limit=limit
        )

        return {
            'symbol': order_book['symbol'],
            'bids': [[float(p), float(q)] for p, q, *_ in order_book['bids']],
            'asks': [[float(p), float(q)] for p, q, *_ in order_book['asks']],
            'timestamp': self.exchange.iso8601(order_book['timestamp']),
        }

    def list_symbols(self) -> List[str]:
        """List all available trading symbols on the exchange."""
        return sorted(self._ensure_markets().keys())
