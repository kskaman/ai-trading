"""
Alpaca account queries (orders to come).

Credentials and the paper/live switch come from ExecutionSettings, which
falls back to ALPACA_API_KEY / ALPACA_API_SECRET / ALPACA_PAPER.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.config.settings import DataSettings, ExecutionSettings, get_settings
from src.market_data.providers.alpaca_client import alpaca_retryable_errors
from src.market_data.resilience import ResilientExecutor

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient


class AlpacaBroker:
    """Account information and open positions from Alpaca's trading API."""

    name = 'alpaca'

    def __init__(
        self,
        settings: Optional[ExecutionSettings] = None,
        retry_settings: Optional[DataSettings] = None,
        trading_client: Optional['TradingClient'] = None,
    ):
        """
        Check credentials now; the SDK client is created lazily.

        Args:
            settings: Execution settings with credentials and the paper/live
                switch; defaults to the global settings.
            retry_settings: Retry/circuit-breaker policy (DATA_* settings);
                defaults to the global settings.
            trading_client: Pre-built alpaca TradingClient (or a test double).

        Raises:
            ValueError: If the Alpaca credentials are missing.
        """
        settings = settings if settings is not None else get_settings().execution
        key = settings.api_key.get_secret_value() if settings.api_key else ''
        secret = settings.api_secret.get_secret_value() if settings.api_secret else ''
        if not key or not secret:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET (or EXEC_API_KEY / EXEC_API_SECRET)."
            )
        self._api_key, self._api_secret = key, secret
        self.paper = settings.paper_trading
        self._client = trading_client
        self._executor = ResilientExecutor(
            name='alpaca-trading',
            retryable_exceptions=alpaca_retryable_errors(),
            settings=retry_settings,
        )

    def _get_client(self):
        """
        Return the SDK client, creating it on first use.

        Returns:
            The injected client, or a new `TradingClient` for the paper or
            live account depending on settings.
        """
        if self._client is None:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(self._api_key, self._api_secret, paper=self.paper)
        return self._client

    def get_account_info(self) -> Dict[str, Any]:
        """
        Retrieve account information (retried on network errors).

        Returns:
            Dict with total_equity, cash, buying_power and last_equity (all
            floats in account currency; last_equity is the previous close)
            and daytrade_count (an int).

        Raises:
            CircuitBreakerOpen: If Alpaca has been failing repeatedly.
        """
        def _fetch():
            """
            Make one account request; called through the retry executor.

            Returns:
                The account dict described above.
            """
            account = self._get_client().get_account()
            return {
                'total_equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'daytrade_count': int(account.daytrade_count),
                'last_equity': float(account.last_equity),
            }

        return self._executor.call(_fetch)

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Retrieve current open positions (retried on network errors).

        Returns:
            One dict per position with symbol, qty (negative for shorts),
            market_value, unrealized_pl and avg_entry_price. Empty if flat.

        Raises:
            CircuitBreakerOpen: If Alpaca has been failing repeatedly.
        """
        def _fetch():
            """
            Make one positions request; called through the retry executor.

            Returns:
                The list of position dicts described above.
            """
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'avg_entry_price': float(pos.avg_entry_price),
                }
                for pos in self._get_client().get_all_positions()
            ]

        return self._executor.call(_fetch)
