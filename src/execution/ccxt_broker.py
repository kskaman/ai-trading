"""
Balances and order placement on a CCXT exchange.

Exchange id and API keys come from DataSettings (CCXT_API_KEY /
CCXT_API_SECRET). While ExecutionSettings.ccxt_testnet is on (EXEC_CCXT_TESTNET,
the default) orders go to the exchange testnet; if the exchange has none,
construction fails rather than silently sending orders to production. This
switch is independent of Alpaca's paper/live setting.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from src.config.settings import DataSettings, ExecutionSettings, get_settings
from src.market_data.providers.ccxt_client import (
    ccxt_error_policy, create_exchange, normalize_ccxt_symbol,
)
from src.market_data.resilience import ResilientExecutor

if TYPE_CHECKING:
    import ccxt

logger = logging.getLogger(__name__)


def _maybe_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def summarize_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a CCXT order into plain fields.

    Never raises on missing values: by the time this runs the order already
    exists on the exchange, and an exception here would look like a failed
    order to the caller (who might then place it again). CCXT reports
    unknown fields such as 'average' and 'filled' as None until the order
    fills, so they map to None here.
    """
    fee = order.get('fee') or {}
    return {
        'id': order.get('id'),
        'symbol': order.get('symbol'),
        'side': order.get('side'),
        'status': order.get('status'),
        'amount': _maybe_float(order.get('amount')),
        'filled': _maybe_float(order.get('filled')),
        'average_price': _maybe_float(order.get('average')),
        'cost': _maybe_float(order.get('cost')),
        'fee': _maybe_float(fee.get('cost')),
        'fee_currency': fee.get('currency'),
    }


class CcxtBroker:
    """Account balance and market orders on one CCXT exchange."""

    def __init__(
        self,
        settings: Optional[ExecutionSettings] = None,
        data_settings: Optional[DataSettings] = None,
        exchange: Optional['ccxt.Exchange'] = None,
    ):
        """
        Args:
            settings: Execution settings (ccxt_testnet); defaults to global.
            data_settings: Exchange id, API keys and retry policy; defaults to global.
            exchange: Pre-built CCXT exchange (or a test double).
        """
        settings = settings if settings is not None else get_settings().execution
        data_settings = data_settings if data_settings is not None else get_settings().data

        self.exchange_id = data_settings.ccxt_exchange_id
        self.testnet = settings.ccxt_testnet
        key = data_settings.ccxt_api_key
        secret = data_settings.ccxt_api_secret
        self._api_key = key.get_secret_value() if key else ''
        self._api_secret = secret.get_secret_value() if secret else ''

        self.exchange = exchange if exchange is not None else create_exchange(
            self.exchange_id,
            api_key=self._api_key,
            api_secret=self._api_secret,
            sandbox=self.testnet,
            require_sandbox=self.testnet,
        )

        retryable, non_retryable = ccxt_error_policy()
        self._executor = ResilientExecutor(
            name=f'ccxt-trading-{self.exchange_id}',
            non_retryable=non_retryable,
            retryable_exceptions=retryable,
            settings=data_settings,
        )

    def _require_credentials(self, action: str) -> None:
        if not self._api_key or not self._api_secret:
            raise ValueError(f"CCXT_API_KEY and CCXT_API_SECRET are required for {action}.")

    def get_balance(self, currency: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """Total balance of `currency` (0.0 if absent), or all non-empty balances."""
        self._require_credentials('balance queries')
        balance = self._executor.call(self.exchange.fetch_balance)
        totals = {
            code: float(amount)
            for code, amount in (balance.get('total') or {}).items()
            if amount is not None
        }
        if currency:
            return totals.get(currency.upper(), 0.0)
        return totals

    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """Place a market order ('buy' or 'sell', amount in base currency)."""
        self._require_credentials('trading')
        if side not in ('buy', 'sell'):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if not amount > 0:
            raise ValueError(f"amount must be positive, got {amount}")

        # Deliberately not retried: order placement is not idempotent.
        order = self.exchange.create_market_order(normalize_ccxt_symbol(symbol), side, amount)
        summary = summarize_order(order)
        logger.info(
            "Placed %s %s %s on %s%s: id=%s status=%s",
            side, amount, summary['symbol'], self.exchange_id,
            ' (testnet)' if self.testnet else '', summary['id'], summary['status'],
        )
        return summary
