"""Tests for the execution layer (brokers), using fakes (no network, no orders)."""

from types import SimpleNamespace

import ccxt
import pytest

from src.config.settings import ExecutionSettings
from src.execution import AlpacaBroker, CcxtBroker, summarize_order
from tests.factories import data_settings

UNFILLED_ORDER = {
    # CCXT reports unknown values as None until the order fills.
    'id': '42', 'symbol': 'BTC/USDT', 'side': 'buy', 'status': 'open',
    'amount': 0.5, 'filled': None, 'average': None, 'cost': None, 'fee': None,
}


def test_summarize_order_tolerates_missing_values():
    summary = summarize_order(UNFILLED_ORDER)
    assert summary['id'] == '42'
    assert summary['amount'] == 0.5
    assert summary['filled'] is None and summary['average_price'] is None
    assert summary['fee'] is None


def test_summarize_order_reports_fee_not_cost():
    order = {**UNFILLED_ORDER, 'status': 'closed', 'filled': 0.5, 'average': 40_000,
             'cost': 20_000, 'fee': {'cost': 20.0, 'currency': 'USDT'}}
    summary = summarize_order(order)
    assert (summary['cost'], summary['fee'], summary['fee_currency']) == (20_000, 20.0, 'USDT')


class FakeTradingExchange:
    def __init__(self, error=None):
        self.error = error
        self.orders = []

    def create_market_order(self, symbol, side, amount):
        self.orders.append((symbol, side, amount))
        if self.error is not None:
            raise self.error
        return dict(UNFILLED_ORDER, symbol=symbol, side=side, amount=amount)

    def fetch_balance(self):
        return {'total': {'BTC': 0.5, 'ETH': None}}


def ccxt_broker(exchange, **settings):
    return CcxtBroker(
        settings=ExecutionSettings(ccxt_testnet=True),
        data_settings=data_settings(ccxt_api_key='key', ccxt_api_secret='secret', **settings),
        exchange=exchange,
    )


def test_market_order_is_placed_and_summarized():
    exchange = FakeTradingExchange()
    summary = ccxt_broker(exchange).create_market_order('btc-usdt', 'buy', 0.5)
    assert exchange.orders == [('BTC/USDT', 'buy', 0.5)]
    assert summary['status'] == 'open'


def test_market_orders_are_never_retried():
    exchange = FakeTradingExchange(error=ccxt.RequestTimeout('timeout'))
    with pytest.raises(ccxt.RequestTimeout):
        ccxt_broker(exchange, max_retry_attempts=5).create_market_order('BTC/USDT', 'buy', 0.5)
    assert len(exchange.orders) == 1


@pytest.mark.parametrize('side,amount', [('hold', 1), ('buy', 0), ('sell', -1)])
def test_invalid_orders_are_rejected(side, amount):
    exchange = FakeTradingExchange()
    with pytest.raises(ValueError):
        ccxt_broker(exchange).create_market_order('BTC/USDT', side, amount)
    assert exchange.orders == []


def test_ccxt_broker_requires_credentials():
    broker = CcxtBroker(ExecutionSettings(), data_settings(), exchange=FakeTradingExchange())
    with pytest.raises(ValueError, match='CCXT_API_KEY'):
        broker.create_market_order('BTC/USDT', 'buy', 1)


def test_balance():
    broker = ccxt_broker(FakeTradingExchange())
    assert broker.get_balance('btc') == 0.5
    assert broker.get_balance('SOL') == 0.0
    assert broker.get_balance() == {'BTC': 0.5}


def test_testnet_mode_refuses_exchange_without_testnet(monkeypatch):
    def no_sandbox(self, enabled):
        raise ccxt.NotSupported('no testnet')
    monkeypatch.setattr(ccxt.binance, 'set_sandbox_mode', no_sandbox)

    with pytest.raises(ValueError, match='no testnet'):
        CcxtBroker(ExecutionSettings(ccxt_testnet=True), data_settings(ccxt_exchange_id='binance'))


def test_alpaca_live_does_not_make_crypto_orders_live(monkeypatch):
    monkeypatch.setenv('ALPACA_PAPER', 'false')
    broker = CcxtBroker(data_settings=data_settings(ccxt_exchange_id='binance'))
    assert broker.testnet is True
    assert broker.exchange.isSandboxModeEnabled


class FakeTradingClient:
    def get_account(self):
        return SimpleNamespace(equity='1000.5', cash='500', buying_power='2000',
                               daytrade_count=1, last_equity='990')

    def get_all_positions(self):
        return [SimpleNamespace(symbol='AAPL', qty='3', market_value='600',
                                unrealized_pl='12.5', avg_entry_price='196')]


def test_alpaca_broker_uses_alpaca_env_credentials(monkeypatch):
    monkeypatch.setenv('ALPACA_API_KEY', 'key')
    monkeypatch.setenv('ALPACA_API_SECRET', 'secret')
    monkeypatch.setenv('ALPACA_PAPER', 'false')

    broker = AlpacaBroker(trading_client=FakeTradingClient())

    assert broker.paper is False
    assert broker.get_account_info()['total_equity'] == 1000.5
    assert broker.get_positions()[0]['qty'] == 3.0


def test_alpaca_broker_requires_credentials():
    with pytest.raises(ValueError, match='ALPACA_API_KEY'):
        AlpacaBroker(trading_client=FakeTradingClient())
