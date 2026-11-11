"""Tests for the Alpaca and CCXT providers, using in-memory fakes (no network)."""

import math
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import ccxt
import pandas as pd
import pytest
import requests

from src.config.settings import DataSettings
from src.market_data.providers import AlpacaDataClient, CryptoExchangeClient
from src.market_data.providers.base import to_float
from src.market_data.providers.ccxt_client import create_exchange
from src.market_data.resilience import CircuitBreakerOpen
from tests.factories import data_settings

HOUR_MS = 3_600_000
T0 = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp() * 1000)


class FakeExchange:
    """CCXT stand-in serving hourly BTC/USDT candles with a per-page cap."""

    iso8601 = staticmethod(ccxt.Exchange.iso8601)

    def __init__(self, n_candles=2000, page_cap=300, failures=()):
        self.candles = [
            [T0 + i * HOUR_MS, 100.0, 101.0, 99.0, 100.5, 0.25] for i in range(n_candles)
        ]
        self.page_cap = page_cap
        self.failures = list(failures)
        self.calls = []

    def load_markets(self):
        return {'BTC/USDT': {}}

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append(since)
        if self.failures:
            raise self.failures.pop(0)
        n = min(limit or self.page_cap, self.page_cap)
        if since is None:
            return self.candles[-n:]
        return [c for c in self.candles if c[0] >= since][:n]


def crypto_client(exchange=None, **settings):
    return CryptoExchangeClient(data_settings(**settings), exchange=exchange or FakeExchange())


# --- CCXT ------------------------------------------------------------------

def test_paging_continues_past_short_pages():
    exchange = FakeExchange(page_cap=300)  # e.g. Coinbase/OKX, below page_limit=1000
    result = crypto_client(exchange).fetch_ohlcv(
        'BTC/USDT', '2024-01-01', '2024-02-11T16:00', '1h'
    )

    assert result['n_bars'] == 1000
    ts = pd.to_datetime([b['timestamp'] for b in result['data']])
    assert (ts.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert len(exchange.calls) == 4


def test_end_date_is_exclusive():
    result = crypto_client().fetch_ohlcv('BTC/USDT', '2024-01-01', '2024-01-02', '1h')
    assert result['n_bars'] == 24
    assert result['data'][-1]['timestamp'] == '2024-01-01T23:00:00.000Z'


def test_end_without_start_returns_candles_before_end():
    result = crypto_client(ccxt_default_limit=24).fetch_ohlcv(
        'BTC/USDT', end_date='2024-01-03', interval='1h'
    )
    assert result['n_bars'] == 24
    assert result['data'][0]['timestamp'] == '2024-01-02T00:00:00.000Z'


def test_no_dates_returns_latest_candles():
    exchange = FakeExchange()
    result = crypto_client(exchange, ccxt_default_limit=10).fetch_ohlcv('BTC/USDT', interval='1h')
    assert result['n_bars'] == 10
    assert exchange.calls == [None]


def test_symbol_is_normalized_and_volume_kept_fractional():
    result = crypto_client().fetch_ohlcv('btc-usdt', '2024-01-01', '2024-01-01T02:00', '1h')
    assert result['symbol'] == 'BTC/USDT'
    assert result['data'][0]['volume'] == 0.25


def test_unknown_symbol_fails_without_fetching():
    exchange = FakeExchange()
    with pytest.raises(ValueError, match='not found'):
        crypto_client(exchange).fetch_ohlcv('DOGE/EUR', '2024-01-01')
    assert exchange.calls == []


def test_network_errors_are_retried():
    exchange = FakeExchange(failures=[ccxt.RequestTimeout('slow'), ccxt.ExchangeNotAvailable('503')])
    result = crypto_client(exchange, max_retry_attempts=3).fetch_ohlcv(
        'BTC/USDT', '2024-01-01', '2024-01-01T05:00', '1h'
    )
    assert result['n_bars'] == 5


def test_repeated_network_failures_open_the_circuit():
    exchange = FakeExchange(failures=[ccxt.RequestTimeout('slow')] * 10)
    client = crypto_client(exchange, max_retry_attempts=0, circuit_breaker_threshold=1)

    with pytest.raises(ccxt.RequestTimeout):
        client.fetch_ohlcv('BTC/USDT', '2024-01-01', '2024-01-02', '1h')
    with pytest.raises(CircuitBreakerOpen):
        client.fetch_ohlcv('BTC/USDT', '2024-01-01', '2024-01-02', '1h')


def test_cache_serves_repeat_requests():
    exchange = FakeExchange()
    client = crypto_client(exchange, cache_ttl_seconds=60)
    first = client.fetch_ohlcv('BTC/USDT', '2024-01-01', '2024-01-02', '1h')
    second = client.fetch_ohlcv('BTC/USDT', '2024-01-01', '2024-01-02', '1h')

    assert not first['from_cache'] and second['from_cache']
    assert second['data'] == first['data']
    assert len(exchange.calls) == 1


def test_create_exchange_supports_coinbase_and_rejects_unknown():
    assert create_exchange('coinbase').id == 'coinbase'
    with pytest.raises(ValueError, match='not found'):
        create_exchange('not-an-exchange')


def test_market_data_does_not_use_testnet_by_default():
    assert DataSettings().ccxt_sandbox is False


def test_require_sandbox_refuses_production(monkeypatch):
    def no_sandbox(self, enabled):
        raise ccxt.NotSupported('no testnet')
    monkeypatch.setattr(ccxt.binance, 'set_sandbox_mode', no_sandbox)

    assert create_exchange('binance', sandbox=True).id == 'binance'  # data: warn only
    with pytest.raises(ValueError, match='no testnet'):
        create_exchange('binance', sandbox=True, require_sandbox=True)


# --- Alpaca ----------------------------------------------------------------

def _alpaca_bar(ts):
    return SimpleNamespace(timestamp=ts, open=1.0, high=2.0, low=0.5, close=1.5, volume=1000)


class FakeAlpacaData:
    def __init__(self, bars, failures=()):
        self.bars = bars
        self.failures = list(failures)
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(data={request.symbol_or_symbols: self.bars})


def alpaca_client(fake, **settings):
    return AlpacaDataClient(
        data_settings(alpaca_api_key='key', alpaca_api_secret='secret', **settings),
        data_client=fake,
    )


def test_alpaca_end_date_is_exclusive():
    jan2 = datetime(2024, 1, 2, 5, tzinfo=timezone.utc)
    jan3 = datetime(2024, 1, 3, 5, tzinfo=timezone.utc)
    fake = FakeAlpacaData([_alpaca_bar(jan2), _alpaca_bar(jan3)])

    result = alpaca_client(fake).fetch_ohlcv('aapl', '2024-01-01', '2024-01-03T05:00Z')

    assert result['symbol'] == 'AAPL'
    assert [b['timestamp'] for b in result['data']] == [jan2.isoformat()]
    # alpaca-py stores request bounds as naive UTC
    assert fake.requests[0].start.replace(tzinfo=None) == datetime(2024, 1, 1)


def test_alpaca_connection_errors_are_retried():
    fake = FakeAlpacaData(
        [_alpaca_bar(datetime(2024, 1, 2, 5, tzinfo=timezone.utc))],
        failures=[requests.exceptions.ConnectionError('reset'), requests.exceptions.ReadTimeout('slow')],
    )
    result = alpaca_client(fake, max_retry_attempts=2).fetch_ohlcv('AAPL', '2024-01-01')
    assert result['n_bars'] == 1
    assert len(fake.requests) == 3


def test_alpaca_requires_credentials():
    with pytest.raises(ValueError, match='ALPACA_API_KEY'):
        AlpacaDataClient(data_settings())


@pytest.mark.parametrize('value,expected', [
    (101, 101.0),
    ('101.5', 101.5),
    (Decimal('2.25'), 2.25),
])
def test_to_float_accepts_numbers_and_numeric_strings(value, expected):
    assert to_float(value) == expected


def test_to_float_missing_is_nan_but_garbage_raises():
    assert math.isnan(to_float(None))
    for garbage in ('abc', ''):
        with pytest.raises(ValueError):
            to_float(garbage)
