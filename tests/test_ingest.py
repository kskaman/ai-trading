"""Tests for DataIngestionService routing, failover and the ingest pipeline."""

import pytest

from src.market_data.ingest import DataIngestionService
from src.market_data.providers.base import MarketDataProvider
from src.market_data.store import FinancialDataStore
from tests.factories import data_settings, make_ohlcv, to_bars


class FakeProvider(MarketDataProvider):
    def __init__(self, name, bars=(), error=None):
        self.name = name
        super().__init__(data_settings(max_retry_attempts=0))
        self.bars = list(bars)
        self.error = error
        self.calls = 0

    def _fetch_bars(self, symbol, start_date, end_date, interval):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.bars)


def service(store=None, **providers):
    return DataIngestionService(providers, store=store, settings=data_settings())


BARS = to_bars(make_ohlcv(30))


def test_equities_route_to_primary_and_pairs_to_crypto_source():
    alpaca, ccxt = FakeProvider('alpaca', BARS), FakeProvider('ccxt', BARS)
    svc = service(alpaca=alpaca, ccxt=ccxt)

    assert svc.fetch('AAPL')['source'] == 'alpaca'
    assert svc.fetch('BTC/USDT')['source'] == 'ccxt'


def test_primary_failure_fails_over_to_backup():
    svc = service(
        alpaca=FakeProvider('alpaca', error=RuntimeError('down')),
        yfinance=FakeProvider('yfinance', BARS),
    )
    assert svc.fetch('AAPL')['source'] == 'yfinance'


def test_crypto_does_not_fail_over_to_equity_backup():
    backup = FakeProvider('yfinance', BARS)
    svc = service(ccxt=FakeProvider('ccxt', error=RuntimeError('down')), yfinance=backup)
    with pytest.raises(RuntimeError, match='All data sources failed'):
        svc.fetch('BTC/USDT')
    assert backup.calls == 0


def test_pinned_source_disables_failover():
    backup = FakeProvider('yfinance', BARS)
    svc = service(alpaca=FakeProvider('alpaca', error=RuntimeError('down')), yfinance=backup)
    with pytest.raises(RuntimeError):
        svc.fetch('AAPL', source='alpaca')
    assert backup.calls == 0


def test_ingest_persists_raw_bars_and_returns_canonical_frame(tmp_path):
    with FinancialDataStore(str(tmp_path / 'm.duckdb'), str(tmp_path / 'pq')) as store:
        svc = service(store=store, alpaca=FakeProvider('alpaca', BARS))
        df = svc.ingest('aapl', start_date='2024-01-01')

        assert len(df) == 30
        assert (df['symbol'] == 'AAPL').all()
        assert str(df['timestamp'].dtype) == 'datetime64[ns, UTC]'
        assert len(store.query_ohlcv('AAPL')) == 30


def test_ingest_can_fill_gaps_and_clean():
    bars = to_bars(make_ohlcv(48, freq='h').drop(index=[20]))
    svc = service(ccxt=FakeProvider('ccxt', bars))

    df = svc.ingest('BTC/USDT', interval='1h', persist=False, fill_gaps=True, clean=True)

    assert len(df) == 48
    assert df['was_forward_filled'].sum() == 1
    assert {'is_outlier', 'is_stale', 'ohlc_valid'} <= set(df.columns)


def test_ingest_with_no_bars_returns_empty_frame_and_stores_nothing(tmp_path):
    with FinancialDataStore(str(tmp_path / 'm.duckdb'), str(tmp_path / 'pq')) as store:
        df = service(store=store, alpaca=FakeProvider('alpaca', [])).ingest('AAPL')
        assert df.empty and 'close' in df.columns
        assert store.query_ohlcv('AAPL').empty


def test_ingest_many_skips_failures():
    svc = service(alpaca=FakeProvider('alpaca', BARS), ccxt=FakeProvider('ccxt', error=RuntimeError('x')))
    frames = svc.ingest_many(['AAPL', 'BTC/USDT'], persist=False)
    assert list(frames) == ['AAPL']


def test_from_settings_skips_unavailable_providers():
    svc = DataIngestionService.from_settings(data_settings(ccxt_exchange_id='not-an-exchange'))
    assert svc.providers == {}

    svc = DataIngestionService.from_settings(
        data_settings(alpaca_api_key='key', alpaca_api_secret='secret')
    )
    assert set(svc.providers) == {'alpaca', 'ccxt'}
