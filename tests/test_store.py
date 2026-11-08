"""Tests for the DuckDB/Parquet store."""

import pandas as pd
import pytest

from src.market_data.store import FinancialDataStore
from tests.factories import make_ohlcv, to_bars


@pytest.fixture
def store(tmp_path):
    with FinancialDataStore(
        db_path=str(tmp_path / 'db' / 'market.duckdb'),
        parquet_dir=str(tmp_path / 'parquet'),
    ) as s:
        # A non-UTC session time zone exposes any implicit local-time conversion.
        s.con.execute("SET TimeZone='America/St_Johns'")
        yield s


def test_creates_missing_parent_directories(tmp_path):
    db_path = tmp_path / 'does' / 'not' / 'exist' / 'market.duckdb'
    with FinancialDataStore(db_path=str(db_path), parquet_dir=str(tmp_path / 'pq')):
        pass
    assert db_path.exists()


def test_round_trip_keeps_utc_timestamps_and_values(store):
    df = make_ohlcv(5, start='2024-01-02 05:00')
    assert store.ingest_ohlcv('AAPL', '1d', to_bars(df)) == 5

    out = store.query_ohlcv('AAPL')
    pd.testing.assert_series_equal(out['timestamp'], df['timestamp'], check_names=False)
    pd.testing.assert_frame_equal(
        out[['open', 'high', 'low', 'close', 'volume']].reset_index(drop=True),
        df[['open', 'high', 'low', 'close', 'volume']],
    )
    assert (out['symbol'] == 'AAPL').all() and (out['interval'] == '1d').all()


def test_fractional_volume_is_not_rounded(store):
    df = make_ohlcv(2)
    df['volume'] = [0.4, 123.6]
    store.ingest_ohlcv('BTC/USDT', '1d', df)
    assert list(store.query_ohlcv('BTC/USDT')['volume']) == [0.4, 123.6]


def test_upsert_replaces_existing_bar(store):
    df = make_ohlcv(1)
    store.ingest_ohlcv('AAPL', '1d', df)
    df.loc[0, 'close'] = df.loc[0, 'high']
    store.ingest_ohlcv('AAPL', '1d', df)

    out = store.query_ohlcv('AAPL')
    assert len(out) == 1
    assert out['close'].iloc[0] == df.loc[0, 'high']


def test_raw_zone_keeps_invalid_bars(store):
    df = make_ohlcv(3)
    df.loc[1, 'high'] = df.loc[1, 'low'] - 1  # high < low
    store.ingest_ohlcv('AAPL', '1d', df)
    assert len(store.query_ohlcv('AAPL')) == 3


def test_query_start_inclusive_end_exclusive(store):
    store.ingest_ohlcv('AAPL', '1d', make_ohlcv(10, start='2024-01-01'))
    out = store.query_ohlcv('AAPL', start_date='2024-01-03', end_date='2024-01-06')
    assert [t.day for t in out['timestamp']] == [3, 4, 5]


def test_interval_aliases_are_normalized(store):
    store.ingest_ohlcv('AAPL', '1Day', make_ohlcv(2))
    assert len(store.query_ohlcv('AAPL', interval='1d')) == 2


def test_empty_data_writes_nothing(store):
    assert store.ingest_ohlcv('AAPL', '1d', []) == 0
    assert store.query_ohlcv('AAPL').empty


def test_parquet_round_trip(store, tmp_path):
    df = make_ohlcv(70, start='2024-01-01')
    store.ingest_ohlcv('BTC/USDT', '1d', df)

    out_dir = store.export_to_parquet('BTC/USDT', '1d', partition_by='month')
    assert out_dir.endswith('1d') and 'btc_usdt' in out_dir
    assert len(list((tmp_path / 'parquet' / 'btc_usdt' / '1d').glob('*.parquet'))) == 3

    pd.testing.assert_frame_equal(store.query_parquet('BTC/USDT'), store.query_ohlcv('BTC/USDT'))


def test_export_rejects_unknown_partition(store, tmp_path):
    store.ingest_ohlcv('AAPL', '1d', make_ohlcv(2))
    with pytest.raises(ValueError, match='partition_by'):
        store.export_to_parquet('AAPL', '1d', partition_by='week')
    with pytest.raises(ValueError, match='partition_by'):
        FinancialDataStore(':memory:', str(tmp_path / 'pq'), partition_by='week')


def test_partition_default_comes_from_settings(monkeypatch, tmp_path):
    monkeypatch.setenv('DATA_PARTITION_BY', 'year')
    with FinancialDataStore(':memory:', str(tmp_path / 'pq')) as s:
        s.ingest_ohlcv('AAPL', '1d', make_ohlcv(70, start='2024-12-01'))
        s.export_to_parquet('AAPL', '1d')
    assert sorted(p.name for p in (tmp_path / 'pq' / 'aapl' / '1d').glob('*')) == [
        '2024.parquet', '2025.parquet',
    ]


def test_query_parquet_without_files_is_empty(store):
    assert store.query_parquet('AAPL').empty
