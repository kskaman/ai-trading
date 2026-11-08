"""
Financial data storage using DuckDB and Parquet.

Zones:
    raw_ohlcv      bars exactly as the provider returned them (not filtered;
                   cleaning flags bad bars later via ohlc_valid)
    curated_ohlcv  reserved for cleaned/adjusted bars (not written yet)

Time zones: timestamps are stored as naive UTC in TIMESTAMP columns and
returned as UTC-aware pandas timestamps. The explicit conversion matters:
DuckDB casts a tz-aware value into a TIMESTAMP column using the session time
zone, which silently shifts every bar to the machine's local time.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Union

import duckdb
import pandas as pd

from src.market_data.schema import (
    OHLCV_COLUMNS, bars_to_frame, normalize_interval, normalize_ohlcv,
)

logger = logging.getLogger(__name__)

_RAW_COLUMNS = ('symbol', 'timestamp', 'interval', 'open', 'high', 'low', 'close', 'volume')
_PARTITION_FORMATS = {'day': '%Y-%m-%d', 'month': '%Y-%m', 'year': '%Y'}


def _quoted(columns: Iterable[str]) -> str:
    # "timestamp" and "interval" are SQL type names; quote every identifier.
    return ', '.join(f'"{c}"' for c in columns)


def _utc_naive(value: str) -> datetime:
    """Parse a date/time bound and express it as naive UTC (naive input is UTC)."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    return ts.to_pydatetime()


def _check_partition(partition_by: str) -> str:
    if partition_by not in _PARTITION_FORMATS:
        raise ValueError(
            f"partition_by must be one of {sorted(_PARTITION_FORMATS)}, got '{partition_by}'"
        )
    return partition_by


def symbol_dir(symbol: str) -> str:
    """Filesystem-safe directory name: 'BTC/USDT' -> 'btc_usdt'."""
    return re.sub(r'[^a-z0-9.]+', '_', symbol.lower()).strip('_')


class FinancialDataStore:
    """
    DuckDB-backed financial data store with Parquet export.

    Use as a context manager (or call close()) to release the database file.
    Arguments left as None come from DataSettings (DATA_DUCKDB_PATH,
    DATA_PARQUET_BASE_DIR, DATA_PARTITION_BY).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        parquet_dir: Optional[str] = None,
        partition_by: Optional[str] = None,
    ):
        if db_path is None or parquet_dir is None or partition_by is None:
            from src.config.settings import get_settings
            data_settings = get_settings().data
            db_path = db_path or data_settings.duckdb_path
            parquet_dir = parquet_dir or data_settings.parquet_base_dir
            partition_by = partition_by or data_settings.partition_by

        self.partition_by = _check_partition(partition_by)

        if db_path != ':memory:':
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(database=db_path)
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def __enter__(self) -> 'FinancialDataStore':
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _init_tables(self) -> None:
        """Create the database schema if tables don't exist."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS raw_ohlcv (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                interval VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, timestamp, interval)
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS curated_ohlcv (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                interval VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                adjusted_close DOUBLE,
                PRIMARY KEY (symbol, timestamp, interval)
            )
        """)

    def ingest_ohlcv(
        self,
        symbol: str,
        interval: str,
        data: Union[pd.DataFrame, Iterable[Mapping]],
    ) -> int:
        """
        Upsert OHLCV bars into the raw zone.

        Args:
            symbol: Ticker symbol.
            interval: Bar interval ('1d', '5m', etc.).
            data: Provider bar dicts or an OHLCV DataFrame.

        Returns:
            Number of rows written (new or replaced).
        """
        interval = normalize_interval(interval)
        frame = normalize_ohlcv(data) if isinstance(data, pd.DataFrame) else bars_to_frame(data)
        if frame.empty:
            return 0

        rows = frame[list(OHLCV_COLUMNS)].copy()
        rows['timestamp'] = rows['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        rows.insert(0, 'symbol', symbol)
        rows.insert(2, 'interval', interval)

        columns = _quoted(_RAW_COLUMNS)
        self.con.register('_incoming_bars', rows)
        try:
            self.con.execute(
                f"INSERT OR REPLACE INTO raw_ohlcv ({columns}) "
                f"SELECT {columns} FROM _incoming_bars"
            )
        finally:
            self.con.unregister('_incoming_bars')

        logger.debug("Stored %d %s bars for %s", len(rows), interval, symbol)
        return len(rows)

    def query_ohlcv(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d',
    ) -> pd.DataFrame:
        """
        Query raw OHLCV bars in [start_date, end_date).

        Args:
            symbol: Ticker symbol.
            start_date: Inclusive bound (date or datetime; naive means UTC).
            end_date: Exclusive bound (date or datetime; naive means UTC).
            interval: Bar interval.

        Returns:
            DataFrame sorted by timestamp, timestamps UTC-aware.
        """
        sql = 'SELECT * FROM raw_ohlcv WHERE symbol = ? AND "interval" = ?'
        params: List[object] = [symbol, normalize_interval(interval)]

        if start_date:
            sql += ' AND "timestamp" >= ?'
            params.append(_utc_naive(start_date))
        if end_date:
            sql += ' AND "timestamp" < ?'
            params.append(_utc_naive(end_date))

        sql += ' ORDER BY "timestamp"'
        return self._fetch(sql, params)

    def export_to_parquet(
        self,
        symbol: str,
        interval: str,
        partition_by: Optional[str] = None,
    ) -> str:
        """
        Export a symbol's raw bars to one Parquet file per partition.

        Files go to `<parquet_dir>/<symbol_dir(symbol)>/<interval>/<partition>.parquet`.

        Args:
            symbol: Ticker symbol.
            interval: Bar interval.
            partition_by: 'day', 'month' or 'year'; defaults to the store's
                `partition_by` (DATA_PARTITION_BY).

        Returns:
            Path to the exported directory ("" if there was nothing to export).
        """
        partition_by = _check_partition(partition_by or self.partition_by)

        df = self.query_ohlcv(symbol, interval=interval)
        if df.empty:
            return ""

        out_dir = self.parquet_dir / symbol_dir(symbol) / normalize_interval(interval)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Files hold naive UTC, the same convention as the database, so DuckDB
        # reads them back without a local time zone conversion.
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        partition = df['timestamp'].dt.strftime(_PARTITION_FORMATS[partition_by])

        for key, group in df.groupby(partition):
            group.to_parquet(out_dir / f"{key}.parquet", index=False, compression='snappy')

        return str(out_dir)

    def query_parquet(self, symbol: str, interval: str = '1d') -> pd.DataFrame:
        """Read a symbol's exported Parquet files (empty frame if none exist)."""
        directory = self.parquet_dir / symbol_dir(symbol) / normalize_interval(interval)
        if not any(directory.glob('*.parquet')):
            return pd.DataFrame(columns=list(_RAW_COLUMNS))

        return self._fetch(
            'SELECT * FROM read_parquet(?) ORDER BY "timestamp"',
            [str(directory / '*.parquet')],
        )

    def _fetch(self, sql: str, params: List[object]) -> pd.DataFrame:
        df = self.con.execute(sql, params).fetchdf()
        df['timestamp'] = (
            pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.as_unit('ns')
        )
        return df

    def close(self) -> None:
        """Close the database connection."""
        self.con.close()


# Usage example (needs ALPACA_API_KEY / ALPACA_API_SECRET for equities)
if __name__ == '__main__':
    from src.market_data.ingest import DataIngestionService

    logging.basicConfig(level=logging.INFO)

    with FinancialDataStore() as store:
        service = DataIngestionService.from_settings(store=store)
        service.ingest_many(['AAPL', 'MSFT', 'GOOGL'],
                            start_date='2024-01-01', end_date='2024-07-01')

        data = store.query_ohlcv('AAPL', start_date='2024-03-01')
        print(f"\nQueried {len(data)} AAPL bars from March 2024:")
        print(data.head())

        print(f"\nExported to: {store.export_to_parquet('AAPL', '1d')}")
