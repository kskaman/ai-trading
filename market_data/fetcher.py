from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Final

import ccxt
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)
OHLCV_COLUMNS: Final = ["open", "high", "low", "close", "volume"]


class MarketDataFetcher:
    """Fetch equity and cryptocurrency candles in one normalized format."""

    def __init__(self, exchange_id: str = "binance") -> None:
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown CCXT exchange: {exchange_id}")

        self.exchange = exchange_class({"enableRateLimit": True})

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=index, dtype="float64")

    @classmethod
    def _normalize_ohlcv(
        cls,
        data: pd.DataFrame,
        timestamp_column: str | None = None,
        timestamp_unit: str | None = None,
    ) -> pd.DataFrame:
        if data.empty:
            return cls._empty_frame()

        normalized = data.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            for level in range(normalized.columns.nlevels):
                labels = {
                    str(value).strip().lower()
                    for value in normalized.columns.get_level_values(level)
                }
                if set(OHLCV_COLUMNS).issubset(labels):
                    normalized.columns = normalized.columns.get_level_values(level)
                    break

        normalized.columns = [str(column).strip().lower() for column in normalized.columns]
        timestamp_name = timestamp_column.lower() if timestamp_column else None
        timestamps = normalized.pop(timestamp_name) if timestamp_name else normalized.index

        missing_columns = set(OHLCV_COLUMNS).difference(normalized.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Provider response is missing OHLCV columns: {missing}")

        normalized = normalized.loc[:, OHLCV_COLUMNS]
        normalized.index = pd.DatetimeIndex(
            pd.to_datetime(
                timestamps, unit=timestamp_unit, utc=True, errors="coerce"
            ),
            name="timestamp",
        )

        for column in OHLCV_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        normalized = normalized[~normalized.index.isna()]
        normalized = normalized.dropna(subset=OHLCV_COLUMNS)
        normalized = normalized[~normalized.index.duplicated(keep="last")]
        return normalized.sort_index().astype("float64")

    @staticmethod
    def _parse_timestamp(value: str) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _fetch_equity(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=timeframe,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            return self._normalize_ohlcv(data)
        except Exception as error:
            LOGGER.warning("Could not fetch equity data for %s: %s", symbol, error)
            return self._empty_frame()

    def _fetch_crypto(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        start = self._parse_timestamp(start_date)
        end = self._parse_timestamp(end_date)
        start_milliseconds = int(start.timestamp() * 1000)
        since = start_milliseconds
        end_milliseconds = int(end.timestamp() * 1000)
        rows: list[list[float]] = []

        try:
            while since < end_milliseconds:
                batch = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since, limit=1000
                )
                if not batch:
                    break

                rows.extend(
                    row
                    for row in batch
                    if start_milliseconds <= row[0] < end_milliseconds
                )
                next_since = int(batch[-1][0]) + 1
                if next_since <= since:
                    break
                since = next_since

            data = pd.DataFrame(
                rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            return self._normalize_ohlcv(
                data, timestamp_column="timestamp", timestamp_unit="ms"
            )
        except (ccxt.NetworkError, ccxt.ExchangeError) as error:
            LOGGER.warning("Could not fetch crypto data for %s: %s", symbol, error)
            return self._empty_frame()

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return candles in the half-open interval ``[start_date, end_date)``."""
        if not symbol.strip():
            raise ValueError("symbol must not be empty")
        if not timeframe.strip():
            raise ValueError("timeframe must not be empty")

        now = datetime.now(UTC)
        end_date = end_date or now.strftime("%Y-%m-%d")
        start_date = start_date or (now - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

        start = self._parse_timestamp(start_date)
        end = self._parse_timestamp(end_date)
        if start >= end:
            raise ValueError("start_date must be earlier than end_date")

        if "/" in symbol:
            return self._fetch_crypto(symbol, timeframe, start_date, end_date)
        return self._fetch_equity(symbol, timeframe, start_date, end_date)