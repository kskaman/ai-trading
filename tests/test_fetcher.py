import unittest
from unittest.mock import Mock, patch

import pandas as pd

from market_data.fetcher import OHLCV_COLUMNS, MarketDataFetcher


class MarketDataFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = MarketDataFetcher.__new__(MarketDataFetcher)

    @patch("market_data.fetcher.yf.download")
    def test_equity_data_is_fetched_for_explicit_dates_and_normalized(
        self, download: Mock
    ) -> None:
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL"]]
        )
        download.return_value = pd.DataFrame(
            [
                [101, 103, 100, 102, 102, 1200],
                [100, 102, 99, 101, 101, 1000],
            ],
            index=pd.to_datetime(["2026-01-02", "2026-01-01"]),
            columns=columns,
        )

        result = self.fetcher.fetch_historical_data(
            "AAPL", "1d", "2026-01-01", "2026-01-03"
        )

        download.assert_called_once_with(
            "AAPL",
            start="2026-01-01",
            end="2026-01-03",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        self.assertEqual(list(result.columns), OHLCV_COLUMNS)
        self.assertEqual(result.index.name, "timestamp")
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertTrue(all(dtype == "float64" for dtype in result.dtypes))

    def test_crypto_data_is_paginated_bounded_and_normalized(self) -> None:
        start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
        end = int(pd.Timestamp("2026-01-03", tz="UTC").timestamp() * 1000)
        exchange = Mock()
        exchange.fetch_ohlcv.side_effect = [
            [
                [start - 1, 9, 11, 8, 10, 90],
                [start, 10, 12, 9, 11, 100],
                [start + 86_400_000, 11, 13, 10, 12, 110],
                [end, 12, 14, 11, 13, 120],
            ],
            [],
        ]
        self.fetcher.exchange = exchange

        result = self.fetcher.fetch_historical_data(
            "BTC/USDT", "1d", "2026-01-01", "2026-01-03"
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns), OHLCV_COLUMNS)
        self.assertTrue((result.index >= pd.Timestamp("2026-01-01", tz="UTC")).all())
        self.assertTrue((result.index < pd.Timestamp("2026-01-03", tz="UTC")).all())
        exchange.fetch_ohlcv.assert_called_once_with(
            "BTC/USDT", timeframe="1d", since=start, limit=1000
        )

    def test_invalid_date_range_is_rejected_before_fetching(self) -> None:
        with self.assertRaisesRegex(ValueError, "start_date must be earlier"):
            self.fetcher.fetch_historical_data(
                "AAPL", "1d", "2026-01-03", "2026-01-03"
            )


if __name__ == "__main__":
    unittest.main()