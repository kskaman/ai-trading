# How-to guides

Short recipes for common tasks. Every snippet runs from the project root with
`python -m ...` (see [Getting started](getting-started.md)). For every
parameter, see the [API reference](reference/ingest.md).

## Fetch bars

```python
from src.market_data.ingest import DataIngestionService

service = DataIngestionService.from_settings()  # builds providers from your settings
df = service.ingest('AAPL', start_date='2024-01-01', end_date='2024-07-01', persist=False)
```

- `interval`: `'1m'`, `'5m'`, `'15m'`, `'1h'`, `'4h'` or `'1d'` (Alpaca names like `'1Day'` work too).
- Pairs such as `'BTC/USDT'` go to the crypto exchange automatically.
- `source='ccxt'` forces one provider and turns off failover.
- `service.ingest_many(['AAPL', 'MSFT'], start_date=...)` returns `{symbol: DataFrame}`;
  failures are logged and skipped.
- `service.fetch(...)` returns the raw bar dicts instead of a DataFrame.

The result follows the [standard DataFrame shape](architecture.md#rules-every-module-follows).

## Save and load bars

```python
from src.market_data.ingest import DataIngestionService
from src.market_data.store import FinancialDataStore

with FinancialDataStore() as store:  # data/market_data.duckdb
    service = DataIngestionService.from_settings(store=store)
    service.ingest('AAPL', start_date='2020-01-01', end_date='2025-01-01')  # saved by default

    bars = store.query_ohlcv('AAPL', start_date='2024-01-01')  # no API call
    store.export_to_parquet('AAPL', '1d')  # data/parquet/aapl/1d/2024-01.parquet, ...
```

- Re-ingesting the same dates replaces those bars, so reruns are safe.
- The database file stays locked while open: use `with`, or call `store.close()`.
- `store.query_parquet('AAPL')` reads the exported files back.

## Fill gaps and flag bad data

```python
from src.market_data.cleaning import CleaningReport, clean_ohlcv, forward_fill_gaps

df = forward_fill_gaps(bars, interval='1d', market='equity')  # do this first
df = clean_ohlcv(df)
print(CleaningReport.from_frame(df))  # "124 bars: 0 return outliers, ..."

good = df[df['ohlc_valid'] & ~df['is_outlier'] & ~df['is_stale']]
```

`service.ingest(..., fill_gaps=True, clean=True)` does both in one call.

| Column added | Meaning |
|---|---|
| `was_forward_filled` | Bar was missing and inserted: flat at the last close, volume 0 |
| `log_return` | Bar-to-bar log return |
| `is_outlier` | Return is more than `max_return_std` (5) standard deviations from the previous 50 bars |
| `is_stale` | Volume is below `min_volume_threshold` (10%) of the previous 50 bars' median |
| `ohlc_valid` | Prices make sense together (high is highest, low is lowest, all positive) |

Gaps longer than `max_gap_bars` (5) are inserted as empty (NaN) rows rather than
filled. Stocks use the real NYSE calendar, so weekends and holidays are never
gaps; intraday stock data needs an explicit timeline (`expected=`).

## Build a training table

```python
from src.features.engineer import FeatureEngineer
from src.features.labels import forward_returns

engineer = FeatureEngineer(interval='1d', market='equity')
table = engineer.compute_all_features(df).join(forward_returns(df, horizons=(5,)))

usable = table['ohlc_valid'] & ~table['is_outlier'] & ~table['is_stale']
table = table[usable].dropna(subset=engineer.feature_names + ['forward_return_5'])

X = table[engineer.feature_names]  # model inputs
y = table['forward_return_5']      # what to predict: return over the next 5 bars
```

- The first ~200 rows drop out: the longest feature (`price_vs_ma_200`) needs that much history.
- Split train/test **by time**, never randomly: random splits leak the future into training.
- Use `market='crypto'` for crypto so volatility is annualized over 365 days.

All windows are counted in bars.

| Group | Columns |
|---|---|
| Returns | `return_1`, `return_3`, `return_5`, `return_10`, `return_20`, `return_60`, `log_return_1` |
| Volatility | `volatility_5`, `volatility_10`, `volatility_20`, `volatility_60`, `vol_ratio_5_60`, `garman_klass_var`, `gk_volatility_5`, `gk_volatility_20` |
| Momentum | `ma_crossover_5_20`, `ma_crossover_10_50`, `ma_crossover_20_60`, `price_vs_ma_20`, `price_vs_ma_50`, `price_vs_ma_200`, `rsi_14`, `rsi_21`, `macd`, `macd_signal`, `macd_histogram` |
| Volume | `volume_ratio_10`, `volume_ratio_20`, `volume_ratio_60`, `obv_ratio_20`, `vwap_deviation_20`, `mfi_14` |
| Patterns | `range_pct`, `close_position`, `body_size`, `upper_wick`, `lower_wick`, `bb_position_20` |

What each column means is in the [features API reference](reference/features.md).

## Backtest a strategy

Strategies need several symbols already in the store (see
[Save and load bars](#save-and-load-bars)). How each strategy works is
explained on the [Strategies](strategies.md) page.

```python
from src.analytics.performance import PerformanceAnalyzer
from src.analytics.portfolio import portfolio_returns
from src.market_data.panel import load_panel
from src.market_data.store import FinancialDataStore
from src.strategies.momentum import CrossSectionalMomentum

symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'JPM', 'XOM', 'JNJ', 'WMT']
with FinancialDataStore() as store:
    prices = load_panel(store, symbols, start_date='2020-01-01', end_date='2025-01-01')

strategy = CrossSectionalMomentum(n_long=3, n_short=3)
weights = strategy.target_weights(prices)       # one column per symbol
result = portfolio_returns(weights, prices)     # trades next bar, charges RISK_SLIPPAGE_BPS

report = PerformanceAnalyzer.for_interval(result['net_return'], '1d', 'equity')
print(report.summary(n_trials=1))               # raise n_trials for every variant you try
```

- `load_panel` returns one close-price column per symbol. A symbol with no stored data is left out, so check `prices.columns`.
- `portfolio_returns` also returns `gross_return`, `turnover` and `cost`, so you can see how much of the result the costs ate.
- The returns start flat until the strategy has enough history (here `lookback + skip` = 141 bars). Drop those rows if you want metrics for the trading period only.

**Pairs:** test the pair on an earlier period, then trade the later one:

```python
from src.strategies.pairs_trading import PairsTradingStrategy, check_cointegration

formation, trading_start = prices.loc[:'2022-12-31'], '2023-01-01'
test = check_cointegration(formation['KO'], formation['PEP'])
print(test.is_cointegrated, test.p_value, test.hedge_ratio, test.half_life)

strategy = PairsTradingStrategy('KO', 'PEP', lookback=100)
weights = strategy.target_weights(prices[['KO', 'PEP']]).loc[trading_start:]
spread = strategy.spread(prices[['KO', 'PEP']])  # hedge_ratio, z_score, position per bar
```

`KalmanPairsStrategy('KO', 'PEP')` is used the same way. `MultiFactorModel()`
works like momentum, but by default needs 40 symbols (20 long, 20 short).

## Stream real-time data

```python
import asyncio

from src.market_data.streaming import AlpacaWebSocketFeed


async def main():
    feed = AlpacaWebSocketFeed()

    async def on_trade(trade):
        print(trade['symbol'], trade['price'], trade['size'])

    feed.on_trade(on_trade)
    await feed.connect_and_stream(['AAPL', 'MSFT'], streams=['trades'])


asyncio.run(main())
```

- Streams: `'trades'`, `'quotes'`, `'bars'` (one-minute bars); register handlers
  with `on_trade`, `on_quote`, `on_bar`, `on_status`.
- Runs until you press Ctrl+C or call `await feed.stop()`.
- A handler that raises is logged and skipped; the feed keeps running.
- The default URL is Alpaca's free IEX feed (`DATA_ALPACA_WS_URL`).

## Check the account and place orders

```python
from src.execution import AlpacaBroker, CcxtBroker

alpaca = AlpacaBroker()  # paper account while ALPACA_PAPER=true
print(alpaca.get_account_info())  # equity, cash, buying power, ...
print(alpaca.get_positions())

crypto = CcxtBroker()  # needs CCXT_API_KEY / CCXT_API_SECRET
print(crypto.get_balance('USDT'))
crypto.create_market_order('BTC/USDT', 'buy', 0.001)
```

> **Safety rules**
>
> - Crypto orders go to the exchange **testnet** while `EXEC_CCXT_TESTNET=true`
>   (the default). If the exchange has no testnet, `CcxtBroker()` raises
>   instead of trading for real. This switch is separate from `ALPACA_PAPER`.
> - Orders are **never retried**: a retry could place the order twice.
> - Placing Alpaca orders is not implemented yet.

## Add a data provider

Subclass `MarketDataProvider` and implement `_fetch_bars`. The base class adds
caching, retries and the circuit breaker.

```python
import requests

from src.market_data.providers.base import MarketDataProvider


class PolygonClient(MarketDataProvider):
    name = 'polygon'

    def __init__(self, settings=None):
        super().__init__(
            settings,
            retryable=(requests.exceptions.ConnectionError, requests.exceptions.Timeout),
        )

    def _fetch_bars(self, symbol, start_date, end_date, interval):
        # Call the API here and return bars in [start_date, end_date):
        return [
            {'timestamp': '2024-01-02T05:00:00+00:00', 'open': 187.15, 'high': 188.44,
             'low': 183.89, 'close': 185.64, 'volume': 82488700.0},
        ]
```

Checklist:

- `symbol` and `interval` arrive already normalized (`'AAPL'`, `'1d'`).
- Return only bars with `start_date <= timestamp < end_date`.
- Timestamps are ISO 8601 in UTC; daily stock bars at midnight New York time.
- Pass the exceptions your client library raises for network problems as
  `retryable`; raise `ValueError` for bad input (it is never retried).

Use it by passing it in, e.g.
`DataIngestionService({'polygon': PolygonClient()})` with
`DATA_PRIMARY_SOURCE=polygon`, or add it to `DataIngestionService.from_settings`.

## Add features

1. Write a function in `src/features/` that takes the bar DataFrame and returns
   a DataFrame with **only the new columns** (same index).
2. Add it to the list in `FeatureEngineer.compute_all_features`.
3. Add it to the table above.

The existing tests then check the new columns automatically: no look-ahead, no
duplicates of other features, and listed in these docs.

## Keep these docs current

| What changed | Do this |
|---|---|
| A function or class | Update its docstring (Google style: `Args:`, `Returns:`, `Raises:`); the API reference picks it up |
| A setting | Edit its `Field(description=...)`, then run `python -m scripts.gen_config_docs` (a test fails if you forget) |
| Behavior or structure | Edit these guides or [Architecture](architecture.md) by hand |

Preview the site with `mkdocs serve`; `mkdocs build --strict` fails on broken
links or docstring problems.
