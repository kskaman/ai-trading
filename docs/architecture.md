# Architecture

## The pipeline

`DataIngestionService.ingest()` runs the whole data path for one symbol:

```text
 'AAPL' / 'BTC/USDT'
        │
        ▼
 1. Route          pairs ('/') → crypto source (ccxt), everything else → primary (alpaca)
        │
        ▼
 2. Fetch          provider.fetch_ohlcv()  ── every call goes through ResilientExecutor:
        │                                     retry transient errors, circuit breaker,
        │                                     then fail over primary → backup (stocks only)
        ▼
 3. Normalize      bars_to_frame()          one DataFrame shape for every source
        │
        ├────────► 4. Store raw   FinancialDataStore.ingest_ohlcv()   (exactly as fetched)
        ▼
 5. Fill gaps      forward_fill_gaps()      optional: insert bars missing from the calendar
        │
        ▼
 6. Clean          clean_ohlcv()            optional: add is_outlier / is_stale / ohlc_valid
        │
        ▼
     DataFrame ──► FeatureEngineer.compute_all_features()   model inputs
              └──► forward_returns()                         prediction targets (kept separate)
```

Strategies work across symbols, so they start from the store instead:

```text
 load_panel()                 one close-price column per symbol
        │
        ▼
 strategy.target_weights()    one weight column per symbol (see rule 9)
        │
        ├────────► weights_to_signals()   LONG_ENTER / LONG_EXIT / SHORT_ENTER / SHORT_EXIT
        ▼
 portfolio_returns()          hold weights from the next bar, subtract costs
        │
        ▼
 PerformanceAnalyzer          Sharpe, Sortino, drawdown, deflated Sharpe, ...
```

## Where things live

| Module | Job | Main entry points |
|---|---|---|
| `src/config/settings.py` | All configuration, from env vars / `.env` | `get_settings()` |
| `src/config/di.py` | Optional wiring of shared components | `DependencyContainer` |
| `src/market_data/ingest.py` | Orchestrates the pipeline above | `DataIngestionService` |
| `src/market_data/providers/` | Fetch historical bars from an API | `AlpacaDataClient`, `CryptoExchangeClient` |
| `src/market_data/resilience.py` | Retries and circuit breaker | `ResilientExecutor` |
| `src/market_data/schema.py` | The canonical bar DataFrame | `bars_to_frame`, `ohlc_valid_mask` |
| `src/market_data/store.py` | DuckDB storage, Parquet export | `FinancialDataStore` |
| `src/market_data/panel.py` | Several symbols side by side | `load_panel`, `to_panel` |
| `src/market_data/cleaning/` | Gap filling and quality flags | `forward_fill_gaps`, `clean_ohlcv` |
| `src/market_data/streaming/` | Real-time data over WebSocket | `AlpacaWebSocketFeed` |
| `src/features/` | Model features and labels | `FeatureEngineer`, `forward_returns` |
| `src/strategies/` | Prices → target weights → signals | `CrossSectionalMomentum`, `PairsTradingStrategy`, `KalmanPairsStrategy`, `MultiFactorModel` |
| `src/analytics/` | Research backtest and performance metrics | `portfolio_returns`, `PerformanceAnalyzer` |
| `src/execution/` | Account info, balances, orders | `AlpacaBroker`, `CcxtBroker` |
| `src/models/` | Interface for ML models (next) | `BaseModel` |

Dependencies point one way: `features`, `strategies` and `execution` may use
`data`; `analytics` uses neither `strategies` nor `execution` (it only needs
weights and prices); `data` never imports anything above it; everything
reads `config`.

## Rules every module follows

These are the contracts that let you use one piece without reading the others.

**1. One DataFrame shape.** Every source produces the same columns:

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | `datetime64[ns, UTC]` | When the bar **opened** |
| `open`, `high`, `low`, `close` | `float64` | Prices (Alpaca bars are split-adjusted) |
| `volume` | `float64` | Fractional for crypto |
| `symbol` | `str` | Added by `ingest()` |

Rows are sorted by time with no duplicate timestamps.

**2. Timestamps are UTC everywhere.** Daily stock bars are stamped at midnight
New York time (Alpaca's convention, e.g. `2024-01-02 05:00 UTC`); daily crypto
bars at midnight UTC.

**3. Date ranges are start-inclusive, end-exclusive** in every provider and in
the store: `start_date='2024-01-01', end_date='2024-07-01'` means January
through June. Dates without a time zone are UTC.

**4. Raw data is never changed.** The store keeps exactly what the provider
returned. Cleaning never deletes bars; it adds flag columns and you decide what
to exclude.

**5. One symbol per DataFrame** for cleaning and features. Rolling windows must
not run from one symbol into the next, so mixed frames raise `ValueError`.

**6. Windows are counted in bars**, so `return_5` is five days on daily data
and 25 minutes on 5-minute data. Volatility is annualized using the interval and
market (252 sessions a year for stocks, 365 days for crypto).

**7. Features never look at the future; labels do.** Every feature at row *t*
uses only rows up to *t* (a test enforces this). Forward returns live in
`src/features/labels.py` and must never be used as model inputs.

**8. Configuration comes only from settings.** No numbers or keys are hard-coded
or passed around loosely. Every class also accepts a settings object, which is
how tests configure it.

**9. Strategies output weights.** `target_weights(prices)` returns one column
per symbol: positive is long, negative is short, and each row's absolute sum
is 1 (invested) or 0 (flat). Row *t* may use bar *t*'s close, so it is traded
from bar *t + 1*. Leverage belongs to the risk layer, not the strategy.

## How errors are handled

| Situation | What happens |
|---|---|
| Network blip, timeout, rate limit | Retried with exponential backoff (`DATA_MAX_RETRY_ATTEMPTS`) |
| Same provider keeps failing | Circuit breaker opens: calls fail fast with `CircuitBreakerOpen` for `DATA_CIRCUIT_BREAKER_TIMEOUT_SECONDS`, then one probe call is let through |
| Bad input (unknown symbol, bad interval) | Raised immediately as `ValueError`, never retried |
| Primary stock source fails | `ingest()` tries `DATA_BACKUP_SOURCE`; crypto has no backup |
| Every source fails | `RuntimeError("All data sources failed ...")` from the last error |
| Order placement fails | **Never retried** (it could place the order twice) |
| WebSocket drops | Reconnects with backoff; gives up after `DATA_WS_MAX_RECONNECT_ATTEMPTS` short-lived connections |

Each provider declares which exceptions count as transient, because client
libraries raise their own types (`ccxt.NetworkError`, `requests` errors) rather
than Python's built-in `ConnectionError`.

## Not built yet

Model training (walk-forward settings already exist in `ModelSettings`),
position sizing and risk checks (the limits in `RiskSettings`), an
event-driven backtester with fills and cash, order management and
monitoring.
