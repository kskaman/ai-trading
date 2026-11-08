# AI Trading System

Collects market data, cleans it, turns it into features, and runs trading
strategies through a quick research backtest with performance metrics. Model
training, risk management, the event-driven backtester and automated live
trading are not built yet.

## Where to start

| If you want to... | Read |
|---|---|
| Install it and run something | [Getting started](getting-started.md) |
| Understand how the pieces fit and the rules they follow | [Architecture](architecture.md) |
| Do a specific task (fetch, store, clean, features, backtest, stream, trade, extend) | [How-to guides](guides.md) |
| Understand a strategy or a performance metric | [Strategies](strategies.md) |
| Add API keys or change a setting | [Configuration](configuration.md) |
| Look up a class or function | [API reference](reference/ingest.md) |

## What exists today

| Area | Status | Code |
|---|---|---|
| Historical bars from Alpaca (stocks) and CCXT exchanges (crypto) | Done | `src/market_data/providers` |
| Retries and circuit breaker for every API call | Done | `src/market_data/resilience.py` |
| Local storage in DuckDB, Parquet export | Done | `src/market_data/store.py` |
| Gap filling on the NYSE calendar, quality flags | Done | `src/market_data/cleaning` |
| 38 model features and forward-return labels | Done | `src/features` |
| Real-time trades/quotes/bars from Alpaca | Done | `src/market_data/streaming` |
| Momentum, pairs, Kalman pairs and multi-factor strategies | Done | `src/strategies` |
| Research backtest (weights → returns after costs) and performance metrics | Done | `src/analytics` |
| Account queries and market orders | Basic | `src/execution` |
| Model training, risk management, event-driven backtester | Not started | `src/models` (interface only) |
