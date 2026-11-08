# AI Trading System

Collects market data (stocks from Alpaca, crypto from CCXT exchanges), stores
it in DuckDB, cleans it and turns it into features for machine-learning models.
Includes momentum, pairs, Kalman-pairs and multi-factor strategies with a
research backtest and performance metrics. Model training, risk management and
the event-driven backtester are not built yet.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
copy .env.example .env        # then add your Alpaca keys
.venv\Scripts\Activate.ps1

python -m pytest              # tests (offline, never read .env)
python -m src.market_data.ingest     # live demo: fetch AAPL + BTC/USDT, fill gaps, clean
```

```python
from src.market_data.ingest import DataIngestionService
from src.market_data.store import FinancialDataStore
from src.features.engineer import FeatureEngineer

with FinancialDataStore() as store:
    service = DataIngestionService.from_settings(store=store)
    df = service.ingest('AAPL', start_date='2024-01-01', end_date='2025-01-01',
                        fill_gaps=True, clean=True)

features = FeatureEngineer(interval='1d', market='equity').compute_all_features(df)
```

## Layout

```text
src/config/      settings (env vars / .env) and dependency wiring
src/market_data/        providers, retries, storage, cleaning, real-time stream
src/features/    model features and prediction labels
src/strategies/  momentum, pairs, Kalman pairs, multi-factor (prices -> weights)
src/analytics/   research backtest and performance metrics
src/execution/   account info, balances, orders
src/models/      model interface (not implemented yet)
docs/            documentation (Markdown + generated API reference)
tests/           pytest suite
```

## Documentation

| Read | For |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | Setup, running things, troubleshooting |
| [docs/architecture.md](docs/architecture.md) | How the pieces fit and the rules they follow |
| [docs/guides.md](docs/guides.md) | Recipes: fetch, store, clean, features, backtest, stream, trade, extend |
| [docs/strategies.md](docs/strategies.md) | How each strategy works, what the metrics mean, changes from the book |
| [docs/configuration.md](docs/configuration.md) | Every setting and its environment variable |

For the full site with the API reference generated from docstrings, run
`mkdocs serve` and open http://127.0.0.1:8000.
