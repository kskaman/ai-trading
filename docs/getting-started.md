# Getting started

All commands run from the project root in PowerShell.

## 1. Install

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

Activate the environment in each new terminal with `.venv\Scripts\Activate.ps1`
(if PowerShell blocks it, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once). In VS Code, run **Python: Select Interpreter** and pick `.venv`.

## 2. Add API keys

Copy `.env.example` to `.env` and fill in your Alpaca keys. Crypto price data
needs no keys. Every setting is listed in [Configuration](configuration.md).

```ini
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
ALPACA_PAPER=true
```

## 3. Run things

| What | Command |
|---|---|
| Run the tests (no network, never reads your `.env`) | `python -m pytest` |
| Fetch AAPL and BTC/USDT, fill gaps, clean (saves nothing) | `python -m src.market_data.ingest` |
| Fetch AAPL/MSFT/GOOGL into `data\market_data.duckdb`, export Parquet | `python -m src.market_data.store` |
| Feature demo on made-up data | `python -m src.features.engineer` |
| Browse these docs with the API reference | `mkdocs serve`, then open http://127.0.0.1:8000 |

Always use `python -m src...` from the project root. Running a file by path
(`python src\market_data\ingest.py`) fails with `No module named 'src'`, because the
code imports itself as `src.…`.

## 4. Write your own script

Save as `scripts\fetch_data.py` and run with `python -m scripts.fetch_data`:

```python
import logging

from src.market_data.ingest import DataIngestionService
from src.market_data.store import FinancialDataStore
from src.features.engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO)  # shows gap and cleaning summaries

with FinancialDataStore() as store:  # data\market_data.duckdb
    service = DataIngestionService.from_settings(store=store)
    df = service.ingest('AAPL', start_date='2024-01-01', end_date='2025-01-01',
                        fill_gaps=True, clean=True)

features = FeatureEngineer(interval='1d', market='equity').compute_all_features(df)
print(features.tail())
```

For crypto, pass a pair such as `'BTC/USDT'` and use `market='crypto'`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'src'` | Run with `python -m ...` from the project root. |
| VS Code says packages are not installed | Select the `.venv` interpreter. |
| `Alpaca provider disabled: ALPACA_API_KEY/ALPACA_API_SECRET not set` | Add the keys to `.env` in the project root. |
| `All data sources failed for ...` | The log lines above it name each source and its error. |
| `CircuitBreakerOpen` | A provider failed repeatedly; calls resume after `DATA_CIRCUIT_BREAKER_TIMEOUT_SECONDS`. |
