"""
Shared test setup.

Every test runs in an empty temporary directory with trading-related
environment variables removed, so the project's real .env (and its API keys)
is never read, nothing touches the network, and settings reload fresh.
"""

import os

import pytest

from src.config.settings import reset_settings

_ENV_PREFIXES = ('DATA_', 'MODEL_', 'RISK_', 'EXEC_', 'TRADING_', 'ALPACA_', 'CCXT_')


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.upper().startswith(_ENV_PREFIXES):
            monkeypatch.delenv(name)
    reset_settings()
    yield
    reset_settings()
