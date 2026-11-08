"""Tests for settings loading and startup validation."""

import pytest

from src.config.settings import DataSettings, ExecutionSettings, TradingSettings


def test_execution_falls_back_to_alpaca_variables(monkeypatch):
    monkeypatch.setenv('ALPACA_API_KEY', 'alpaca-key')
    monkeypatch.setenv('ALPACA_API_SECRET', 'alpaca-secret')
    monkeypatch.setenv('ALPACA_PAPER', 'false')

    s = ExecutionSettings()
    assert s.api_key.get_secret_value() == 'alpaca-key'
    assert s.api_secret.get_secret_value() == 'alpaca-secret'
    assert s.paper_trading is False


def test_exec_variables_take_precedence(monkeypatch):
    monkeypatch.setenv('ALPACA_API_KEY', 'alpaca-key')
    monkeypatch.setenv('EXEC_API_KEY', 'exec-key')
    assert ExecutionSettings().api_key.get_secret_value() == 'exec-key'


def test_env_file_with_inline_comment(tmp_path):
    (tmp_path / '.env').write_text(
        'ALPACA_API_KEY=key\n'
        'ALPACA_PAPER=false # set to false for live\n'
    )
    assert DataSettings().alpaca_api_key.get_secret_value() == 'key'
    assert ExecutionSettings().paper_trading is False


def test_secrets_are_not_shown_in_repr(monkeypatch):
    monkeypatch.setenv('ALPACA_API_KEY', 'super-secret')
    assert 'super-secret' not in repr(DataSettings())
    assert 'super-secret' not in repr(ExecutionSettings())


def test_validate_startup_requires_keys_for_paper_env(monkeypatch):
    monkeypatch.setenv('TRADING_ENV', 'paper')
    with pytest.raises(ValueError, match='API key required'):
        TradingSettings().validate_startup()

    monkeypatch.setenv('ALPACA_API_KEY', 'key')
    TradingSettings().validate_startup()


def test_validate_startup_rejects_live_env_with_paper_trading(monkeypatch):
    monkeypatch.setenv('TRADING_ENV', 'live')
    monkeypatch.setenv('ALPACA_API_KEY', 'key')
    with pytest.raises(ValueError, match='paper_trading'):
        TradingSettings().validate_startup()


def test_default_settings_are_valid():
    TradingSettings().validate_startup()
