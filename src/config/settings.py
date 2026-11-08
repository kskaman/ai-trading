"""

Configuration management for the trading framework.

This module establishes the centralized configuration system used
throughout the entire codebase. All components (data, models, strategies,
execution, monitoring) read their settings from this single source.


Design decisions:

1. We use Pydantic's BaseSettings, which automatically loads from
environment variables and .env files. Values are type-checked when the
settings are first loaded, catching misconfiguration early.

2. Settings are organized into nested classes (DataSettings, ModelSettings,
RiskSettings) that mirror the module structure. Each component imports
only the settings it needs, creating clear dependency boundaries.
 
3. We provide three configuration profiles (default, paper, live) loaded
via an environment variable TRADING_ENV. This eliminates the risk of
accidentally running live configuration in development.
 
4. Secrets (API keys, database passwords) are never stored in configuration
files. They are loaded from environment variables set by the secrets
manager (AWS Secrets Manager, HashiCorp Vault, or a .env file for
local development only).

Common pitfalls:
 
- Forgetting to validate settings at startup can cause cryptic errors
deep in execution (e.g., a missing API key discovered only when the
first data request fails). Always call TradingSettings.validate_startup()
during application initialization.

- Storing default values for secrets in code (even as empty strings)
creates a false sense of security. Use Optional[str] with explicit
None defaults to force configuration at deployment time.

- Loading all settings into a single monolithic class makes it harder
to test individual components in isolation. Nested classes with
clear boundaries enable unit testing without full system initialization.
"""

from typing import Optional
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    """Configuration for data acquisition and storage."""

    # Fields with an alias read that exact variable name (no DATA_ prefix);
    # every other field is read as DATA_<FIELD>.
    model_config = SettingsConfigDict(
        env_prefix='DATA_',
        env_file='.env',
        extra='ignore',
        populate_by_name=True,
    )

    # Data source configuration
    primary_source: str = Field(
        default='alpaca',
        description="Provider for equity symbols. Built in: 'alpaca'"
    )

    backup_source: str = Field(
        default='yfinance',
        description=(
            "Equity provider tried when the primary fails. No 'yfinance' "
            "provider exists yet, so failover is skipped with a warning"
        )
    )

    crypto_source: str = Field(
        default='ccxt',
        description="Source used for pair symbols such as 'BTC/USDT'"
    )

    # Storage configuration
    parquet_base_dir: str = Field(
        default='./data/parquet',
        description="Base directory for Parquet file storage"
    )
    
    duckdb_path: str = Field(
        default='./data/market_data.duckdb',
        description="Path to DuckDB database file"
    )

    partition_by: str = Field(
        default='month',
        description="Default Parquet file split for exports: 'day', 'month' or 'year'"
    )

    # Ingestion parameters
    max_retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts per failed API operation"
    )

    retry_base_delay: float = Field(
        default=1.0,
        ge=0,
        allow_inf_nan=False,
        description="Base delay in seconds for exponential backoff"
    )

    retry_max_delay: float = Field(
        default=30.0,
        ge=0,
        allow_inf_nan=False,
        description="Maximum delay in seconds between retry attempts"
    )

    retry_jitter: bool = Field(
        default=True,
        description="Randomize retry backoff to avoid synchronized retries"
    )

    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive fully failed operations before opening circuit breaker"
    )

    circuit_breaker_timeout_seconds: float = Field(
        default=30.0,
        ge=0,
        allow_inf_nan=False,
        description="Seconds the circuit stays open before a probe request"
    )

    cache_ttl_seconds: float = Field(
        default=60.0,
        ge=0,
        allow_inf_nan=False,
        description="Seconds a cached API response stays valid (0 disables caching)"
    )

    # Alpaca (equities); credentials come from the environment or .env only
    alpaca_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices('ALPACA_API_KEY', 'DATA_ALPACA_API_KEY'),
        description="Alpaca API key"
    )
    alpaca_api_secret: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices('ALPACA_API_SECRET', 'DATA_ALPACA_API_SECRET'),
        description="Alpaca API secret"
    )
    alpaca_ws_url: str = Field(
        default='wss://stream.data.alpaca.markets/v2/iex',
        description="Alpaca real-time market data WebSocket URL"
    )

    # Alpaca WebSocket feed
    ws_max_reconnect_attempts: int = Field(
        default=10,
        ge=1,
        description="Consecutive failed reconnects before the feed gives up"
    )
    ws_base_reconnect_delay: float = Field(
        default=2.0,
        ge=0,
        allow_inf_nan=False,
        description="Base delay in seconds for reconnect backoff"
    )
    ws_max_reconnect_delay: float = Field(
        default=60.0,
        ge=0,
        allow_inf_nan=False,
        description="Upper bound in seconds for reconnect backoff"
    )
    ws_ping_interval_seconds: float = Field(
        default=20.0,
        gt=0,
        allow_inf_nan=False,
        description="WebSocket ping interval used to detect stale connections"
    )
    ws_stable_after_seconds: float = Field(
        default=60.0,
        ge=0,
        allow_inf_nan=False,
        description=(
            "A connection that stays up this long resets the reconnect counter; "
            "shorter-lived connections count as failures"
        )
    )
    ws_auth_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        allow_inf_nan=False,
        description="Seconds to wait for the server greeting and auth reply"
    )

    # CCXT (crypto)
    ccxt_exchange_id: str = Field(
        default='binance',
        description=(
            "CCXT exchange: 'binance', 'coinbase', 'kraken', 'bybit', 'okx', "
            "'kucoin' or any other CCXT exchange id"
        )
    )
    ccxt_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices('CCXT_API_KEY', 'DATA_CCXT_API_KEY'),
        description="Exchange API key (private endpoints only)"
    )
    ccxt_api_secret: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices('CCXT_API_SECRET', 'DATA_CCXT_API_SECRET'),
        description="Exchange API secret (private endpoints only)"
    )
    ccxt_sandbox: bool = Field(
        default=False,
        description=(
            "Read market data from the exchange testnet. Testnet candles are "
            "NOT real market prices; leave off for research data. Order "
            "placement uses ExecutionSettings.paper_trading instead."
        )
    )
    ccxt_default_limit: int = Field(
        default=100,
        ge=1,
        description="Candles returned when no start date is given"
    )
    ccxt_page_limit: int = Field(
        default=1000,
        ge=1,
        description=(
            "Candles requested per call when paging a date range. Exchanges "
            "may return fewer (Kraken 720, Coinbase/OKX 300); paging handles that."
        )
    )

class ModelSettings(BaseSettings):
    """Configuration for ML model training and inference."""

    model_config = SettingsConfigDict(env_prefix='MODEL_', env_file='.env', extra='ignore')
    
    # Training parameters
    walk_forward_train_size: int = Field(
        default=504,
        description="Training window size in trading days (2 years)"
    )
    walk_forward_test_size: int = Field(
        default=63,
        description="Test window size in trading days (quarter)"
    )
    purge_size: int = Field(
        default=5,
        description="Observations to purge near train/test boundary"
    )
    embargo_size: int = Field(
        default=5,
        description="Test observations to ignore (label uncertainty)"
    )

    # Model hyperparameters (XGBoost defaults)
    n_estimators: int = Field(
        default=200,
        description="Number of boosting rounds"
    )
    max_depth: int = Field(
        default=4,
        description="Maximum tree depth"
    )
    learning_rate: float = Field(
        default=0.05,
        description="Step size shrinkage for each tree"
    )
    subsample: float = Field(
        default=0.8,
        description="Fraction of samples used per boosting round"
    )
    min_child_weight: int = Field(
        default=20,
        description="Minimum sum of instance weight in a leaf"
    )

    # MLflow configuration
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking server URI (None for file store)"
    )
    model_registry_name: str = Field(
        default='trading_alpha_model',
        description="Name for the model in MLflow registry"
    )


class RiskSettings(BaseSettings):
    """Configuration for risk management limits."""

    model_config = SettingsConfigDict(env_prefix='RISK_', env_file='.env', extra='ignore')
    
    # Position limits
    max_position_pct: float = Field(
        default=0.10,
        description="Maximum single position as fraction of portfolio"
    )
    max_gross_exposure: float = Field(
        default=2.0,
        description="Maximum gross exposure (sum of abs weights)"
    )
    max_net_exposure: float = Field(
        default=0.20,
        description="Maximum net exposure (long minus short)"
    )

    # Loss limits
    daily_loss_limit_pct: float = Field(
        default=0.02,
        description="Maximum daily loss as fraction of equity"
    )
    max_drawdown_pct: float = Field(
        default=0.10,
        description="Maximum drawdown from peak before circuit breaker"
    )

    # Transaction cost assumptions
    commission_per_share: float = Field(
        default=0.005,
        description="Commission per share in dollars"
    )
    slippage_bps: int = Field(
        default=5,
        description="Estimated slippage in basis points"
    )

class ExecutionSettings(BaseSettings):
    """Configuration for order execution and broker integration."""

    # EXEC_* variables win; otherwise the ALPACA_* variables are reused, so the
    # same keys are not configured twice.
    model_config = SettingsConfigDict(
        env_prefix='EXEC_',
        env_file='.env',
        extra='ignore',
        populate_by_name=True,
    )

    # Broker configuration
    broker: str = Field(
        default='alpaca',
        description="Broker API: 'alpaca' (not used yet; brokers are chosen in code)"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices('EXEC_API_KEY', 'ALPACA_API_KEY'),
        description="Alpaca trading API key"
    )
    api_secret: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices('EXEC_API_SECRET', 'ALPACA_API_SECRET'),
        description="Alpaca trading API secret"
    )

    # Execution parameters
    paper_trading: bool = Field(
        default=True,
        validation_alias=AliasChoices('EXEC_PAPER_TRADING', 'ALPACA_PAPER'),
        description="Use the Alpaca paper trading account instead of the live one"
    )
    ccxt_testnet: bool = Field(
        default=True,
        description=(
            "Send CCXT orders to the exchange testnet. Separate from paper_trading "
            "so switching Alpaca to live never makes crypto orders live. Orders "
            "are refused when the exchange has no testnet."
        )
    )
    max_orders_per_minute: int = Field(
        default=60,
        description="Order rate limit (not enforced yet)"
    )
    order_timeout_seconds: int = Field(
        default=30,
        description="Timeout for order confirmation from broker (not used yet)"
    )

class TradingSettings(BaseSettings):
    """
    Top-level trading system configuration.

    This is the main entry point for all settings. Components should
    import this class and access nested settings as needed:
    
        settings = TradingSettings()
        data_config = settings.data
        risk_config = settings.risk
    """

    model_config = SettingsConfigDict(env_prefix='TRADING_', env_file='.env', extra='ignore')
    
    # Environment profile
    env: str = Field(
        default='default',
        description="Configuration profile: 'default', 'paper', 'live'"
    )

    # Nested settings
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)

    def validate_startup(self) -> None:
        """
        Validate cross-field rules at startup.

        Call this during application initialization to catch misconfiguration
        before any trading activity begins. (Named validate_startup so it does
        not shadow pydantic's BaseModel.validate.)

        Raises:
            ValueError: If any required setting is missing or invalid.
        """

        if self.env == 'live' and self.execution.paper_trading:
            raise ValueError(
                "Cannot use paper_trading=True with env='live'. "
                "Set TRADING_ENV=paper for paper trading."
            )

        api_key = self.execution.api_key
        if self.env in ('paper', 'live') and (
            api_key is None or not api_key.get_secret_value()
        ):
            raise ValueError(
                f"Broker API key required for env='{self.env}'. "
                "Set ALPACA_API_KEY (or EXEC_API_KEY) in the environment or .env."
            )

        if self.risk.daily_loss_limit_pct <= 0:
            raise ValueError("daily_loss_limit_pct must be positive")

        if self.risk.max_position_pct > 1.0:
            raise ValueError("max_position_pct cannot exceed 1.0 (100%)")

# Singleton instance (lazy-loaded)
_settings_instance: Optional[TradingSettings] = None

def get_settings() -> TradingSettings:
    """
    Get the global settings instance.

    Returns a cached instance to avoid re-parsing environment variables
    on every call. Call reset_settings() to force a reload (useful in
    testing scenarios).
    """

    global _settings_instance
    
    if _settings_instance is None:
        _settings_instance = TradingSettings()

    return _settings_instance

def reset_settings() -> None:
    """Reset the settings cache (for testing)."""
    global _settings_instance
    _settings_instance = None
