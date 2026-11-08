"""Builders for test data and settings."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config.settings import DataSettings


def make_ohlcv(
    n: int = 100,
    start: str = '2024-01-01',
    freq: str = 'D',
    seed: int = 0,
    timestamps: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """Internally consistent random-walk OHLCV bars with UTC timestamps."""
    rng = np.random.default_rng(seed)
    if timestamps is None:
        timestamps = pd.date_range(start, periods=n, freq=freq, tz='UTC')
    n = len(timestamps)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': rng.uniform(1_000, 2_000, n),
    })


def to_bars(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Provider-shaped bar dicts (ISO timestamps) from an OHLCV frame."""
    bars = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    bars['timestamp'] = bars['timestamp'].map(lambda t: t.isoformat())
    return bars.to_dict('records')


def data_settings(**overrides: Any) -> DataSettings:
    """DataSettings with instant retries and no cache unless overridden."""
    values = {'retry_base_delay': 0.0, 'retry_jitter': False, 'cache_ttl_seconds': 0.0}
    values.update(overrides)
    return DataSettings(**values)


def make_panel(
    n: int = 300,
    symbols: Sequence[str] = ('AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'),
    seed: int = 0,
    drift_step: float = 0.001,
    vol_step: float = 0.0,
) -> pd.DataFrame:
    """
    Close-price panel of random walks. Symbol i drifts by i * drift_step per
    bar and has volatility 0.01 + i * vol_step, so later symbols have the
    strongest momentum (and, with vol_step > 0, the highest volatility).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n, freq='D', tz='UTC', name='timestamp')
    columns = {}
    for i, symbol in enumerate(symbols):
        steps = rng.normal(i * drift_step, 0.01 + i * vol_step, n)
        columns[symbol] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(columns, index=index)


def panel_to_long(panel: pd.DataFrame) -> pd.DataFrame:
    """Long OHLCV-shaped frame (timestamp, symbol, close) from a panel."""
    long = panel.rename_axis('timestamp').stack().rename('close').reset_index()
    return long.rename(columns={'level_1': 'symbol'})


def make_pair(
    n: int = 600,
    hedge_ratio: float = 1.5,
    intercept: float = 0.2,
    phi: float = 0.9,
    seed: int = 0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Cointegrated prices: log A = intercept + hedge_ratio * log B + AR(1)
    noise with persistence `phi`; log B is a random walk.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range('2024-01-01', periods=n, freq='D', tz='UTC', name='timestamp')
    log_b = np.log(50) + np.cumsum(rng.normal(0, 0.02, n))
    noise = np.zeros(n)
    shocks = rng.normal(0, 0.01, n)
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + shocks[t]
    log_a = intercept + hedge_ratio * log_b + noise
    return (
        pd.Series(np.exp(log_a), index=index, name='AAA'),
        pd.Series(np.exp(log_b), index=index, name='BBB'),
    )
