"""Cointegration, the pairs state machine, rolling and Kalman hedge ratios."""

import math

import numpy as np
import pandas as pd
import pytest

from src.strategies.kalman_pairs import KalmanHedgeRatio, KalmanPairsStrategy
from src.strategies.pairs_trading import (
    PairsTradingStrategy,
    _next_position,
    check_cointegration,
    half_life,
    hold_positions,
    rolling_spread,
)
from tests.factories import make_pair, panel_to_long


def pair_panel(**kwargs) -> pd.DataFrame:
    a, b = make_pair(**kwargs)
    return pd.concat([a, b], axis=1)


# --- cointegration --------------------------------------------------------

def test_cointegrated_pair_is_detected_with_its_hedge_ratio():
    a, b = make_pair(hedge_ratio=1.5, phi=0.9)

    result = check_cointegration(a, b)

    assert result.is_cointegrated and result.p_value < 0.01
    assert result.t_statistic < result.critical_value_5pct
    assert result.hedge_ratio == pytest.approx(1.5, abs=0.05)
    assert result.half_life == pytest.approx(-math.log(2) / math.log(0.9), rel=0.5)
    assert result.n_obs == 600


def test_independent_random_walks_are_not_cointegrated():
    a, _ = make_pair(seed=1)
    _, b = make_pair(seed=2)
    assert not check_cointegration(a, b).is_cointegrated


def test_cointegration_needs_enough_aligned_positive_prices():
    a, b = make_pair(n=100)
    with pytest.raises(ValueError, match='60'):
        check_cointegration(a.iloc[:59], b)
    bad = a.copy()
    bad.iloc[3] = 0.0
    with pytest.raises(ValueError, match='positive'):
        check_cointegration(bad, b)


def test_half_life_is_infinite_for_a_trending_spread():
    assert half_life(np.arange(50, dtype=float) ** 1.5) == math.inf


# --- rolling spread -------------------------------------------------------

def test_rolling_spread_uses_only_past_bars():
    a, b = make_pair(n=200)
    full = rolling_spread(a, b, window=50)
    cut = rolling_spread(a.iloc[:120], b.iloc[:120], window=50)

    pd.testing.assert_frame_equal(full.iloc[:120], cut)
    assert full.iloc[:50].isna().all().all() and full.iloc[50:].notna().all().all()


def test_rolling_spread_recovers_the_hedge_ratio():
    a, b = make_pair(n=400, hedge_ratio=1.5, phi=0.5)
    frame = rolling_spread(a, b, window=150)
    assert frame['hedge_ratio'].iloc[-1] == pytest.approx(1.5, abs=0.15)
    assert 0.5 < frame['z_score'].std() < 2.0


# --- state machine --------------------------------------------------------

@pytest.mark.parametrize('position,z,expected', [
    (0, 0.0, 0), (0, 1.9, 0), (0, -1.9, 0),   # inside the band: stay flat
    (0, -2.0, 1), (0, -2.5, 1),               # spread low: long A, short B
    (0, 2.0, -1), (0, 2.5, -1),               # spread high: short A, long B
    (0, -3.5, 0), (0, 4.0, 0),                # already past the stop: never open
    (1, -1.0, 1), (-1, 1.0, -1),              # between exit and entry: hold
    (1, -0.5, 0), (1, 0.0, 0), (-1, 0.3, 0),  # back near zero: take profit
    (1, -3.5, 0), (-1, 3.6, 0),               # kept diverging: stop out
])
def test_next_position_transitions(position, z, expected):
    assert _next_position(position, z, entry=2.0, exit=0.5, stop=3.5) == expected


def test_hold_positions_keeps_trades_until_exit_or_stop():
    z = pd.Series([np.nan, 0.0, -2.5, -1.5, np.nan, -1.0, -0.4, 0.0, 2.2, 1.0, 3.8, 1.0])
    expected = [0, 0, 1, 1, 1, 1, 0, 0, -1, -1, 0, 0]
    assert hold_positions(z).tolist() == expected


def test_hold_positions_rejects_unordered_thresholds():
    with pytest.raises(ValueError):
        hold_positions(pd.Series([0.0]), entry=2.0, exit=2.5, stop=3.5)


# --- strategies -----------------------------------------------------------

def test_pairs_strategy_weights_are_hedged_and_fully_invested():
    prices = pair_panel(n=500, phi=0.95)
    strategy = PairsTradingStrategy('AAA', 'BBB', lookback=60)

    weights = strategy.target_weights(prices)
    invested = weights[weights.abs().sum(axis=1) > 0]

    assert len(invested) > 0
    assert invested.abs().sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert (np.sign(invested['AAA']) == -np.sign(invested['BBB'])).all()
    assert (weights.iloc[:61] == 0).all().all()


def test_pairs_strategy_signals_cover_both_legs():
    prices = pair_panel(n=500, phi=0.95)
    signals = PairsTradingStrategy('AAA', 'BBB', lookback=60).generate_signals(
        panel_to_long(prices)
    )
    assert set(signals['symbol']) == {'AAA', 'BBB'}
    enters = signals[signals['signal_type'].str.endswith('ENTER')]
    assert enters.groupby('timestamp')['symbol'].nunique().eq(2).all()


def test_pairs_strategy_validates_parameters():
    with pytest.raises(ValueError):
        PairsTradingStrategy('AAA', 'AAA')
    with pytest.raises(ValueError):
        PairsTradingStrategy('AAA', 'BBB', entry_z=4.0, stop_z=3.5)


# --- Kalman ---------------------------------------------------------------

def test_kalman_tracks_the_hedge_ratio():
    a, b = make_pair(n=1000, hedge_ratio=1.5, phi=0.5)
    out = KalmanHedgeRatio(delta=1e-5, warmup=30).filter(a, b)

    assert out.iloc[:30].isna().all().all()
    assert out['hedge_ratio'].iloc[-1] == pytest.approx(1.5, abs=0.15)
    assert 0.8 < out['z_score'].iloc[100:].std() < 1.25  # innovations ~ N(0, 1)
    fitted = out['intercept'] + out['hedge_ratio'] * np.log(b)
    assert (np.log(a) - fitted).iloc[-1] == pytest.approx(
        out['spread'].iloc[-1], abs=0.05)  # intercept is for uncentred prices


def test_kalman_calibration_does_not_depend_on_price_level():
    a, b = make_pair(n=600)
    cheap = KalmanHedgeRatio().filter(a, b)['z_score']
    pricey = KalmanHedgeRatio().filter(a * 1000, b * 1000)['z_score']
    pd.testing.assert_series_equal(cheap, pricey, atol=1e-6)


def test_kalman_uses_only_past_bars():
    a, b = make_pair(n=300)
    kalman = KalmanHedgeRatio()
    pd.testing.assert_frame_equal(
        kalman.filter(a, b).iloc[:200], kalman.filter(a.iloc[:200], b.iloc[:200])
    )


def test_kalman_keeps_missing_bars_aligned():
    a, b = make_pair(n=200)
    a = a.drop(a.index[100])

    out = KalmanHedgeRatio().filter(a, b)

    assert out.index.equals(b.index)
    assert out.iloc[100].isna().all() and out.iloc[101].notna().all()


def test_kalman_rejects_bad_parameters():
    with pytest.raises(ValueError):
        KalmanHedgeRatio(delta=0)
    with pytest.raises(ValueError):
        KalmanHedgeRatio(warmup=2)
    a, b = make_pair(n=20)
    with pytest.raises(ValueError, match='30'):
        KalmanHedgeRatio(warmup=30).filter(a, b)


def test_kalman_strategy_produces_hedged_weights():
    prices = pair_panel(n=500, phi=0.95)
    strategy = KalmanPairsStrategy('AAA', 'BBB')

    weights = strategy.target_weights(prices)
    invested = weights[weights.abs().sum(axis=1) > 0]

    assert len(invested) > 0
    assert invested.abs().sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert strategy.get_parameters()['delta'] == 1e-5
