"""Tests for feature engineering and labels."""

import itertools

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer
from src.features.labels import forward_returns
from src.features.momentum import rsi
from src.features.volatility import garman_klass_variance
from tests.factories import make_ohlcv


@pytest.fixture
def bars():
    return make_ohlcv(400, freq='B')


def test_features_do_not_look_ahead(bars):
    """Changing the last bar must not change any feature on earlier bars."""
    engineer = FeatureEngineer()
    before = engineer.compute_all_features(bars)

    shocked = bars.copy()
    shocked.loc[shocked.index[-1], ['open', 'high', 'low', 'close']] *= 1.5
    shocked.loc[shocked.index[-1], 'volume'] *= 10
    after = engineer.compute_all_features(shocked)

    cols = engineer.feature_names
    pd.testing.assert_frame_equal(before[cols].iloc[:-1], after[cols].iloc[:-1])


def test_no_target_columns_in_features(bars):
    engineer = FeatureEngineer()
    engineer.compute_all_features(bars)
    assert not [c for c in engineer.feature_names if 'forward' in c]


def test_no_duplicate_features(bars):
    engineer = FeatureEngineer()
    out = engineer.compute_all_features(bars)
    for a, b in itertools.combinations(engineer.feature_names, 2):
        assert not np.allclose(out[a], out[b], equal_nan=True), f"{a} duplicates {b}"


def test_volume_features_do_not_depend_on_start_date(bars):
    engineer = FeatureEngineer()
    full = engineer.compute_all_features(bars).set_index('timestamp')
    late_start = engineer.compute_all_features(bars.iloc[100:].reset_index(drop=True)).set_index('timestamp')

    cols = ['obv_ratio_20', 'vwap_deviation_20', 'mfi_14', 'volume_ratio_60']
    common = late_start.index[100:]
    pd.testing.assert_frame_equal(full.loc[common, cols], late_start.loc[common, cols])


def test_garman_klass_matches_textbook_formula():
    bar = pd.DataFrame({'open': [100.0], 'high': [101.0], 'low': [99.0], 'close': [100.0]})
    assert garman_klass_variance(bar).iloc[0] == pytest.approx(0.5 * np.log(101 / 99) ** 2)


def test_annualization_depends_on_market(bars):
    equity = FeatureEngineer(market='equity').compute_all_features(bars)
    crypto = FeatureEngineer(market='crypto').compute_all_features(bars)
    ratio = (crypto['volatility_20'] / equity['volatility_20']).dropna()
    assert ratio.to_numpy() == pytest.approx(np.sqrt(365 / 252))


def test_feature_names_exclude_input_columns(bars):
    engineer = FeatureEngineer()
    out = engineer.compute_all_features(bars)
    assert set(engineer.feature_names).isdisjoint(bars.columns)
    assert list(out.columns) == list(bars.columns) + engineer.feature_names


def test_recomputing_features_replaces_columns(bars):
    engineer = FeatureEngineer()
    once = engineer.compute_all_features(bars)
    twice = engineer.compute_all_features(once)
    pd.testing.assert_frame_equal(once, twice)


@pytest.mark.parametrize('mutate,match', [
    (lambda df: df.drop(columns='high'), 'missing'),
    (lambda df: df.iloc[::-1], 'Sort'),
    (lambda df: df.assign(symbol=['A', 'B'] * (len(df) // 2)), 'one symbol'),
])
def test_invalid_input_raises(bars, mutate, match):
    with pytest.raises(ValueError, match=match):
        FeatureEngineer().compute_all_features(mutate(bars))


def test_rsi_edge_cases():
    rising = pd.Series(np.arange(1.0, 40.0))
    flat = pd.Series(np.full(40, 5.0))
    assert rsi(rising, 14).iloc[-1] == 100.0
    assert rsi(flat, 14).iloc[-1] == 50.0
    assert rsi(rising, 14).iloc[:14].isna().all()


def test_forward_returns():
    df = pd.DataFrame({'close': [100.0, 110.0, 99.0, 99.0]})
    labels = forward_returns(df, horizons=(1, 2))
    assert labels['forward_return_1'].tolist()[:3] == pytest.approx([0.1, -0.1, 0.0])
    assert labels['forward_return_2'].iloc[0] == pytest.approx(-0.01)
    assert labels['forward_return_2'].iloc[-2:].isna().all()
