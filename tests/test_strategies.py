"""Price panels, weight helpers, cross-sectional momentum and the factor model."""

import numpy as np
import pandas as pd
import pytest

from src.market_data.panel import load_panel, to_panel
from src.market_data.store import FinancialDataStore
from src.strategies.factors import MultiFactorModel, combine_factors
from src.strategies.momentum import CrossSectionalMomentum
from src.strategies.weights import (
    SIGNAL_COLUMNS,
    cross_sectional_zscore,
    long_short_weights,
    weights_to_signals,
)
from tests.factories import make_ohlcv, make_panel, panel_to_long


# --- panels ---------------------------------------------------------------

def test_to_panel_pivots_and_leaves_gaps_as_nan():
    panel = make_panel(5, symbols=('BBB', 'AAA'))
    long = panel_to_long(panel)
    long = long[~((long['symbol'] == 'AAA') & (long['timestamp'] == panel.index[2]))]

    result = to_panel(long.sample(frac=1, random_state=0))

    assert list(result.columns) == ['AAA', 'BBB']
    assert result.index.is_monotonic_increasing and result.index.name == 'timestamp'
    assert np.isnan(result['AAA'].iloc[2])
    assert result['BBB'].iloc[2] == pytest.approx(panel['BBB'].iloc[2])


def test_to_panel_rejects_duplicates_and_missing_columns():
    long = panel_to_long(make_panel(3, symbols=('AAA',)))
    with pytest.raises(ValueError, match='Duplicate'):
        to_panel(pd.concat([long, long.iloc[:1]]))
    with pytest.raises(ValueError, match='columns'):
        to_panel(long.drop(columns='symbol'))


def test_load_panel_reads_the_store(tmp_path):
    with FinancialDataStore(':memory:', str(tmp_path / 'pq')) as store:
        store.ingest_ohlcv('AAA', '1d', make_ohlcv(10, seed=1))
        store.ingest_ohlcv('BBB', '1d', make_ohlcv(8, seed=2))

        panel = load_panel(store, ['AAA', 'BBB', 'NONE'], start_date='2024-01-03')

        assert list(panel.columns) == ['AAA', 'BBB']
        assert len(panel) == 8 and panel['BBB'].isna().sum() == 2
        with pytest.raises(ValueError, match='field'):
            load_panel(store, ['AAA'], field='symbol')
        with pytest.raises(ValueError, match='No stored bars'):
            load_panel(store, ['NONE'])


# --- weight helpers -------------------------------------------------------

def test_cross_sectional_zscore_standardizes_each_row():
    frame = pd.DataFrame({'A': [1.0, 5.0], 'B': [2.0, np.nan], 'C': [3.0, np.nan]})
    z = cross_sectional_zscore(frame)
    assert z.iloc[0].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert z.iloc[1].isna().all()  # one value: no spread to standardize by


def test_long_short_weights_hold_between_rebalances():
    """The book filled with 0.0 before ffill, so holdings vanished after day one."""
    scores = make_panel(30).pct_change(5)
    weights = long_short_weights(scores, n_long=2, n_short=2, rebalance_every=10)

    first = weights.index[5]  # pct_change(5) leaves 5 NaN rows
    assert (weights.loc[:weights.index[4]] == 0).all().all()
    held = weights.iloc[5:15]
    assert held.abs().sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert (held.eq(weights.loc[first], axis=1)).all().all()
    assert weights.sum(axis=1).iloc[5:].abs().max() < 1e-12  # dollar neutral


def test_long_short_weights_one_sided_book_is_fully_invested():
    scores = make_panel(10).pct_change()
    weights = long_short_weights(scores, n_long=3, n_short=0, rebalance_every=1)
    assert weights.iloc[-1].sum() == pytest.approx(1.0)
    assert (weights.iloc[-1] >= 0).all()


@pytest.mark.parametrize('n_long,n_short,every', [(0, 0, 1), (-1, 2, 1), (2, 2, 0), (4, 4, 1)])
def test_long_short_weights_rejects_bad_parameters(n_long, n_short, every):
    with pytest.raises(ValueError):
        long_short_weights(make_panel(5), n_long, n_short, every)


def test_weights_to_signals_marks_side_changes_only():
    index = pd.date_range('2024-01-01', periods=4, freq='D', tz='UTC')
    weights = pd.DataFrame(
        {'A': [0.0, 0.5, 0.25, -0.5], 'B': [0.0, -0.5, -0.75, 0.0]}, index=index
    )

    signals = weights_to_signals(weights)

    assert list(signals.columns) == list(SIGNAL_COLUMNS)
    rows = list(signals[['symbol', 'signal_type', 'strength']].itertuples(index=False, name=None))
    assert rows == [
        ('A', 'LONG_ENTER', 0.5), ('B', 'SHORT_ENTER', 0.5),   # day 2: open
        # day 3: resizes only, nothing emitted
        ('A', 'LONG_EXIT', 0.25), ('B', 'SHORT_EXIT', 0.75),   # day 4: exits first
        ('A', 'SHORT_ENTER', 0.5),
    ]
    assert signals['timestamp'].tolist() == [index[1]] * 2 + [index[3]] * 3


def test_weights_to_signals_empty_when_always_flat():
    signals = weights_to_signals(pd.DataFrame({'A': [0.0, 0.0]}))
    assert signals.empty and list(signals.columns) == list(SIGNAL_COLUMNS)


# --- momentum -------------------------------------------------------------

def test_momentum_buys_winners_and_shorts_losers():
    prices = make_panel(200, drift_step=0.005)  # FFF strongest, AAA weakest
    strategy = CrossSectionalMomentum(lookback=60, skip=5, rebalance_every=20, n_long=2, n_short=2)

    weights = strategy.target_weights(prices)

    last = weights.iloc[-1]
    assert set(last[last > 0].index) == {'EEE', 'FFF'}
    assert set(last[last < 0].index) == {'AAA', 'BBB'}
    assert last.abs().sum() == pytest.approx(1.0) and last.sum() == pytest.approx(0.0)
    assert (weights.iloc[:65] == 0).all().all()  # needs lookback + skip bars first


def test_momentum_scores_ignore_the_skipped_bars():
    prices = make_panel(100)
    strategy = CrossSectionalMomentum(lookback=20, skip=5, n_long=1, n_short=1)
    shocked = prices.copy()
    shocked.iloc[-5:] *= 3  # only the skipped window changes

    pd.testing.assert_series_equal(
        strategy.scores(prices).iloc[-1], strategy.scores(shocked).iloc[-1]
    )


def test_momentum_generate_signals_from_long_data():
    prices = make_panel(120, drift_step=0.002)
    strategy = CrossSectionalMomentum(lookback=30, skip=5, rebalance_every=200, n_long=1, n_short=1)

    signals = strategy.generate_signals(panel_to_long(prices))

    assert signals[['symbol', 'signal_type']].values.tolist() == [
        ['AAA', 'SHORT_ENTER'], ['FFF', 'LONG_ENTER'],
    ]
    assert signals['timestamp'].iloc[0] == prices.index[35]
    assert strategy.get_parameters()['lookback'] == 30


def test_momentum_validates_parameters():
    with pytest.raises(ValueError):
        CrossSectionalMomentum(lookback=0)
    with pytest.raises(ValueError):
        CrossSectionalMomentum(n_long=0, n_short=0)
    with pytest.raises(ValueError, match='at least'):
        CrossSectionalMomentum(n_long=5, n_short=5).target_weights(make_panel(50))


# --- multi-factor ---------------------------------------------------------

def test_low_volatility_factor_prefers_calm_symbols():
    prices = make_panel(200, drift_step=0.0, vol_step=0.005)  # AAA calmest
    model = MultiFactorModel(volatility_window=60, n_long=1, n_short=1)

    factor = model.low_volatility_factor(prices).iloc[-1]

    assert factor.idxmax() == 'AAA' and factor.idxmin() == 'FFF'
    assert factor.mean() == pytest.approx(0.0, abs=1e-12)


def test_factor_model_combines_weighted_factors():
    prices = make_panel(200, drift_step=0.005)
    model = MultiFactorModel(factor_weights={'momentum': 1.0}, n_long=2, n_short=2,
                             momentum_lookback=60, momentum_skip=5)

    pd.testing.assert_frame_equal(model.composite_score(prices), model.momentum_factor(prices))
    last = model.target_weights(prices).iloc[-1]
    assert set(last[last > 0].index) == {'EEE', 'FFF'}


def test_factor_model_accepts_extra_factors():
    prices = make_panel(100)
    value = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    value['CCC'] = 5.0  # cheapest by far
    model = MultiFactorModel(factor_weights={'value': 1.0}, n_long=1, n_short=0)

    weights = model.target_weights(prices, extra_factors={'value': value})

    assert weights.iloc[0].to_dict() == {s: (1.0 if s == 'CCC' else 0.0) for s in prices}


def test_combine_factors_requires_every_weighted_factor():
    panel = make_panel(5)
    with pytest.raises(ValueError, match='value'):
        combine_factors({'momentum': panel}, {'momentum': 0.5, 'value': 0.5})
    with pytest.raises(ValueError):
        combine_factors({'momentum': panel}, {})
    gapped = panel.copy()
    gapped.iloc[0, 0] = np.nan
    combined = combine_factors({'a': panel, 'b': gapped}, {'a': 1.0, 'b': 1.0})
    assert np.isnan(combined.iloc[0, 0]) and not combined.iloc[1].isna().any()
