# Strategies

Four strategies adapted from chapter 4 of *Intelligent Markets*, plus the
tools to measure them. Each one takes a **price panel** (one column per
symbol) and returns **target weights**. Signals, backtests and metrics are
all built from those weights.

[![Strategy research backtest](diagrams/strategy-backtest.svg)](diagrams/strategy-backtest.html)

For runnable code see [Backtest a strategy](guides.md#backtest-a-strategy).

## Weights, not orders

Every strategy's `target_weights(prices)` returns a frame shaped like the
price panel:

| Value | Meaning |
|---|---|
| `0.25` | Hold 25% of capital long |
| `-0.25` | Hold 25% of capital short |
| Row's absolute sum `1.0` | Fully invested (gross exposure 1) |
| Row's absolute sum `0.0` | Flat |

- **Leverage isn't the strategy's job.** Scaling exposure up or down (volatility targeting, Kelly sizing) belongs to the risk layer, which comes later.
- **Weights on row *t* use bar *t*'s close.** `portfolio_returns` therefore trades them one bar later, so a strategy never earns the return of the price that triggered it.
- **Entry/exit events come from the weights.** `generate_signals()` produces the `LONG_ENTER` / `LONG_EXIT` / `SHORT_ENTER` / `SHORT_EXIT` rows that `BaseStrategy` promises, emitting one only when a symbol's side changes.

## Cross-sectional momentum

`CrossSectionalMomentum` ranks symbols on their return from `lookback + skip`
bars ago to `skip` bars ago. It buys the top `n_long`, shorts the bottom
`n_short`, rebalances every `rebalance_every` bars and holds in between.

| Parameter | Default | Notes |
|---|---|---|
| `lookback` | 120 | About six months of daily bars |
| `skip` | 21 | Leaves out the last month, which tends to reverse |
| `rebalance_every` | 21 | Monthly on daily bars |
| `n_long`, `n_short` | 10, 10 | Needs at least `n_long + n_short` symbols |

With both sides, each side gets half the capital (dollar neutral). Set
`n_short=0` for a long-only portfolio.

## Pairs trading

`PairsTradingStrategy` trades the spread between two related prices:

```text
log A = intercept + hedge_ratio × log B + spread
```

1. **Check the pair first.** `check_cointegration(a, b)` runs the Engle-Granger test. Use a *formation* period that comes before the bars you trade. A pair chosen on the same data it is tested on will look better than it is. The result also gives the half-life: how many bars a deviation takes to shrink by half.
2. **Build the spread without peeking.** Each bar's hedge ratio comes from a regression over the previous `lookback` bars. The z-score is that bar's residual divided by the window's residual standard deviation.
3. **Hold until the spread comes back.** Each bar, the position follows these rules:

```text
flat ── z ≤ −entry ──► long spread  (buy A, short B) ── |z| ≤ exit ──► flat
     ── z ≥ +entry ──► short spread (short A, buy B) ── |z| ≥ stop ──► flat (stop-loss)
```

| Parameter | Default | Notes |
|---|---|---|
| `lookback` | 100 | Bars in the rolling regression |
| `entry_z` | 2.0 | Open a trade |
| `exit_z` | 0.5 | Take profit |
| `stop_z` | 3.5 | Give up: the relationship may have broken |

Leg weights are `1/(1+|b|)` in A and `−b/(1+|b|)` in B. Because prices are in
logs, the hedge ratio `b` is a dollar ratio.

## Kalman pairs

`KalmanPairsStrategy` trades the same spread, with two differences:

- **How the hedge ratio is estimated.** A Kalman filter (`KalmanHedgeRatio`) treats the intercept and hedge ratio as values that drift slowly, and updates them each bar. There is no window to choose.
- **How the z-score is computed.** It is the filter's prediction error divided by its predicted standard deviation, so it only uses earlier bars.

| Parameter | Default | Notes |
|---|---|---|
| `delta` | 1e-5 | How fast the hedge ratio may drift. Larger means it adapts faster but is noisier |
| `warmup` | 30 | Bars for the starting regression; no output for them |

Entry, exit and stop thresholds work as in pairs trading.

## Multi-factor model

`MultiFactorModel` standardizes each factor across symbols on every bar
(z-score), adds the factors with weights, and trades the top and bottom
names like the momentum strategy.

| Factor | Higher score means |
|---|---|
| `momentum` | Stronger past return (skipping the last month) |
| `low_volatility` | Calmer price over `volatility_window` bars |

Pass your own factor panels, such as value from earnings data or a model's
predictions, through `extra_factors`, and give them a weight:

```python
model = MultiFactorModel(factor_weights={'momentum': 0.4, 'low_volatility': 0.3, 'value': 0.3})
weights = model.target_weights(prices, extra_factors={'value': value_panel})
```

## Measuring results

`PerformanceAnalyzer.for_interval(returns, interval, market)` annualizes with
the right number of bars per year. `summary()` returns:

| Metric | Meaning |
|---|---|
| `total_return`, `annual_return` | Compounded growth; annual is CAGR |
| `annual_volatility` | Standard deviation of returns, annualized |
| `sharpe_ratio` | Return per unit of volatility, annualized |
| `sortino_ratio` | Like Sharpe, but only losses count as risk |
| `max_drawdown`, `calmar_ratio` | Worst fall from a peak (negative), and annual return divided by it |
| `win_rate`, `average_win`, `average_loss`, `profit_factor` | Per-bar statistics; flat bars are ignored |
| `omega_ratio` | Gains above the risk-free rate divided by losses below it |
| `probabilistic_sharpe` | Probability that the true Sharpe is above 0, given track length, skew and fat tails |
| `deflated_sharpe` | The same probability, after accounting for the `n_trials` variants you tried |

**Always pass `n_trials`.** If you tried 50 parameter sets and kept the best,
call `summary(n_trials=50)`. With enough tries a good-looking Sharpe appears
by luck. A deflated Sharpe below about 0.95 means the result could be one of
those flukes.

## What changed from the book

| Book code | Problem | Here |
|---|---|---|
| `MomentumStrategy` | Weight frame filled with `0.0` before `ffill()`, so positions existed only on rebalance days | Holds between rebalances |
| `PairsTradingStrategy` | Reads `self.stop_loss_threshold`; the attribute is `stop_loss_thresh` (AttributeError) | Stop-loss works |
| `PairsTradingStrategy` | Stateless signals: a trade lasted only while \|z\| stayed above entry | Positions held until exit or stop |
| `PairsTradingStrategy` | Hedge ratio from the full sample (look-ahead) | Rolling regression on past bars only |
| `test_cointegration` | ADF p-values on regression residuals overstate significance | statsmodels Engle-Granger test |
| `KalmanFilterPairs` | Observation noise fixed at 1.0 (only right for one price scale); no intercept; z-score uses the updated hedge ratio | Noise estimated from warm-up; intercept included; prediction-error z-score |
| `MultiFactorModel` | Latest date only; missing `Optional` import | Every bar, so it can be backtested |
| `PerformanceAnalyzer` | Sortino mixed annual and per-period figures; deflated Sharpe mixed annual and per-observation units | Consistent per-period formulas |

## Not adapted

- **Value factor:** needs earnings data, which the data layer doesn't collect. Add it through `extra_factors` once it does.
- **Order-flow imbalance:** needs tick-level bid and ask volume. The quote stream exists (`src/market_data/streaming`), but nothing aggregates it yet.
