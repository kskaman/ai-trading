# Concerns

| # | Where | Status |
|---|---|---|
| 1 | `schema.periods_per_year` | Fixed |
| 2 | `schema.ohlc_valid_mask` | Fixed |
| 3 | `portfolio.portfolio_returns` | Known limit, not fixed |
| 4 | `weights.weights_to_signals` | By design |

---

## 1. Wrong number of bars per year for intraday stock data

**File:** `src/market_data/schema.py`
**Function:** `periods_per_year(interval, market)`

### Background

Volatility (how much a price jumps around) is first measured per bar, for
example per hour. To compare it across intervals we convert it to a yearly
figure. That conversion needs one number: **how many bars are in a year**.
If that number is wrong, every yearly volatility figure is wrong too.

### The issue

A US stock trading day runs 09:30 to 16:00, which is 390 minutes. The
function divided 390 by the bar length:

| Bar size | 390 ÷ minutes | Bars you really get in a day |
|---|---|---|
| 1 hour | 6.5 | 7 (09:30–10:00 is still a bar) |
| 4 hours | 1.625 | 2 |

You can't have half a bar, so the count was too low.

Even the corrected count is only an assumption. Real data can differ:

- **Extended hours:** some data includes trading from 04:00 to 20:00, which
  gives up to 16 hourly bars a day instead of 7.
- **Short days:** the market closes early before some holidays.
- **Missing bars:** an hour with no trades may have no bar at all.

### What was done

- The function now counts whole bars (1 hour → 7 a day, 4 hours → 2).
- New function `estimate_periods_per_year(timestamps, market)` counts the
  bars your data really has per day.
- `FeatureEngineer` accepts that measured number as `periods_per_year`.

Daily data was not affected.

---

## 2. Infinite prices were accepted as valid

**File:** `src/market_data/schema.py`
**Function:** `ohlc_valid_mask(df)`

### Background

This function checks that each bar makes sense: prices above zero, the
high is the highest price, the low is the lowest, and volume is not
negative. Bars that fail get flagged so you can leave them out.

### The issue

Bad data can contain "infinity" instead of a number. Infinity is bigger
than everything, so it passes every "is the high at least…" check. A bar
whose prices were all infinity was marked **valid**.

### What was done

The function now also requires every price and the volume to be a normal,
finite number. Infinity and missing values both fail.

---

## 3. The backtest ignores weight drift

**File:** `src/analytics/portfolio.py`
**Function:** `portfolio_returns(weights, prices)`

### Background

A strategy outputs **target weights**: what share of your money it wants in
each stock, e.g. 10% in AAPL. What you actually own are **holdings**: a
number of shares. The two start equal, then prices move.

### The issue

Say you put 10% in AAPL and AAPL rises while everything else stays flat.
AAPL is now more than 10% of your portfolio. This is called **drift**.

The backtest assumes you are always back at exactly 10%, every single bar.
In real life that would mean trading a little every day, and those trades
cost money. So the backtest:

- slightly misstates returns between rebalances, and
- leaves out the cost of the trades that undo drift.

The gap grows with longer rebalance periods and more volatile stocks.

### Status

Not fixed. This is a quick research backtest; tracking real shares and cash
is the job of the event-driven backtester, which is not built yet. The
function's docstring now states the assumption.

---

## 4. Signals don't show every trade

**File:** `src/strategies/weights.py`
**Function:** `weights_to_signals(weights)`

### The issue

This function turns weights into events such as `LONG_ENTER` and
`LONG_EXIT`. It only creates an event when a stock **changes side**
(flat, long or short). A stock that stays selected gets no new event, even
if its weight changes from 5% to 10%.

| Month | AAPL weight | Event |
|---|---|---|
| Jan | 5% | `LONG_ENTER` |
| Feb | 5% | none |
| Mar | 10% | none (still long, just bigger) |
| Apr | 0% | `LONG_EXIT` |

### Who it affects

- **Momentum and multi-factor:** every selected stock gets the same weight,
  so weights don't change while a stock stays selected. Nothing is missed.
- **Pairs and Kalman pairs:** weights change every bar as the hedge ratio
  moves, and none of those changes appear as events.

### Status

By design. Signals are a short summary for people. Anything that calculates
money, like `portfolio_returns`, uses the weights directly, so backtest
results are not affected.
