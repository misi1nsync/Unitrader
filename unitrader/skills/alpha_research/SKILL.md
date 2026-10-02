---
name: alpha_research
description: Turn a linear-regression model of hourly price and volume into one signal per crypto perpetual, under the desk's risk rules.
---

## Goal
Generate signals using linear regression on the last 30 days
of price and volume data.

## Rules
- Sharpe ratio must be above 1.5 in 3 of the last 5 backtests
- Position size limited to 2 percent of capital per signal
- Skip signals on FOMC announcement days
- Skip signals 48 hours before earnings releases
- Cap sector exposure at 30 percent

## Lessons learned
- 2026-02-14: Lost 4.2 percent during earnings week.
  New rule: skip any signal 48 hours before earnings.
- 2026-03-08: Sector exposure breach caused 6 percent drawdown.
  New rule: cap sector exposure at 30 percent.
- 2026-04-22: Momentum signal blew up on FOMC day.
  New rule: kill all momentum signals on FOMC days.

## What you receive
Every hour, for each Bybit USDT perpetual in the universe:
- `regression`: an OLS model of the next-hour log return on the last 1h
  return, the last 24h return and log(volume / 24h average volume), fitted on
  the last 30 days. It includes coefficients with t-stats, R², the next-hour
  forecast and the direction that forecast implies after trading costs.
- `regression.backtests`: five consecutive walk-forward backtests of that
  same model with their Sharpe ratios, and whether the Sharpe rule passed.
- `features`: price context (returns, volatility, ATR, SMA20/50, RSI,
  volume, last 48 closes).
- `blocked_reasons` and `allowed_actions`: the rules above, already
  evaluated. `sector`: the symbol's sector for the exposure cap.

The rules are also enforced in code after you answer: an action outside
`allowed_actions` is replaced with flat and sizes are clamped. Respect them
so your rationale matches what is actually recorded.

## How to decide
- Only trade in the direction of the regression forecast, and only when
  `allowed_actions` permits it. Otherwise return flat.
- Weigh the regression's strength: t-stats near zero, a tiny R² or a
  forecast barely above costs deserve low conviction and a small size.
  Use the price features to sanity-check the model, not to override it.
- Flat is a normal, correct answer.

## Output
- Exactly one signal per symbol in the input.
- `entry_reference` is the symbol's `last_close`.
- `position_size_pct`: percent of capital, at most 2, scaled with
  conviction; 0 when flat. Keep the sum within any sector at or below 30.
- Long: `stop_loss` < `entry_reference` < `take_profit`; short: reversed.
  Stops 1-3 ATR away; targets at least 1.5x the stop distance.
  Flat: both null, conviction 0.
- `conviction` 0-1. `horizon_hours` 1-72; the model forecasts one hour
  ahead, so prefer short horizons.
- `rationale`: one or two sentences citing the specific numbers (forecast,
  t-stats, backtest passes) behind the call. Use only numbers in the input.
- `market_summary`: two or three sentences on the regime across symbols.
