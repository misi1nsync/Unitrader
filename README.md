# Unitrader

Market data ingestion and trading research tools.

## Hourly market data ingestion

`unitrader.ingest` pulls the last 30 days of 1h candles for a universe of
Bybit USDT linear perpetuals and writes them to `data/latest_data.parquet`
every hour, on the hour.

```python
@loop(interval="1h")
def ingest_data():
    data = fetch_market_data(symbols=universe, lookback="30d")
    state.write("latest_data.parquet", data)
```

```bash
pip install -e '.[dev]'
unitrader-ingest --once   # single run, exit code 1 on failure
unitrader-ingest          # run immediately, then every hour on the hour
unitrader-run             # hourly ingest + a signal after every successful update
pytest
```

Output is one long-format table: `symbol, timestamp (UTC bar open), open,
high, low, close, volume, turnover`. Only fully closed bars are included.

### Configuration (environment variables)

| Variable | Default |
|---|---|
| `UNITRADER_UNIVERSE` | `BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT` |
| `UNITRADER_STATE_DIR` | `data` |
| `UNITRADER_INTERVAL` | `1h` |
| `UNITRADER_LOOKBACK` | `30d` |

### Failure behaviour

- A symbol that errors is logged and skipped; the rest are still written.
- If every symbol fails, the run fails and the previous file is left untouched.
- Writes are atomic (temp file + rename), so readers never see a partial file.
- An exception in one run is logged and the loop carries on to the next hour.

## Signal generation

After every successful ingest, `data_updated` fires and `generate_signal`
asks Claude for one signal per symbol, written to `data/pending_signal.json`.

```python
@loop(trigger="data_updated")
def generate_signal():
    data = state.read("latest_data.parquet")
    signal = claude.run_skill("alpha_research", data)
    state.write("pending_signal.json", signal)
```

- **Skill:** `unitrader/skills/alpha_research/SKILL.md` holds the goal,
  rules and lessons learned and is the system prompt; `schema.json` next to it
  is enforced with structured outputs.
- **Model:** an OLS regression of the next-hour return on the last 1h
  return, 24h return and relative volume, fitted on the 30-day window
  (`unitrader/research.py`). Five walk-forward backtests of the same model
  are run every hour.
- **Rules are enforced in code** (`unitrader/rules.py`), not left to the LLM:
  a symbol may only trade in the regression's direction, and only if Sharpe
  > 1.5 in at least 3 of the 5 backtests, it isn't an FOMC statement day, and
  it has no earnings within 48h. Sizes are clamped to 2% per signal and 30%
  per sector (`config.SECTORS`). Overrides are listed under `adjustments`.
- **Calendar:** `unitrader/calendar.json` holds FOMC statement days and
  earnings dates. Add each new year's FOMC dates when the Fed publishes them.
- **No-trade hours skip the model call:** if no symbol may trade, an all-flat
  signal is written without calling Claude.
- **Model:** `claude-opus-5-5` at effort `high`, with server-side
  `fallbacks: "default"`; the model that answered is recorded in the file.
- **Validation:** malformed output (stops on the wrong side, missing
  symbols, conviction outside [0, 1]), a refusal, or truncation leaves the
  previous file untouched.
- **Pending only:** nothing is traded. The file carries `status: "pending"`
  and an `expires_at` one interval out; consumers must ignore expired signals.

## Signal verification

Every non-flat signal must pass each registered `@checker` before it stands;
one that fails is forced flat and the reason is recorded under `verification`
and `adjustments`.

`verify_signal` (`unitrader/verification.py`) walk-forward backtests the
regression strategy on the symbol's long hourly history and requires, in code:

- Sharpe ratio above 1.5
- Max drawdown below 10 percent
- Newey-West t-stat above 2.0
- Out of sample period at least 2 years

Only if all four pass does Claude review the signal with the
`backtest_verification` skill. The review can reject but never overrides a
failed rule; if the review errors or is declined, the signal is rejected.

The 2-year rule needs more than the 30-day ingest window. Backfill once:

```bash
unitrader-backfill            # 3 years of 1h bars per symbol into data/history/
```

after which each hourly ingest appends new bars. Without a backfill, every
non-flat signal fails verification.

## Execution (paper only)

After each signal is written, `signal_ready` fires and `@auto_mode execute`
acts on it with the **paper broker** (`unitrader/broker.py`), which fills at
the last close and simulates stop/target/horizon exits from the ingested bars.
State lives in `data/paper_account.json`; every action is logged to
`data/active_trades.json`.

Before any order: no `data/KILL` file, daily loss under 3%, signal still
pending and unexpired, not already executed; per symbol, the recorded
verification approved it, it has a stop-loss and target, no position is
already open, and size stays within 2% of equity per position and 10% in
total.

### Risk monitor

`monitor_risk` (`unitrader/risk.py`) runs every minute on a background thread
under `unitrader-run`. It marks open positions to live Bybit prices and tracks
the equity high-water mark in `data/risk.json`. If equity falls more than 5%
below the peak, it closes every position, writes `data/KILL` so no new trades
open, and appends the event to `data/STATE.md`. It fires once; delete
`data/KILL` after review to resume, which also resets the peak. If a live
price is missing for an open position, that tick does nothing and logs an
error rather than acting on stale data.

No live broker is implemented: `UNITRADER_EXECUTION` accepts `paper`
(default) or `off`. A live broker must implement the `Broker` protocol and
attach stop-loss and take-profit to the exchange order itself.

Needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile). Run once against
the current data with `unitrader-signal`.

| Variable | Default |
|---|---|
| `UNITRADER_MODEL` | `claude-opus-5-5` |
| `UNITRADER_EFFORT` | `high` |
