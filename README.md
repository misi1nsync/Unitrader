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

Needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile). Run once against
the current data with `unitrader-signal`.

| Variable | Default |
|---|---|
| `UNITRADER_MODEL` | `claude-opus-5-5` |
| `UNITRADER_EFFORT` | `high` |
