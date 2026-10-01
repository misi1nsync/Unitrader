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

- **Skill:** `unitrader/skills/alpha_research/SKILL.md` is the system prompt;
  `schema.json` next to it is enforced with structured outputs. Add a skill by
  adding a directory with those two files.
- **Input:** the 30-day bars are reduced to per-symbol features (returns,
  volatility, ATR, SMA20/50, RSI, volume, last 48 closes) rather than sent raw.
- **Model:** `claude-opus-5-5` at effort `high`, with server-side
  `fallbacks: "default"` so a safety decline is retried on Anthropic's
  recommended fallback model; the model that answered is recorded in the file.
- **Validation:** stops/targets must sit on the correct side of entry,
  conviction in [0, 1], exactly one signal per ingested symbol. Invalid
  output, a refusal, or truncation leaves the previous file untouched.
- **Pending only:** nothing is traded. The file carries `status: "pending"`
  and an `expires_at` one interval out; consumers must ignore expired signals.

Needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile). Run once against
the current data with `unitrader-signal`.

| Variable | Default |
|---|---|
| `UNITRADER_MODEL` | `claude-opus-5-5` |
| `UNITRADER_EFFORT` | `high` |
