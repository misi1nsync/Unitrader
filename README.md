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
