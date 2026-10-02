"""Long hourly history per symbol, for backtests the 30-day window can't support.

Stored as data/history/<SYMBOL>.parquet. Backfill once with
`unitrader-backfill`; afterwards each hourly ingest appends its new bars.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from unitrader import config
from unitrader.market_data import fetch_symbol
from unitrader.state import State

log = logging.getLogger(__name__)

DEFAULT_YEARS = 3


def _store() -> State:
    return State(config.state_dir)


def _name(symbol: str) -> str:
    return f"history/{symbol}.parquet"


def load(symbol: str, store: State | None = None) -> pd.DataFrame | None:
    store = store or _store()
    path = store.path(_name(symbol))
    return store.read(_name(symbol)) if path.exists() else None


def merge(symbol: str, new: pd.DataFrame, store: State | None = None) -> pd.DataFrame:
    """Add bars to a symbol's history; later fetches win on duplicate timestamps."""
    store = store or _store()
    old = load(symbol, store)
    combined = new if old is None else pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp", ignore_index=True)
    store.write(_name(symbol), combined)
    return combined


def append_from_ingest(data: pd.DataFrame, store: State | None = None) -> None:
    """Keep already-backfilled histories current. Symbols never backfilled are skipped."""
    store = store or _store()
    for symbol, bars in data.groupby("symbol"):
        if store.path(_name(symbol)).exists():
            merge(symbol, bars, store)


def backfill(symbols: list[str], years: int = DEFAULT_YEARS, store: State | None = None) -> None:
    for symbol in symbols:
        df = fetch_symbol(symbol, lookback=f"{years * 365}d", timeframe="1h")
        merged = merge(symbol, df, store)
        log.info("%s: %d bars from %s to %s", symbol, len(merged),
                 merged["timestamp"].iloc[0].isoformat(), merged["timestamp"].iloc[-1].isoformat())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill long hourly history for verification backtests.")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("symbols", nargs="*", help="defaults to the configured universe")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backfill(args.symbols or config.universe, args.years)


if __name__ == "__main__":
    main()
