"""Hourly market data ingestion job."""

from __future__ import annotations

import argparse
import logging

from unitrader import config, events
from unitrader.market_data import fetch_market_data
from unitrader.scheduler import loop
from unitrader.state import State

state = State(config.state_dir)
universe = config.universe


@loop(interval=config.interval)
def ingest_data():
    data = fetch_market_data(symbols=universe, lookback=config.lookback)
    state.write("latest_data.parquet", data)
    events.emit("data_updated")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest market data on a schedule.")
    parser.add_argument("--once", action="store_true", help="run a single ingest and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.once:
        raise SystemExit(0 if ingest_data.run_once() else 1)
    ingest_data.run_forever()


if __name__ == "__main__":
    main()
