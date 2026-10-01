"""Run the full pipeline: hourly ingest -> signal -> paper execution,
plus the risk monitor every minute on a background thread."""

import logging
import threading

from unitrader import execution, signals  # noqa: F401  (subscribe generate_signal and execute)
from unitrader.ingest import ingest_data
from unitrader.risk import monitor_risk


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    threading.Thread(target=monitor_risk.run_forever, name="risk-monitor", daemon=True).start()
    ingest_data.run_forever()


if __name__ == "__main__":
    main()
