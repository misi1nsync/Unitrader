"""Run the full pipeline: hourly ingest, with a signal after each update."""

import logging

from unitrader import signals  # noqa: F401  (subscribes generate_signal to data_updated)
from unitrader.ingest import ingest_data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ingest_data.run_forever()


if __name__ == "__main__":
    main()
