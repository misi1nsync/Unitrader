"""Generate a pending trading signal whenever fresh market data lands."""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timedelta, timezone

from unitrader import claude, config
from unitrader import ingest
from unitrader.scheduler import loop
from unitrader.timeutil import parse_duration

log = logging.getLogger(__name__)

SKILL = "alpha_research"


class InvalidSignal(ValueError):
    pass


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(output: dict, symbols: set[str]) -> None:
    """Reject output that breaks the skill's contract.

    Structured outputs guarantee the JSON shape; this checks the trading
    logic the schema can't express (sides of stops, ranges, coverage).
    """
    seen = [s["symbol"] for s in output["signals"]]
    if sorted(seen) != sorted(symbols):
        raise InvalidSignal(f"expected one signal per symbol {sorted(symbols)}, got {seen}")
    for s in output["signals"]:
        sym, action = s["symbol"], s["action"]
        entry, stop, target = s["entry_reference"], s["stop_loss"], s["take_profit"]
        if not (_finite(s["conviction"]) and 0 <= s["conviction"] <= 1):
            raise InvalidSignal(f"{sym}: conviction {s['conviction']} outside [0, 1]")
        if not 1 <= s["horizon_hours"] <= 168:
            raise InvalidSignal(f"{sym}: horizon_hours {s['horizon_hours']} outside [1, 168]")
        if not (_finite(entry) and entry > 0):
            raise InvalidSignal(f"{sym}: bad entry_reference {entry}")
        if action == "flat":
            if stop is not None or target is not None:
                raise InvalidSignal(f"{sym}: flat signal must not carry stop/target")
            continue
        if not (_finite(stop) and _finite(target)):
            raise InvalidSignal(f"{sym}: {action} signal needs numeric stop_loss and take_profit")
        ok = stop < entry < target if action == "long" else target < entry < stop
        if not ok:
            raise InvalidSignal(f"{sym}: {action} levels out of order (stop={stop}, entry={entry}, target={target})")


@loop(trigger="data_updated")
def generate_signal():
    data = ingest.state.read("latest_data.parquet")
    result = claude.run_skill(SKILL, data)
    validate(result.output, set(data["symbol"].unique()))

    now = datetime.now(timezone.utc)
    signal = {
        "status": "pending",
        "skill": SKILL,
        "model": result.model,
        "generated_at": now.isoformat(),
        "data_as_of": data["timestamp"].max().isoformat(),
        # Consumers must ignore a signal once the next hourly run is due.
        "expires_at": (now + timedelta(seconds=parse_duration(config.interval))).isoformat(),
        "usage": result.usage,
        **result.output,
    }
    ingest.state.write("pending_signal.json", signal)
    actions = ", ".join(f"{s['symbol']}={s['action']}" for s in result.output["signals"])
    log.info("pending signal written: %s", actions)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a signal from the latest ingested data.")
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(0 if generate_signal.run_once() else 1)


if __name__ == "__main__":
    main()
