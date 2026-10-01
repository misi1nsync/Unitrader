"""Generate a pending trading signal whenever fresh market data lands."""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timedelta, timezone

from unitrader import claude, config, ingest, rules
from unitrader.scheduler import loop
from unitrader.timeutil import parse_duration

log = logging.getLogger(__name__)

SKILL = "alpha_research"


class InvalidSignal(ValueError):
    pass


def clock() -> datetime:
    return datetime.now(timezone.utc)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(output: dict, symbols: set[str]) -> None:
    """Reject output that is malformed beyond what rules.enforce can repair.

    Structured outputs guarantee the JSON shape; this checks what the schema
    can't express (coverage, ranges, sides of stops).
    """
    seen = [s["symbol"] for s in output["signals"]]
    if sorted(seen) != sorted(symbols):
        raise InvalidSignal(f"expected one signal per symbol {sorted(symbols)}, got {seen}")
    for s in output["signals"]:
        sym, action = s["symbol"], s["action"]
        entry, stop, target = s["entry_reference"], s["stop_loss"], s["take_profit"]
        if not (_finite(s["conviction"]) and 0 <= s["conviction"] <= 1):
            raise InvalidSignal(f"{sym}: conviction {s['conviction']} outside [0, 1]")
        if not (_finite(s["position_size_pct"]) and s["position_size_pct"] >= 0):
            raise InvalidSignal(f"{sym}: bad position_size_pct {s['position_size_pct']}")
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
    now = clock()
    context = rules.evaluate(data, now)

    if all(info["allowed_actions"] == ["flat"] for info in context["symbols"].values()):
        # Nothing may trade (FOMC day, gates failed, no forecast edge): skip the model call.
        output, adjustments, model, usage = rules.all_flat(context), [], None, {}
    else:
        result = claude.run_skill(SKILL, context)
        validate(result.output, set(context["symbols"]))
        output, adjustments = rules.enforce(result.output, context)
        model, usage = result.model, result.usage

    signal = {
        "status": "pending",
        "skill": SKILL,
        "model": model,
        "generated_at": now.isoformat(),
        "data_as_of": context["data_as_of"],
        # Consumers must ignore a signal once the next hourly run is due.
        "expires_at": (now + timedelta(seconds=parse_duration(config.interval))).isoformat(),
        "fomc_day": context["fomc_day"],
        "gates": {
            sym: {
                "backtests_passed": info["regression"]["backtests_passed"],
                "sharpes": [b["sharpe"] for b in info["regression"]["backtests"]],
                "forecast_next_1h_pct": info["regression"]["forecast_next_1h_pct"],
                "blocked_reasons": info["blocked_reasons"],
            }
            for sym, info in context["symbols"].items()
        },
        "adjustments": adjustments,
        "usage": usage,
        **output,
    }
    ingest.state.write("pending_signal.json", signal)
    for note in adjustments:
        log.warning("rule enforced: %s", note)
    actions = ", ".join(f"{s['symbol']}={s['action']}" for s in output["signals"])
    log.info("pending signal written: %s", actions)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a signal from the latest ingested data.")
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(0 if generate_signal.run_once() else 1)


if __name__ == "__main__":
    main()
