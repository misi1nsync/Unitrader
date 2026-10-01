"""Minimal interval scheduler: decorate a function with @loop(interval="1h")."""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from unitrader.timeutil import parse_duration

log = logging.getLogger(__name__)


def next_run(now: float, interval_s: int) -> float:
    """Next wall-clock boundary aligned to the interval (e.g. top of the hour)."""
    return (now // interval_s + 1) * interval_s


def loop(interval: str, *, run_immediately: bool = True, align: bool = True):
    """Mark a function to run every ``interval``.

    The decorated function still runs once when called directly; call
    ``fn.run_forever()`` to start the loop. Exceptions are logged and the
    loop continues, so one failed run doesn't stop ingestion.
    """
    interval_s = parse_duration(interval)

    def decorate(fn: Callable[[], object]):
        def run_once() -> bool:
            started = time.monotonic()
            try:
                fn()
            except Exception:
                log.exception("%s failed", fn.__name__)
                return False
            log.info("%s finished in %.1fs", fn.__name__, time.monotonic() - started)
            return True

        def run_forever(
            max_runs: int | None = None,
            *,
            sleep: Callable[[float], None] = time.sleep,
            clock: Callable[[], float] = time.time,
        ) -> None:
            runs = 0
            if run_immediately:
                run_once()
                runs += 1
            while max_runs is None or runs < max_runs:
                now = clock()
                wake = next_run(now, interval_s) if align else now + interval_s
                sleep(max(0.0, wake - now))
                run_once()
                runs += 1

        @functools.wraps(fn)
        def wrapper():
            return fn()

        wrapper.interval_seconds = interval_s
        wrapper.run_once = run_once
        wrapper.run_forever = run_forever
        return wrapper

    return decorate
