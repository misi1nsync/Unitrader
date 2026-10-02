"""Minimal scheduler: run a function on an interval or when an event fires.

    @loop(interval="1h")            # every hour, on the hour
    @loop(trigger="data_updated")   # whenever events.emit("data_updated") runs
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from unitrader import events
from unitrader.timeutil import parse_duration

log = logging.getLogger(__name__)


def next_run(now: float, interval_s: int) -> float:
    """Next wall-clock boundary aligned to the interval (e.g. top of the hour)."""
    return (now // interval_s + 1) * interval_s


def loop(
    interval: str | None = None,
    *,
    trigger: str | None = None,
    run_immediately: bool = True,
    align: bool = True,
):
    """Mark a function to run every ``interval`` or whenever ``trigger`` fires.

    The decorated function still runs once when called directly. For interval
    jobs, call ``fn.run_forever()`` to start the loop. Trigger jobs are
    subscribed to the event at decoration time. Exceptions are logged and
    swallowed, so one failed run doesn't stop the schedule.
    """
    if (interval is None) == (trigger is None):
        raise ValueError("pass exactly one of interval= or trigger=")
    interval_s = parse_duration(interval) if interval is not None else None

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
            if interval_s is None:
                raise RuntimeError(f"{fn.__name__} runs on trigger {trigger!r}, not an interval")
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
        wrapper.trigger = trigger
        wrapper.run_once = run_once
        wrapper.run_forever = run_forever
        if trigger is not None:
            events.subscribe(trigger, run_once)
        return wrapper

    return decorate
