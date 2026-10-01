"""@checker: functions that must approve a trading signal before it stands."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


_checkers: list[Callable[[dict], Verdict]] = []


def checker(fn: Callable[[dict], Verdict]) -> Callable[[dict], Verdict]:
    """Register ``fn(signal) -> Verdict``; every checker must approve a signal."""
    if fn not in _checkers:
        _checkers.append(fn)
    return fn


def run_checkers(signal: dict) -> dict[str, Verdict]:
    """Run every registered checker. A checker that raises counts as a rejection."""
    verdicts = {}
    for fn in _checkers:
        try:
            verdicts[fn.__name__] = fn(signal)
        except Exception as exc:
            log.exception("checker %s failed on %s", fn.__name__, signal.get("symbol"))
            verdicts[fn.__name__] = Verdict(False, [f"checker error: {exc}"])
    return verdicts
