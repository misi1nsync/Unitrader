"""Backtest verification for trading signals.

The four numeric rules are computed and decided in code on the symbol's long
history. Only a signal that passes all of them is shown to Claude, whose
review can reject it but can never approve one that failed a rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from unitrader import claude, history, research
from unitrader.checks import Verdict, checker

SKILL = "backtest_verification_skill.md"


@dataclass(frozen=True)
class Rule:
    key: str
    text: str
    passes: Callable[[dict], bool]


RULES = [
    Rule("sharpe", "Sharpe ratio above 1.5", lambda s: s["sharpe"] > 1.5),
    Rule("max_drawdown", "Max drawdown below 10 percent", lambda s: s["max_drawdown_pct"] < 10.0),
    Rule("newey_west_t", "Newey-West t-stat above 2.0", lambda s: s["newey_west_t"] > 2.0),
    Rule("oos_period", "Out of sample period at least 2 years", lambda s: s["oos_days"] >= 730),
]


@checker
def verify_signal(signal: dict) -> Verdict:
    if signal["action"] == "flat":
        return Verdict(True, ["flat signal: nothing to verify"])

    bars = history.load(signal["symbol"])
    if bars is None:
        return Verdict(False, [f"no long history for {signal['symbol']}; run unitrader-backfill"])

    stats = research.long_backtest(bars)
    checks = {r.key: {"rule": r.text, "passed": bool(r.passes(stats))} for r in RULES}
    failed = [c["rule"] for c in checks.values() if not c["passed"]]
    details = {"stats": stats, "checks": checks}
    if failed:
        return Verdict(False, [f"failed: {rule}" for rule in failed], details)

    try:
        result = claude.invoke(skill=SKILL, signal=signal, backtest=stats, rules=[r.text for r in RULES])
    except claude.SkillError as exc:
        # Fail closed: an unreviewed signal is not a verified one.
        return Verdict(False, [f"review unavailable: {exc}"], details)
    details["review"] = {**result.output, "model": result.model}
    if result.verdict != "approve":
        return Verdict(False, [f"review: {c}" for c in result.output.get("concerns", [])] or ["review rejected"], details)
    return Verdict(True, ["all rules passed; review approved"], details)
