"""Hard rules from the alpha_research skill, enforced in code.

The skill text tells Claude the rules, but nothing here trusts the model to
follow them: the regression, backtest gate and calendar blackouts are
computed before the call, and enforce() overrides any signal that breaks them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from unitrader import config, research
from unitrader.features import summarize_symbol

SHARPE_MIN = 1.5
PASSES_REQUIRED = 3  # of research.N_BACKTESTS
MAX_POSITION_PCT = 2.0
MAX_SECTOR_PCT = 30.0
EARNINGS_BLACKOUT = timedelta(hours=48)
CALENDAR_PATH = Path(__file__).parent / "calendar.json"


def load_calendar(path: Path = CALENDAR_PATH) -> dict:
    raw = json.loads(path.read_text())
    return {
        "fomc": {date.fromisoformat(d) for d in raw.get("fomc_announcements", [])},
        "earnings": [
            {"symbol": e["symbol"], "datetime": datetime.fromisoformat(e["datetime"])}
            for e in raw.get("earnings", [])
        ],
    }


def is_fomc_day(now: datetime, calendar: dict) -> bool:
    return now.date() in calendar["fomc"]


def earnings_within_blackout(symbol: str, now: datetime, calendar: dict) -> datetime | None:
    for e in calendar["earnings"]:
        if e["symbol"] == symbol and timedelta(0) <= e["datetime"] - now <= EARNINGS_BLACKOUT:
            return e["datetime"]
    return None


def sector_of(symbol: str) -> str:
    return config.SECTORS.get(symbol, symbol)


def evaluate(data: pd.DataFrame, now: datetime, calendar: dict | None = None) -> dict:
    """Everything the skill needs: features, regression, gates, blackouts."""
    calendar = calendar or load_calendar()
    fomc = is_fomc_day(now, calendar)
    symbols = {}
    for sym, df in data.groupby("symbol", sort=True):
        model = research.analyze(df, sharpe_min=SHARPE_MIN, passes_required=PASSES_REQUIRED)
        blocked = []
        if fomc:
            blocked.append("FOMC announcement day")
        if (when := earnings_within_blackout(sym, now, calendar)) is not None:
            blocked.append(f"earnings at {when.isoformat()} within 48h")
        if not model["gate_passed"]:
            blocked.append(
                f"backtest gate failed: Sharpe > {SHARPE_MIN} in {model['backtests_passed']}"
                f"/{research.N_BACKTESTS} (need {PASSES_REQUIRED})"
            )
        allowed = ["flat"] if blocked or model["direction"] == "flat" else [model["direction"], "flat"]
        symbols[sym] = {
            "sector": sector_of(sym),
            "features": summarize_symbol(df),
            "regression": model,
            "blocked_reasons": blocked,
            "allowed_actions": allowed,
        }
    return {
        "timeframe": "1h",
        "now": now.isoformat(),
        "data_as_of": data["timestamp"].max().isoformat(),
        "fomc_day": fomc,
        "limits": {"max_position_pct": MAX_POSITION_PCT, "max_sector_pct": MAX_SECTOR_PCT},
        "symbols": symbols,
    }


def flat_signal(symbol: str, info: dict, rationale: str) -> dict:
    return {
        "symbol": symbol, "action": "flat", "conviction": 0.0, "position_size_pct": 0.0,
        "horizon_hours": 1, "entry_reference": info["features"]["last_close"],
        "stop_loss": None, "take_profit": None, "rationale": rationale,
    }


def all_flat(context: dict) -> dict:
    """Output for runs where no symbol may trade; no model call needed."""
    signals = [
        flat_signal(sym, info, "; ".join(info["blocked_reasons"]) or "regression forecast below trading costs")
        for sym, info in context["symbols"].items()
    ]
    summary = "FOMC announcement day: all signals skipped." if context["fomc_day"] else "No symbol passed the rules this hour."
    return {"market_summary": summary, "signals": signals}


def enforce(output: dict, context: dict) -> tuple[dict, list[str]]:
    """Force rule-breaking signals flat and clamp sizes. Returns (output, adjustments)."""
    adjustments: list[str] = []
    signals = []
    for s in output["signals"]:
        s = dict(s)
        info = context["symbols"][s["symbol"]]
        if s["action"] not in info["allowed_actions"]:
            why = "; ".join(info["blocked_reasons"]) or f"regression direction is {info['regression']['direction']}"
            adjustments.append(f"{s['symbol']}: {s['action']} -> flat ({why})")
            s = flat_signal(s["symbol"], info, f"Overridden to flat: {why}. Model said: {s['rationale']}")
        if s["action"] == "flat":
            s["position_size_pct"] = 0.0
        elif s["position_size_pct"] > MAX_POSITION_PCT:
            adjustments.append(f"{s['symbol']}: size {s['position_size_pct']}% -> {MAX_POSITION_PCT}%")
            s["position_size_pct"] = MAX_POSITION_PCT
        signals.append(s)

    by_sector = defaultdict(list)
    for s in signals:
        if s["action"] != "flat":
            by_sector[sector_of(s["symbol"])].append(s)
    for sector, group in by_sector.items():
        total = sum(s["position_size_pct"] for s in group)
        if total > MAX_SECTOR_PCT:
            scale = MAX_SECTOR_PCT / total
            for s in group:
                s["position_size_pct"] = round(s["position_size_pct"] * scale, 4)
            adjustments.append(f"sector {sector}: {total}% -> {MAX_SECTOR_PCT}%")
    return {**output, "signals": signals}, adjustments
