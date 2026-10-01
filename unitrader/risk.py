"""Minute-by-minute drawdown guard for the trading account.

    @loop(interval="1m")
    def monitor_risk(): ...

Equity is marked to live prices and compared with its high-water mark. If it
falls more than MAX_DRAWDOWN below the peak, every position is closed, the
kill switch (data/KILL) is engaged so the next signal can't reopen them, and
the event is appended to data/STATE.md. It fires once; deleting data/KILL
after review re-arms it and resets the peak to the then-current equity.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from unitrader import execution, ingest
from unitrader.broker import ACCOUNT_LOCK
from unitrader.market_data import fetch_last_prices
from unitrader.scheduler import loop

log = logging.getLogger(__name__)

MAX_DRAWDOWN = 0.05
RISK_STATE = "risk.json"
STATE_LOG = "STATE.md"


def clock() -> datetime:
    return datetime.now(timezone.utc)


def drawdown(equity: float, peak: float) -> float:
    return 0.0 if peak <= 0 else max(0.0, 1 - equity / peak)


@loop(interval="1m")
def monitor_risk():
    state = ingest.state
    with ACCOUNT_LOCK:
        now = clock()
        broker = execution.make_broker()
        positions = broker.get_positions()
        prices = fetch_last_prices(list(positions)) if positions else {}
        missing = sorted(set(positions) - set(prices))
        if missing:
            # Can't value the book; don't act on a guess. run_once logs this.
            raise RuntimeError(f"no live price for open positions {missing}; drawdown not evaluated")

        equity = broker.equity(prices)
        risk = state.read(RISK_STATE) if state.path(RISK_STATE).exists() else {}
        killed = state.path(execution.KILL_FILE).exists()
        if risk.get("tripped") and not killed:
            log.warning("kill switch removed; re-arming drawdown guard at equity %.2f", equity)
            risk = {}

        peak = max(risk.get("peak_equity", equity), equity)
        dd = drawdown(equity, peak)
        risk.update(peak_equity=peak, last_equity=equity, last_drawdown=round(dd, 6), checked_at=now.isoformat())

        if dd > MAX_DRAWDOWN and not risk.get("tripped"):
            closed = broker.close_all(prices, now, "drawdown")
            state.path(execution.KILL_FILE).write_text(
                f"Drawdown trigger at {now.isoformat()}. Delete this file to resume trading.\n")
            summary = ", ".join(f"{c['symbol']} {c['side']} @ {c['exit_price']} (pnl {c['pnl']:.2f})" for c in closed)
            state.append(STATE_LOG, (
                f"- {now.isoformat()} Drawdown trigger hit: equity {equity:.2f} is {dd:.2%} below peak "
                f"{peak:.2f} (limit {MAX_DRAWDOWN:.0%}). All positions closed: {summary or 'none open'}. "
                f"Trading halted; delete data/KILL to resume."
            ))
            risk.update(tripped=True, tripped_at=now.isoformat())
            log.error("drawdown %.2f%% > %.0f%%: closed %d positions, kill switch engaged",
                      dd * 100, MAX_DRAWDOWN * 100, len(closed))
        state.write(RISK_STATE, risk)
