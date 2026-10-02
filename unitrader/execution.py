"""Turn verified pending signals into (paper) positions.

    @auto_mode
    def execute(signal): ...

runs after every signal is written. Before any order it checks, in order:
kill switch, daily loss limit, signal freshness, that the signal was not
already executed, and per symbol that the recorded verification approved it,
a stop-loss is present, no position is already open, and the per-position
(2%) and total (10%) exposure caps hold.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from unitrader import config, events, ingest
from unitrader.broker import ACCOUNT_LOCK, Broker, PaperBroker

log = logging.getLogger(__name__)

MAX_POSITION = 0.02          # fraction of equity per position
MAX_TOTAL_EXPOSURE = 0.10    # fraction of equity across all positions
DAILY_LOSS_LIMIT = 0.03      # halt new entries after a 3% drop from the day's start
KILL_FILE = "KILL"           # touch data/KILL to stop all new orders
LEDGER = "active_trades.json"


def clock() -> datetime:
    return datetime.now(timezone.utc)


def make_broker() -> Broker:
    # "off" still loads the paper account so the risk monitor can protect it.
    if config.execution_mode in ("paper", "off"):
        return PaperBroker(ingest.state, config.paper_starting_equity)
    raise RuntimeError(f"execution mode {config.execution_mode!r} has no broker implementation")


def auto_mode(fn):
    """Run ``fn(signal)`` whenever a new signal is written, unless execution is off."""
    def on_signal_ready():
        if config.execution_mode == "off":
            log.info("execution off; not acting on signal")
            return
        try:
            with ACCOUNT_LOCK:
                fn(ingest.state.read("pending_signal.json"))
        except Exception:
            log.exception("%s failed", fn.__name__)

    events.subscribe("signal_ready", on_signal_ready)
    fn.on_signal_ready = on_signal_ready
    return fn


def _ledger() -> dict:
    path = ingest.state.path(LEDGER)
    return ingest.state.read(LEDGER) if path.exists() else {"executed_signals": [], "trades": [], "day": None}


def _approved(signal: dict, symbol: str) -> bool:
    # Use the verdicts recorded at generation time. Checking the truthiness of a
    # Verdict object would always pass; only an explicit approved=True counts.
    verdicts = signal.get("verification", {}).get(symbol)
    return bool(verdicts) and all(v.get("approved") is True for v in verdicts.values())


@auto_mode
def execute(signal: dict, broker: Broker | None = None) -> list[dict]:
    now = clock()
    broker = broker or make_broker()
    ledger = _ledger()
    actions: list[dict] = []

    def record(**entry):
        actions.append({"at": now.isoformat(), **entry})
        log.info("execution: %s", entry)

    # Exits first: stops/targets hit since the last run, then expired horizons.
    data = ingest.state.read("latest_data.parquet")
    for closed in broker.mark(data):
        record(action="closed", **{k: closed[k] for k in ("symbol", "reason", "exit_price", "pnl")})
    last_close = data.sort_values("timestamp").groupby("symbol")["close"].last().to_dict()
    for symbol, pos in broker.positions().items():
        if now >= datetime.fromisoformat(pos["close_by"]) and symbol in last_close:
            closed = broker.close_position(symbol, float(last_close[symbol]), now, "horizon")
            record(action="closed", symbol=symbol, reason="horizon", exit_price=closed["exit_price"], pnl=closed["pnl"])

    def finish(reason: str | None = None):
        if reason:
            record(action="skipped_signal", reason=reason)
        ledger["trades"].extend(actions)
        ingest.state.write(LEDGER, ledger)
        return actions

    equity = broker.equity()
    today = now.date().isoformat()
    if (ledger.get("day") or {}).get("date") != today:
        ledger["day"] = {"date": today, "start_equity": equity, "halted": False}
    day = ledger["day"]
    if ingest.state.path(KILL_FILE).exists():
        return finish("kill switch present")
    if day["halted"] or equity < day["start_equity"] * (1 - DAILY_LOSS_LIMIT):
        day["halted"] = True
        return finish(f"daily loss limit: equity {equity:.2f} vs day start {day['start_equity']:.2f}")
    if signal.get("status") != "pending":
        return finish(f"signal status is {signal.get('status')!r}")
    if now >= datetime.fromisoformat(signal["expires_at"]):
        return finish("signal expired")
    signal_id = signal["generated_at"]
    if signal_id in ledger["executed_signals"]:
        return finish("signal already executed")
    ledger["executed_signals"].append(signal_id)

    open_positions = broker.positions()
    exposure = sum(p["qty"] * p["entry_price"] for p in open_positions.values())
    for sig in signal["signals"]:
        symbol, action = sig["symbol"], sig["action"]
        if action == "flat":
            continue
        reason = None
        if not _approved(signal, symbol):
            reason = "not approved by verification"
        elif sig.get("stop_loss") is None or sig.get("take_profit") is None:
            reason = "missing stop_loss/take_profit"
        elif symbol in open_positions:
            reason = "position already open"
        if reason:
            record(action="skipped", symbol=symbol, reason=reason)
            continue

        fraction = min(sig["position_size_pct"] / 100, MAX_POSITION)
        notional = min(fraction * equity, MAX_TOTAL_EXPOSURE * equity - exposure)
        price = float(last_close.get(symbol, sig["entry_reference"]))
        qty = math.floor(notional / price * 1e6) / 1e6 if notional > 0 else 0.0
        if qty <= 0:
            record(action="skipped", symbol=symbol, reason="exposure cap reached")
            continue
        pos = broker.open_position(
            symbol=symbol, side=action, qty=qty, price=price,
            stop_loss=sig["stop_loss"], take_profit=sig["take_profit"],
            opened_at=now, close_by=now + timedelta(hours=sig["horizon_hours"]), signal_id=signal_id,
        )
        exposure += qty * price
        record(action="opened", symbol=symbol, side=action, qty=qty, price=price,
               stop_loss=pos["stop_loss"], take_profit=pos["take_profit"], mode=config.execution_mode)
    return finish()
