"""Broker interface and a paper broker that simulates fills from market data.

No live broker is implemented. A live one must provide the same methods and
attach the stop-loss and take-profit to the exchange order itself, so a
crash after entry never leaves a position unprotected.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Protocol

import pandas as pd

from unitrader.state import State

FEE = 0.0005  # per side, matches the backtests

# Execution (hourly) and the risk monitor (every minute) run in one process
# on different threads; hold this around any read-modify-write of the account.
ACCOUNT_LOCK = threading.RLock()


class Broker(Protocol):
    def equity(self, prices: dict[str, float] | None = None) -> float: ...
    def positions(self) -> dict[str, dict]: ...
    def get_positions(self) -> dict[str, dict]: ...
    def close_all(self, prices: dict[str, float], at: datetime, reason: str) -> list[dict]: ...
    def open_position(self, *, symbol: str, side: str, qty: float, price: float, stop_loss: float,
                      take_profit: float, opened_at: datetime, close_by: datetime, signal_id: str) -> dict: ...
    def close_position(self, symbol: str, price: float, at: datetime, reason: str) -> dict: ...
    def mark(self, data: pd.DataFrame) -> list[dict]: ...


class PaperBroker:
    """Simulated account persisted in state/paper_account.json."""

    NAME = "paper_account.json"

    def __init__(self, store: State, starting_equity: float):
        self.store = store
        path = store.path(self.NAME)
        self.account = store.read(self.NAME) if path.exists() else {
            "cash": starting_equity, "positions": {}, "closed": [],
        }

    def _save(self) -> None:
        self.store.write(self.NAME, self.account)

    def equity(self, prices: dict[str, float] | None = None) -> float:
        """Realised equity, plus open P&L marked at ``prices`` when given."""
        equity = self.account["cash"]
        for symbol, pos in self.account["positions"].items():
            if prices and symbol in prices:
                direction = 1 if pos["side"] == "long" else -1
                equity += direction * (prices[symbol] - pos["entry_price"]) * pos["qty"]
        return equity

    def positions(self) -> dict[str, dict]:
        return dict(self.account["positions"])

    get_positions = positions

    def close_all(self, prices: dict[str, float], at: datetime, reason: str) -> list[dict]:
        missing = set(self.account["positions"]) - set(prices)
        if missing:
            raise ValueError(f"no price to close {sorted(missing)}")
        return [self.close_position(s, prices[s], at, reason) for s in list(self.account["positions"])]

    def open_position(self, *, symbol, side, qty, price, stop_loss, take_profit, opened_at, close_by, signal_id):
        if symbol in self.account["positions"]:
            raise ValueError(f"{symbol}: position already open")
        pos = {
            "symbol": symbol, "side": side, "qty": qty, "entry_price": price,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "opened_at": opened_at.isoformat(), "close_by": close_by.isoformat(), "signal_id": signal_id,
        }
        self.account["cash"] -= qty * price * FEE
        self.account["positions"][symbol] = pos
        self._save()
        return pos

    def close_position(self, symbol, price, at, reason):
        pos = self.account["positions"].pop(symbol)
        direction = 1 if pos["side"] == "long" else -1
        pnl = direction * (price - pos["entry_price"]) * pos["qty"] - pos["qty"] * price * FEE
        self.account["cash"] += pnl
        closed = {**pos, "exit_price": price, "closed_at": at.isoformat(), "reason": reason, "pnl": round(pnl, 6)}
        self.account["closed"].append(closed)
        self._save()
        return closed

    def mark(self, data: pd.DataFrame) -> list[dict]:
        """Close positions whose stop or target was touched by bars since entry.

        If one bar touches both, the stop is assumed to fill first.
        """
        closed = []
        for symbol, pos in list(self.account["positions"].items()):
            bars = data[(data["symbol"] == symbol) & (data["timestamp"] > pd.Timestamp(pos["opened_at"]))]
            for bar in bars.sort_values("timestamp").itertuples():
                long = pos["side"] == "long"
                hit_stop = bar.low <= pos["stop_loss"] if long else bar.high >= pos["stop_loss"]
                hit_target = bar.high >= pos["take_profit"] if long else bar.low <= pos["take_profit"]
                if hit_stop or hit_target:
                    price = pos["stop_loss"] if hit_stop else pos["take_profit"]
                    closed.append(self.close_position(symbol, price, bar.timestamp.to_pydatetime(),
                                                      "stop_loss" if hit_stop else "take_profit"))
                    break
        return closed
