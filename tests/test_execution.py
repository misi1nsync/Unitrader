from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from conftest import make_bars
from unitrader import config, events, execution, ingest
from unitrader.broker import PaperBroker
from unitrader.state import State

NOW = datetime(2026, 10, 2, 9, 5, tzinfo=timezone.utc)


@pytest.fixture
def env(tmp_path, monkeypatch):
    state = State(tmp_path)
    data = pd.concat([make_bars("BTCUSDT", seed=1), make_bars("ETHUSDT", seed=2, start=50.0)], ignore_index=True)
    data["timestamp"] = data["timestamp"] + (pd.Timestamp("2026-10-02 08:00", tz="UTC") - data["timestamp"].max())
    state.write("latest_data.parquet", data)
    monkeypatch.setattr(ingest, "state", state)
    monkeypatch.setattr(execution, "clock", lambda: NOW)
    monkeypatch.setattr(config, "execution_mode", "paper")
    last = data.sort_values("timestamp").groupby("symbol")["close"].last()
    return state, PaperBroker(state, 10_000.0), last


def make_signal(last, *, approved=True, size=1.5, **overrides):
    btc = float(last["BTCUSDT"])
    sig = {
        "status": "pending",
        "generated_at": NOW.replace(minute=0).isoformat(),
        "expires_at": (NOW + timedelta(minutes=55)).isoformat(),
        "signals": [
            {"symbol": "BTCUSDT", "action": "long", "conviction": 0.6, "position_size_pct": size, "horizon_hours": 4,
             "entry_reference": btc, "stop_loss": btc * 0.99, "take_profit": btc * 1.02, "rationale": "r"},
            {"symbol": "ETHUSDT", "action": "flat", "conviction": 0, "position_size_pct": 0, "horizon_hours": 1,
             "entry_reference": float(last["ETHUSDT"]), "stop_loss": None, "take_profit": None, "rationale": "r"},
        ],
        "verification": {"BTCUSDT": {"verify_signal": {"approved": approved, "reasons": []}}},
    }
    sig.update(overrides)
    return sig


def test_opens_approved_signal_with_two_percent_cap(env):
    state, broker, last = env
    actions = execution.execute(make_signal(last, size=5.0), broker=broker)
    opened = [a for a in actions if a["action"] == "opened"]
    assert len(opened) == 1 and opened[0]["symbol"] == "BTCUSDT"
    notional = opened[0]["qty"] * opened[0]["price"]
    assert notional <= 0.02 * 10_000 + 1e-6 and notional > 0.0199 * 10_000
    assert broker.positions()["BTCUSDT"]["stop_loss"] < opened[0]["price"]
    assert state.read("active_trades.json")["trades"][-1]["action"] == "opened"


def test_rejected_verdict_is_not_executed(env):
    # The original `if verify_signal(signal):` would pass here: a Verdict object is always truthy.
    _, broker, last = env
    actions = execution.execute(make_signal(last, approved=False), broker=broker)
    assert actions == [{"at": NOW.isoformat(), "action": "skipped", "symbol": "BTCUSDT",
                        "reason": "not approved by verification"}]
    assert broker.positions() == {}


def test_missing_verification_is_not_executed(env):
    _, broker, last = env
    execution.execute(make_signal(last, verification={}), broker=broker)
    assert broker.positions() == {}


def test_same_signal_never_executes_twice(env):
    _, broker, last = env
    sig = make_signal(last)
    execution.execute(sig, broker=broker)
    actions = execution.execute(sig, broker=broker)
    assert actions[-1]["reason"] == "signal already executed"
    assert len(broker.positions()) == 1


@pytest.mark.parametrize("overrides,reason", [
    ({"expires_at": (NOW - timedelta(minutes=1)).isoformat()}, "signal expired"),
    ({"status": "executed"}, "signal status is 'executed'"),
])
def test_stale_or_non_pending_signal_skipped(env, overrides, reason):
    _, broker, last = env
    actions = execution.execute(make_signal(last, **overrides), broker=broker)
    assert actions[-1]["reason"] == reason and broker.positions() == {}


def test_kill_switch(env):
    state, broker, last = env
    state.path("KILL").write_text("")
    actions = execution.execute(make_signal(last), broker=broker)
    assert actions[-1]["reason"] == "kill switch present" and broker.positions() == {}


def test_daily_loss_limit_halts_new_entries(env):
    state, broker, last = env
    execution.execute(make_signal(last, generated_at="a"), broker=broker)  # records day start equity 10k
    broker.account["cash"] = 9_600.0  # -4% on the day
    broker.close_position("BTCUSDT", float(last["BTCUSDT"]), NOW, "test")
    actions = execution.execute(make_signal(last, generated_at="b"), broker=broker)
    assert actions[-1]["reason"].startswith("daily loss limit")
    assert broker.positions() == {}


def test_missing_stop_is_never_sent(env):
    _, broker, last = env
    sig = make_signal(last)
    sig["signals"][0]["stop_loss"] = None
    execution.execute(sig, broker=broker)
    assert broker.positions() == {}


def test_paper_stop_and_horizon_exits(env):
    state, broker, last = env
    btc = float(last["BTCUSDT"])
    broker.open_position(symbol="BTCUSDT", side="long", qty=0.01, price=btc, stop_loss=btc * 0.9999,
                         take_profit=btc * 2, opened_at=NOW - timedelta(hours=5), close_by=NOW + timedelta(hours=1),
                         signal_id="x")
    broker.open_position(symbol="ETHUSDT", side="short", qty=1, price=float(last["ETHUSDT"]),
                         stop_loss=1e9, take_profit=1e-9, opened_at=NOW - timedelta(minutes=30),
                         close_by=NOW - timedelta(minutes=1), signal_id="y")
    actions = execution.execute(make_signal(last, approved=False, generated_at="z"), broker=broker)
    reasons = {a["symbol"]: a["reason"] for a in actions if a["action"] == "closed"}
    assert reasons == {"BTCUSDT": "stop_loss", "ETHUSDT": "horizon"}
    assert broker.positions() == {}


def test_signal_ready_event_runs_execute_and_respects_off(env, monkeypatch):
    state, _, last = env
    state.write("pending_signal.json", make_signal(last))
    monkeypatch.setattr(config, "execution_mode", "off")
    events.emit("signal_ready")
    assert not state.path("active_trades.json").exists()
    monkeypatch.setattr(config, "execution_mode", "paper")
    events.emit("signal_ready")
    assert state.read("paper_account.json")["positions"]["BTCUSDT"]["side"] == "long"


def test_live_mode_has_no_broker(monkeypatch):
    monkeypatch.setattr(config, "execution_mode", "live")
    with pytest.raises(RuntimeError, match="no broker implementation"):
        execution.make_broker()
