import copy
from datetime import datetime, timezone

import pandas as pd
import pytest

from conftest import make_bars
from unitrader import claude, events, history, ingest, rules, signals
from unitrader.claude import SkillResult
from unitrader.state import State

GOOD = {
    "market_summary": "BTC trending, ETH ranging.",
    "signals": [
        {"symbol": "BTCUSDT", "action": "long", "conviction": 0.6, "position_size_pct": 1.5, "horizon_hours": 4,
         "entry_reference": 100.0, "stop_loss": 97.0, "take_profit": 106.0, "rationale": "trend"},
        {"symbol": "ETHUSDT", "action": "flat", "conviction": 0.0, "position_size_pct": 0.0, "horizon_hours": 4,
         "entry_reference": 50.0, "stop_loss": None, "take_profit": None, "rationale": "range"},
    ],
}
SYMBOLS = {"BTCUSDT", "ETHUSDT"}


def with_change(**changes):
    out = copy.deepcopy(GOOD)
    out["signals"][0].update(changes)
    return out


def test_validate_accepts_good_output():
    signals.validate(GOOD, SYMBOLS)


@pytest.mark.parametrize("output,match", [
    (with_change(stop_loss=101.0), "out of order"),
    (with_change(action="short"), "out of order"),
    (with_change(conviction=1.5), "conviction"),
    (with_change(position_size_pct=-1), "position_size_pct"),
    (with_change(horizon_hours=0), "horizon"),
    (with_change(take_profit=None), "needs numeric"),
    (with_change(symbol="SOLUSDT"), "one signal per symbol"),
])
def test_validate_rejects_bad_output(output, match):
    with pytest.raises(signals.InvalidSignal, match=match):
        signals.validate(output, SYMBOLS)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """BTC has a planted, tradable edge; ETH is a random walk."""
    state = State(tmp_path)
    data = pd.concat([make_bars("BTCUSDT", ar=0.4, seed=3), make_bars("ETHUSDT", seed=11)], ignore_index=True)
    state.write("latest_data.parquet", data)
    monkeypatch.setattr(ingest, "state", state)
    monkeypatch.setattr(history, "_store", lambda: state)
    monkeypatch.setattr(signals, "clock", lambda: datetime(2026, 10, 2, 9, tzinfo=timezone.utc))
    ctx = rules.evaluate(data, datetime(2026, 10, 2, 9, tzinfo=timezone.utc))
    btc = ctx["symbols"]["BTCUSDT"]
    assert btc["regression"]["gate_passed"] and btc["allowed_actions"] != ["flat"], "fixture needs a tradable BTC"
    assert ctx["symbols"]["ETHUSDT"]["allowed_actions"] == ["flat"]
    return state, ctx


def model_output(ctx, btc_action=None, btc_size=1.5):
    btc = ctx["symbols"]["BTCUSDT"]
    action = btc_action or btc["allowed_actions"][0]
    last = btc["features"]["last_close"]
    sign = 1 if action == "long" else -1
    return {"market_summary": "s", "signals": [
        {"symbol": "BTCUSDT", "action": action, "conviction": 0.6, "position_size_pct": btc_size, "horizon_hours": 2,
         "entry_reference": last, "stop_loss": last * (1 - 0.01 * sign), "take_profit": last * (1 + 0.02 * sign),
         "rationale": "regression"},
        {"symbol": "ETHUSDT", "action": "long", "conviction": 0.5, "position_size_pct": 1.0, "horizon_hours": 2,
         "entry_reference": ctx["symbols"]["ETHUSDT"]["features"]["last_close"],
         "stop_loss": ctx["symbols"]["ETHUSDT"]["features"]["last_close"] * 0.99,
         "take_profit": ctx["symbols"]["ETHUSDT"]["features"]["last_close"] * 1.02, "rationale": "x"},
    ]}


def approve_review(**kw):
    return SkillResult({"verdict": "approve", "concerns": [], "summary": "ok"}, "claude-opus-5-5")


def test_generate_signal_enforces_rules(setup, monkeypatch):
    state, ctx = setup
    history.merge("BTCUSDT", make_bars("BTCUSDT", n=24 * 760, ar=0.4, seed=3), state)
    monkeypatch.setattr(claude, "invoke", approve_review)
    sent = {}

    def fake_run_skill(name, payload):
        sent.update(name=name, payload=payload)
        return SkillResult(output=model_output(ctx, btc_size=4.0), model="claude-opus-5-5", usage={"input_tokens": 1})

    monkeypatch.setattr(claude, "run_skill", fake_run_skill)
    assert signals.generate_signal.run_once() is True
    assert sent["name"] == "alpha_research"
    assert sent["payload"]["symbols"]["BTCUSDT"]["regression"]["backtests_passed"] >= 3

    written = state.read("pending_signal.json")
    by_sym = {s["symbol"]: s for s in written["signals"]}
    assert by_sym["BTCUSDT"]["position_size_pct"] == 2.0          # clamped from 4%
    assert by_sym["ETHUSDT"]["action"] == "flat"                  # gate failed, model said long
    assert any(a.startswith("ETHUSDT: long -> flat") for a in written["adjustments"])
    assert written["gates"]["ETHUSDT"]["blocked_reasons"]
    assert written["status"] == "pending" and written["model"] == "claude-opus-5-5"
    assert written["verification"]["BTCUSDT"]["verify_signal"]["approved"]


def test_unverified_signal_is_forced_flat(setup, monkeypatch):
    state, ctx = setup  # no long history for BTC
    monkeypatch.setattr(claude, "run_skill", lambda name, payload: SkillResult(output=model_output(ctx), model="m"))
    assert signals.generate_signal.run_once() is True
    written = state.read("pending_signal.json")
    btc = next(s for s in written["signals"] if s["symbol"] == "BTCUSDT")
    assert btc["action"] == "flat" and btc["position_size_pct"] == 0
    assert not written["verification"]["BTCUSDT"]["verify_signal"]["approved"]
    assert any("verification" in a for a in written["adjustments"])


def test_fomc_day_skips_model_call(setup, monkeypatch):
    state, _ = setup
    monkeypatch.setattr(signals, "clock", lambda: datetime(2026, 10, 28, 15, tzinfo=timezone.utc))
    monkeypatch.setattr(claude, "run_skill", lambda *a: pytest.fail("model must not be called on FOMC day"))
    assert signals.generate_signal.run_once() is True
    written = state.read("pending_signal.json")
    assert written["fomc_day"] and written["model"] is None
    assert {s["action"] for s in written["signals"]} == {"flat"}


def test_invalid_output_leaves_previous_signal(setup, monkeypatch):
    state, ctx = setup
    state.write("pending_signal.json", {"status": "pending", "signals": []})
    bad = model_output(ctx)
    bad["signals"][0]["stop_loss"], bad["signals"][0]["take_profit"] = bad["signals"][0]["take_profit"], bad["signals"][0]["stop_loss"]
    monkeypatch.setattr(claude, "run_skill", lambda name, payload: SkillResult(output=bad, model="m"))
    assert signals.generate_signal.run_once() is False
    assert state.read("pending_signal.json") == {"status": "pending", "signals": []}


def test_data_updated_event_triggers_signal(setup, monkeypatch):
    state, ctx = setup
    calls = []
    monkeypatch.setattr(claude, "run_skill",
                        lambda name, payload: calls.append(name) or SkillResult(output=model_output(ctx), model="m"))
    events.emit("data_updated")
    assert calls == ["alpha_research"]
    assert state.read("pending_signal.json")["status"] == "pending"
