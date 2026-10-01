import copy
import json

import pytest

from unitrader import claude, events, ingest, signals
from unitrader.claude import SkillResult
from unitrader.state import State

GOOD = {
    "market_summary": "BTC trending, ETH ranging.",
    "signals": [
        {"symbol": "BTCUSDT", "action": "long", "conviction": 0.6, "horizon_hours": 24,
         "entry_reference": 100.0, "stop_loss": 97.0, "take_profit": 106.0, "rationale": "trend"},
        {"symbol": "ETHUSDT", "action": "flat", "conviction": 0.0, "horizon_hours": 24,
         "entry_reference": 50.0, "stop_loss": None, "take_profit": None, "rationale": "range"},
    ],
}


def with_change(**changes):
    out = copy.deepcopy(GOOD)
    out["signals"][0].update(changes)
    return out


def test_validate_accepts_good_output():
    signals.validate(GOOD, {"BTCUSDT", "ETHUSDT"})


@pytest.mark.parametrize("output,match", [
    (with_change(stop_loss=101.0), "out of order"),
    (with_change(action="short"), "out of order"),
    (with_change(conviction=1.5), "conviction"),
    (with_change(horizon_hours=0), "horizon"),
    (with_change(take_profit=None), "needs numeric"),
    (with_change(symbol="SOLUSDT"), "one signal per symbol"),
])
def test_validate_rejects_bad_output(output, match):
    with pytest.raises(signals.InvalidSignal, match=match):
        signals.validate(output, {"BTCUSDT", "ETHUSDT"})


def test_flat_with_stop_is_rejected():
    out = copy.deepcopy(GOOD)
    out["signals"][1]["stop_loss"] = 49.0
    with pytest.raises(signals.InvalidSignal, match="flat"):
        signals.validate(out, {"BTCUSDT", "ETHUSDT"})


@pytest.fixture
def tmp_state(tmp_path, monkeypatch, ohlcv):
    state = State(tmp_path)
    state.write("latest_data.parquet", ohlcv)
    monkeypatch.setattr(ingest, "state", state)
    return state


def test_generate_signal_writes_pending_file(tmp_state, monkeypatch):
    seen = {}

    def fake_run_skill(name, data):
        seen.update(name=name, rows=len(data))
        return SkillResult(output=GOOD, model="claude-opus-5-5", usage={"input_tokens": 1, "output_tokens": 2})

    monkeypatch.setattr(claude, "run_skill", fake_run_skill)
    assert signals.generate_signal.run_once() is True
    assert seen == {"name": "alpha_research", "rows": 1440}

    written = tmp_state.read("pending_signal.json")
    assert written["status"] == "pending"
    assert written["data_as_of"] == "2026-09-30T23:00:00+00:00"
    assert written["expires_at"] > written["generated_at"]
    assert written["signals"] == GOOD["signals"]


def test_invalid_output_leaves_previous_signal(tmp_state, monkeypatch):
    tmp_state.write("pending_signal.json", {"status": "pending", "signals": []})
    monkeypatch.setattr(claude, "run_skill",
                        lambda name, data: SkillResult(output=with_change(stop_loss=999.0), model="m"))
    assert signals.generate_signal.run_once() is False
    assert tmp_state.read("pending_signal.json") == {"status": "pending", "signals": []}


def test_data_updated_event_triggers_signal(tmp_state, monkeypatch):
    calls = []
    monkeypatch.setattr(claude, "run_skill",
                        lambda name, data: calls.append(name) or SkillResult(output=GOOD, model="m"))
    events.emit("data_updated")
    assert calls == ["alpha_research"]
    assert json.loads((tmp_state.root / "pending_signal.json").read_text())["status"] == "pending"
