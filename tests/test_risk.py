from datetime import datetime, timedelta, timezone

import pytest

from unitrader import config, execution, ingest, risk
from unitrader.broker import PaperBroker
from unitrader.market_data import fetch_last_prices
from unitrader.state import State

NOW = datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc)


@pytest.fixture
def env(tmp_path, monkeypatch):
    state = State(tmp_path)
    monkeypatch.setattr(ingest, "state", state)
    monkeypatch.setattr(config, "execution_mode", "paper")
    monkeypatch.setattr(risk, "clock", lambda: NOW)
    broker = PaperBroker(state, 10_000.0)
    broker.open_position(symbol="BTCUSDT", side="long", qty=1.0, price=1000.0, stop_loss=1.0, take_profit=1e9,
                         opened_at=NOW - timedelta(hours=1), close_by=NOW + timedelta(hours=3), signal_id="s")
    prices = {"BTCUSDT": 1000.0}
    monkeypatch.setattr(risk, "fetch_last_prices", lambda symbols: {s: prices[s] for s in symbols if s in prices})
    return state, prices


def account(state):
    return state.read("paper_account.json")


def test_no_action_within_limit(env):
    state, prices = env
    assert risk.monitor_risk.run_once()
    prices["BTCUSDT"] = 700.0  # -300 on 9.9995k peak: about 3%
    assert risk.monitor_risk.run_once()
    assert "BTCUSDT" in account(state)["positions"]
    assert not state.path("KILL").exists() and not state.path("STATE.md").exists()
    assert 0.02 < state.read("risk.json")["last_drawdown"] < 0.05


def test_drawdown_closes_all_engages_kill_switch_and_logs_once(env):
    state, prices = env
    risk.monitor_risk.run_once()  # sets peak
    prices["BTCUSDT"] = 400.0     # -600: about 6% below peak
    assert risk.monitor_risk.run_once()
    assert account(state)["positions"] == {}
    assert account(state)["closed"][-1]["reason"] == "drawdown"
    assert state.path("KILL").exists()
    log = state.path("STATE.md").read_text()
    assert "Drawdown trigger hit" in log and "All positions closed: BTCUSDT long @ 400.0" in log

    risk.monitor_risk.run_once()
    risk.monitor_risk.run_once()
    assert state.path("STATE.md").read_text().count("Drawdown trigger hit") == 1


def test_kill_switch_blocks_execution_after_trigger(env):
    state, prices = env
    risk.monitor_risk.run_once()
    prices["BTCUSDT"] = 400.0
    risk.monitor_risk.run_once()
    sig = {"status": "pending", "generated_at": "g", "expires_at": (NOW + timedelta(hours=1)).isoformat(), "signals": []}
    import pandas as pd
    state.write("latest_data.parquet", pd.DataFrame({"symbol": ["BTCUSDT"], "timestamp": [pd.Timestamp(NOW)],
                                                     "open": [400.0], "high": [400.0], "low": [400.0], "close": [400.0],
                                                     "volume": [1.0], "turnover": [0.0]}))
    actions = execution.execute(sig)
    assert actions[-1]["reason"] == "kill switch present"


def test_removing_kill_switch_rearms_with_fresh_peak(env):
    state, prices = env
    risk.monitor_risk.run_once()
    prices["BTCUSDT"] = 400.0
    risk.monitor_risk.run_once()
    state.path("KILL").unlink()
    risk.monitor_risk.run_once()
    r = state.read("risk.json")
    assert not r.get("tripped")
    assert r["peak_equity"] == pytest.approx(account(state)["cash"])


def test_missing_live_price_does_not_act(env, monkeypatch):
    state, prices = env
    monkeypatch.setattr(risk, "fetch_last_prices", lambda symbols: {})
    assert risk.monitor_risk.run_once() is False
    assert "BTCUSDT" in account(state)["positions"]
    assert not state.path("KILL").exists()


def test_drawdown_math():
    assert risk.drawdown(9_400, 10_000) == pytest.approx(0.06)
    assert risk.drawdown(10_500, 10_000) == 0.0


def test_fetch_last_prices_parses_and_skips_failures():
    class Resp:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): pass
        def json(self): return self.payload

    class Session:
        def get(self, url, params, timeout):
            if params["symbol"] == "BAD":
                return Resp({"retCode": 10001, "retMsg": "bad symbol"})
            return Resp({"retCode": 0, "result": {"list": [{"lastPrice": "123.45"}]}})

    assert fetch_last_prices(["BTCUSDT", "BAD"], session=Session()) == {"BTCUSDT": 123.45}


def test_state_append(tmp_path):
    s = State(tmp_path)
    s.append("STATE.md", "one")
    s.append("STATE.md", "two\n")
    assert s.path("STATE.md").read_text() == "one\ntwo\n"
