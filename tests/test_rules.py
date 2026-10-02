from datetime import datetime, timezone

import pandas as pd

from conftest import make_bars
from unitrader import rules

CAL = {"fomc": {datetime(2026, 10, 28).date()}, "earnings": [
    {"symbol": "AAPL", "datetime": datetime(2026, 10, 30, 20, tzinfo=timezone.utc)},
]}


def ctx(**allowed):
    return {"symbols": {
        sym: {"allowed_actions": acts, "blocked_reasons": [] if "long" in acts or "short" in acts else ["gate failed"],
              "regression": {"direction": acts[0]}, "features": {"last_close": 100.0}}
        for sym, acts in allowed.items()
    }}


def sig(symbol, action="long", size=1.0):
    levels = {"long": (95.0, 110.0), "short": (105.0, 90.0), "flat": (None, None)}[action]
    return {"symbol": symbol, "action": action, "conviction": 0.5, "position_size_pct": size,
            "horizon_hours": 4, "entry_reference": 100.0, "stop_loss": levels[0], "take_profit": levels[1],
            "rationale": "r"}


def test_fomc_day_blocks_every_symbol():
    data = pd.concat([make_bars("BTCUSDT", ar=0.4, seed=3), make_bars("ETHUSDT", ar=0.4, seed=4)])
    out = rules.evaluate(data, datetime(2026, 10, 28, 12, tzinfo=timezone.utc), CAL)
    assert out["fomc_day"]
    for info in out["symbols"].values():
        assert info["allowed_actions"] == ["flat"]
        assert "FOMC announcement day" in info["blocked_reasons"]


def test_earnings_blackout_window():
    assert rules.earnings_within_blackout("AAPL", datetime(2026, 10, 29, 0, tzinfo=timezone.utc), CAL)
    assert rules.earnings_within_blackout("AAPL", datetime(2026, 10, 28, 19, tzinfo=timezone.utc), CAL) is None
    assert rules.earnings_within_blackout("AAPL", datetime(2026, 10, 31, 0, tzinfo=timezone.utc), CAL) is None
    assert rules.earnings_within_blackout("MSFT", datetime(2026, 10, 29, 0, tzinfo=timezone.utc), CAL) is None


def test_gate_pass_allows_only_forecast_direction():
    out = rules.evaluate(make_bars("BTCUSDT", ar=0.4, seed=3), datetime(2026, 10, 2, tzinfo=timezone.utc), CAL)
    info = out["symbols"]["BTCUSDT"]
    assert info["regression"]["gate_passed"]
    assert info["allowed_actions"] in (["long", "flat"], ["short", "flat"], ["flat"])
    assert info["allowed_actions"][0] == info["regression"]["direction"]


def test_enforce_overrides_disallowed_action():
    out, adj = rules.enforce({"market_summary": "", "signals": [sig("BTCUSDT"), sig("ETHUSDT", "short")]},
                             ctx(BTCUSDT=["long", "flat"], ETHUSDT=["flat"]))
    eth = next(s for s in out["signals"] if s["symbol"] == "ETHUSDT")
    assert eth["action"] == "flat" and eth["position_size_pct"] == 0 and eth["stop_loss"] is None
    assert adj == ["ETHUSDT: short -> flat (gate failed)"]


def test_enforce_clamps_size_to_two_percent():
    out, adj = rules.enforce({"market_summary": "", "signals": [sig("BTCUSDT", size=5.0)]}, ctx(BTCUSDT=["long", "flat"]))
    assert out["signals"][0]["position_size_pct"] == 2.0
    assert adj == ["BTCUSDT: size 5.0% -> 2.0%"]


def test_enforce_caps_sector_exposure(monkeypatch):
    monkeypatch.setattr(rules, "MAX_POSITION_PCT", 25.0)
    monkeypatch.setattr(rules.config, "SECTORS", {"ETHUSDT": "layer1", "SOLUSDT": "layer1"})
    out, adj = rules.enforce(
        {"market_summary": "", "signals": [sig("ETHUSDT", size=20.0), sig("SOLUSDT", size=20.0)]},
        ctx(ETHUSDT=["long", "flat"], SOLUSDT=["long", "flat"]),
    )
    assert [s["position_size_pct"] for s in out["signals"]] == [15.0, 15.0]
    assert adj == ["sector layer1: 40.0% -> 30.0%"]
