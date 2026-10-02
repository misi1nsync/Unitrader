import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_bars
from unitrader import claude, history, research, verification
from unitrader.checks import _checkers, run_checkers
from unitrader.claude import SkillError, SkillResult
from unitrader.state import State

TWO_YEARS_PLUS = 24 * (730 + 30)


@pytest.fixture(scope="module")
def strong_history():
    return make_bars("BTCUSDT", n=TWO_YEARS_PLUS, ar=0.4, seed=3)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = State(tmp_path)
    monkeypatch.setattr(history, "_store", lambda: s)
    return s


def long_signal(symbol="BTCUSDT"):
    return {"symbol": symbol, "action": "long", "conviction": 0.6, "position_size_pct": 1.5, "horizon_hours": 2,
            "entry_reference": 100.0, "stop_loss": 99.0, "take_profit": 102.0, "rationale": "r"}


def test_rules_are_exactly_the_four_numeric_rules():
    assert [r.text for r in verification.RULES] == [
        "Sharpe ratio above 1.5",
        "Max drawdown below 10 percent",
        "Newey-West t-stat above 2.0",
        "Out of sample period at least 2 years",
    ]
    assert verification.verify_signal in _checkers


def test_newey_west_matches_plain_t_for_iid_and_shrinks_for_autocorrelated():
    rng = np.random.default_rng(0)
    iid = rng.normal(0.001, 0.01, 5000)
    plain = iid.mean() / (iid.std(ddof=0) / math.sqrt(len(iid)))
    assert abs(research.newey_west_tstat(iid) - plain) / abs(plain) < 0.1

    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.6 * ar[t - 1] + rng.normal(0.0004, 0.01)
    naive = ar.mean() / (ar.std(ddof=0) / math.sqrt(len(ar)))
    assert research.newey_west_tstat(ar) < naive * 0.75


def test_max_drawdown():
    assert research.max_drawdown_pct(np.log([1.1, 0.5, 1.2])) == pytest.approx(50.0)
    assert research.max_drawdown_pct(np.log([1.01, 1.02])) == 0.0


def test_flat_signal_needs_no_history(store):
    sig = dict(long_signal(), action="flat")
    assert verification.verify_signal(sig).approved


def test_missing_history_rejects(store):
    v = verification.verify_signal(long_signal())
    assert not v.approved and "unitrader-backfill" in v.reasons[0]


def test_short_history_fails_oos_rule_without_calling_claude(store, monkeypatch):
    history.merge("BTCUSDT", make_bars("BTCUSDT", n=24 * 60, ar=0.4, seed=3), store)
    monkeypatch.setattr(claude, "invoke", lambda **kw: pytest.fail("must not review a failed strategy"))
    v = verification.verify_signal(long_signal())
    assert not v.approved
    assert "failed: Out of sample period at least 2 years" in v.reasons


def test_random_walk_fails_rules(store, monkeypatch):
    history.merge("BTCUSDT", make_bars("BTCUSDT", n=TWO_YEARS_PLUS, seed=8), store)
    monkeypatch.setattr(claude, "invoke", lambda **kw: pytest.fail("must not review a failed strategy"))
    v = verification.verify_signal(long_signal())
    assert not v.approved
    assert v.details["checks"]["oos_period"]["passed"]
    assert not v.details["checks"]["sharpe"]["passed"]


@pytest.mark.parametrize("review,approved", [
    (SkillResult({"verdict": "approve", "concerns": [], "summary": "ok"}, "claude-opus-5-5"), True),
    (SkillResult({"verdict": "reject", "concerns": ["Sharpe 40 is implausible"], "summary": "no"}, "m"), False),
    (SkillError("declined"), False),
])
def test_review_runs_only_after_rules_pass_and_can_only_reject(store, monkeypatch, strong_history, review, approved):
    history.merge("BTCUSDT", strong_history, store)
    calls = []

    def fake_invoke(**kw):
        calls.append(kw)
        if isinstance(review, Exception):
            raise review
        return review

    monkeypatch.setattr(claude, "invoke", fake_invoke)
    v = verification.verify_signal(long_signal())
    assert all(c["passed"] for c in v.details["checks"].values())
    assert v.approved is approved
    assert calls[0]["skill"] == "backtest_verification_skill.md"
    assert calls[0]["rules"] == [r.text for r in verification.RULES]
    assert calls[0]["backtest"]["oos_days"] >= 730


def test_run_checkers_treats_exceptions_as_rejection(monkeypatch):
    monkeypatch.setattr(history, "load", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    verdicts = run_checkers(long_signal())
    assert not verdicts["verify_signal"].approved
    assert "checker error" in verdicts["verify_signal"].reasons[0]


def test_invoke_resolves_skill_file_name_and_sends_inputs():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(
            stop_reason="end_turn", stop_details=None, model="claude-opus-5-5", _request_id="r",
            content=[SimpleNamespace(type="text", text=json.dumps({"verdict": "approve", "concerns": [], "summary": ""}))],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    res = claude.invoke(skill="backtest_verification_skill.md", client=client, signal={"symbol": "X"}, rules=["a"])
    assert res.verdict == "approve"
    assert "last reviewer" in seen["system"]
    assert seen["output_config"]["format"]["schema"]["properties"]["verdict"]["enum"] == ["approve", "reject"]
    sent = json.loads(seen["messages"][0]["content"].split("\n\n", 1)[1])
    assert sent == {"signal": {"symbol": "X"}, "rules": ["a"]}


def test_history_merge_and_ingest_append(store):
    bars = make_bars("BTCUSDT", n=100, seed=1)
    history.merge("BTCUSDT", bars.iloc[:60], store)
    history.append_from_ingest(bars.iloc[40:], store)  # overlaps 20 bars
    assert len(history.load("BTCUSDT", store)) == 100
    history.append_from_ingest(make_bars("ETHUSDT", n=10), store)  # never backfilled
    assert history.load("ETHUSDT", store) is None
