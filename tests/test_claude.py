import json
from types import SimpleNamespace

import pytest

from unitrader import claude


def text_response(payload, *, stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        model=model,
        _request_id="req_test",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=1200, output_tokens=800),
    )


class FakeClient:
    def __init__(self, response):
        self.response, self.kwargs = response, None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_run_skill_request_and_result(ohlcv):
    payload = {"market_summary": "quiet", "signals": []}
    client = FakeClient(text_response(payload))
    result = claude.run_skill("alpha_research", ohlcv, client=client)

    assert result.output == payload
    assert result.usage == {"input_tokens": 1200, "output_tokens": 800}
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["effort"] == "high"
    assert kw["output_config"]["format"]["schema"]["required"] == ["market_summary", "signals"]
    assert not kw["system"].startswith("---") and "name: alpha_research" not in kw["system"]
    sent = json.loads(kw["messages"][0]["content"].split("\n\n", 1)[1])
    assert set(sent["symbols"]) == {"BTCUSDT", "ETHUSDT"}
    assert "thinking" not in kw  # Opus 5.5 always thinks; effort is the control


def test_run_skill_reports_fallback_model(ohlcv):
    client = FakeClient(text_response({"market_summary": "", "signals": []}, model="claude-opus-4-8"))
    assert claude.run_skill("alpha_research", ohlcv, client=client).model == "claude-opus-4-8"


@pytest.mark.parametrize("stop_reason,match", [("refusal", "declined"), ("max_tokens", "truncated")])
def test_run_skill_raises_on_unusable_stop(ohlcv, stop_reason, match):
    client = FakeClient(text_response({}, stop_reason=stop_reason))
    with pytest.raises(claude.SkillError, match=match):
        claude.run_skill("alpha_research", ohlcv, client=client)


def test_unknown_skill():
    with pytest.raises(claude.SkillError, match="unknown skill"):
        claude.Skill.load("nope")
