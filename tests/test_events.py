import pytest

from unitrader import events
from unitrader.scheduler import loop


def test_trigger_job_runs_on_emit_and_failures_are_contained():
    calls = []

    @loop(trigger="test_event_a")
    def job():
        calls.append(1)
        raise RuntimeError("boom")

    @loop(trigger="test_event_a")
    def other():
        calls.append(2)

    assert events.emit("test_event_a") == 2
    assert calls == [1, 2]
    assert job.trigger == "test_event_a" and job.interval_seconds is None


def test_emit_with_no_subscribers():
    assert events.emit("nobody_listens") == 0


def test_loop_requires_exactly_one_schedule():
    with pytest.raises(ValueError):
        loop()
    with pytest.raises(ValueError):
        loop(interval="1h", trigger="x")


def test_trigger_job_cannot_run_forever():
    @loop(trigger="test_event_b")
    def job():
        pass

    with pytest.raises(RuntimeError):
        job.run_forever(max_runs=1)
