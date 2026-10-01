import pytest

from unitrader.scheduler import loop, next_run
from unitrader.timeutil import parse_duration


def test_parse_duration():
    assert parse_duration("1h") == 3600
    assert parse_duration("30d") == 30 * 86400
    with pytest.raises(ValueError):
        parse_duration("hourly")


def test_next_run_aligns_to_interval():
    assert next_run(3600 * 5 + 1500, 3600) == 3600 * 6
    assert next_run(3600 * 5, 3600) == 3600 * 6


def test_loop_runs_on_aligned_boundaries_and_survives_errors():
    calls = []

    @loop(interval="1h")
    def job():
        calls.append(clock.now)
        if len(calls) == 2:
            raise RuntimeError("transient")

    class Clock:
        now = 3600 * 10 + 900.0

        def __call__(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    job.run_forever(max_runs=3, sleep=clock.sleep, clock=clock)
    assert calls == [3600 * 10 + 900.0, 3600 * 11, 3600 * 12]
    assert job.interval_seconds == 3600


def test_decorated_function_is_still_callable():
    @loop(interval="1h")
    def job():
        return 42

    assert job() == 42
