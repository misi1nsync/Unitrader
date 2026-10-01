import pytest
import requests

from unitrader.market_data import MarketDataError, fetch_market_data, fetch_symbol

HOUR = 3_600_000
NOW = 1_790_812_800_000 + 25 * 60_000  # 25 minutes into an hour


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"status {self.status}")

    def json(self):
        return self.payload


class FakeBybit:
    """Serves hourly candles newest-first, honouring start/end/limit like Bybit."""

    def __init__(self, fail=(), limit_override=None):
        self.fail = set(fail)
        self.limit_override = limit_override
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        if params["symbol"] in self.fail:
            return FakeResponse({"retCode": 10001, "retMsg": "params error: symbol invalid"})
        limit = self.limit_override or params["limit"]
        opens = []
        ts = params["end"] // HOUR * HOUR
        while ts >= params["start"] and len(opens) < limit:
            opens.append(ts)
            ts -= HOUR
        rows = [[str(t), "100", "110", "90", "105", "1.5", "157.5"] for t in opens]
        return FakeResponse({"retCode": 0, "result": {"list": rows}})


def test_fetch_symbol_returns_closed_bars_oldest_first():
    df = fetch_symbol("BTCUSDT", "1d", "1h", session=FakeBybit(), now_ms=NOW)
    assert len(df) == 24
    assert df["timestamp"].is_monotonic_increasing
    # The in-progress bar (opened at the top of the current hour) is excluded.
    current_open = NOW // HOUR * HOUR
    assert df["timestamp"].max().value // 1_000_000 == current_open - HOUR
    assert list(df.columns) == ["symbol", "timestamp", "open", "high", "low", "close", "volume", "turnover"]
    assert df["close"].dtype == "float64"


def test_fetch_symbol_paginates_without_gaps_or_duplicates():
    fake = FakeBybit(limit_override=200)
    df = fetch_symbol("BTCUSDT", "30d", "1h", session=fake, now_ms=NOW)
    assert len(df) == 720
    assert df["timestamp"].is_unique
    assert (df["timestamp"].diff().dropna() == df["timestamp"].diff().dropna().iloc[0]).all()
    assert len(fake.calls) == 4


def test_fetch_symbol_raises_on_exchange_error():
    with pytest.raises(MarketDataError, match="symbol invalid"):
        fetch_symbol("NOPEUSDT", "1d", session=FakeBybit(fail={"NOPEUSDT"}), now_ms=NOW)


def test_fetch_market_data_skips_failed_symbols():
    fake = FakeBybit(fail={"BADUSDT"})
    df = fetch_market_data(["BTCUSDT", "BADUSDT", "ETHUSDT"], "1d", session=fake, now_ms=NOW)
    assert sorted(df["symbol"].unique()) == ["BTCUSDT", "ETHUSDT"]
    assert len(df) == 48


def test_fetch_market_data_raises_when_everything_fails():
    with pytest.raises(MarketDataError, match="all symbols failed"):
        fetch_market_data(["BADUSDT"], "1d", session=FakeBybit(fail={"BADUSDT"}), now_ms=NOW)


def test_unsupported_timeframe():
    with pytest.raises(ValueError, match="unsupported timeframe"):
        fetch_symbol("BTCUSDT", "1d", "7m", session=FakeBybit(), now_ms=NOW)
