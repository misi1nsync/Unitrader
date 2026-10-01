import pandas as pd

from unitrader import ingest


def test_ingest_writes_latest_data(tmp_path, monkeypatch):
    frame = pd.DataFrame({"symbol": ["BTCUSDT"], "close": [1.0]})
    seen = {}

    def fake_fetch(symbols, lookback):
        seen.update(symbols=symbols, lookback=lookback)
        return frame

    monkeypatch.setattr(ingest, "fetch_market_data", fake_fetch)
    monkeypatch.setattr(ingest, "state", ingest.State(tmp_path))
    assert ingest.ingest_data.run_once() is True
    assert seen == {"symbols": ingest.universe, "lookback": "30d"}
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "latest_data.parquet"), frame)


def test_failed_ingest_does_not_overwrite_previous_data(tmp_path, monkeypatch):
    state = ingest.State(tmp_path)
    old = pd.DataFrame({"close": [1.0]})
    state.write("latest_data.parquet", old)

    def failing_fetch(symbols, lookback):
        raise RuntimeError("exchange down")

    monkeypatch.setattr(ingest, "fetch_market_data", failing_fetch)
    monkeypatch.setattr(ingest, "state", state)
    assert ingest.ingest_data.run_once() is False
    pd.testing.assert_frame_equal(state.read("latest_data.parquet"), old)
