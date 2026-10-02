import json

from unitrader.features import RECENT_CLOSES, summarize


def test_summarize_shape_and_ranges(ohlcv):
    out = summarize(ohlcv)
    assert set(out["symbols"]) == {"BTCUSDT", "ETHUSDT"}
    assert out["data_as_of"] == "2026-09-30T23:00:00+00:00"
    btc = out["symbols"]["BTCUSDT"]
    assert btc["bars"] == 720
    assert 0 <= btc["rsi14"] <= 100
    assert btc["window_low"] <= btc["last_close"] <= btc["window_high"]
    assert len(btc["recent_closes"]) == RECENT_CLOSES
    assert btc["recent_closes"][-1] == btc["last_close"]
    expected_24h = (ohlcv[ohlcv.symbol == "BTCUSDT"].close.iloc[-1] / ohlcv[ohlcv.symbol == "BTCUSDT"].close.iloc[-25] - 1) * 100
    assert abs(btc["return_pct"]["24h"] - expected_24h) < 1e-3


def test_summary_is_compact_json(ohlcv):
    text = json.dumps(summarize(ohlcv), separators=(",", ":"))
    assert len(text) < 4000  # two symbols; raw bars would be ~100x larger
