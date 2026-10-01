import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv():
    """30 days of synthetic hourly bars for two symbols."""
    rng = np.random.default_rng(7)
    ts = pd.date_range("2026-09-01", periods=720, freq="h", tz="UTC")
    frames = []
    for sym, start, drift in [("BTCUSDT", 60000.0, 0.0004), ("ETHUSDT", 2500.0, -0.0002)]:
        close = start * np.exp(np.cumsum(rng.normal(drift, 0.004, len(ts))))
        open_ = np.r_[start, close[:-1]]
        frames.append(pd.DataFrame({
            "symbol": sym, "timestamp": ts, "open": open_,
            "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
            "close": close, "volume": rng.uniform(100, 200, len(ts)), "turnover": 0.0,
        }))
    return pd.concat(frames, ignore_index=True)
