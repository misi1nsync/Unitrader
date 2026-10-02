"""Turn raw OHLCV history into a compact per-symbol summary for the model.

Sending 30 days of hourly bars for every symbol would be ~80k tokens per
run; these features carry the same information in a few hundred tokens each.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

RECENT_CLOSES = 48


def _sig(x: float, digits: int = 6) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def _pct(a: float, b: float) -> float | None:
    return _sig((a / b - 1) * 100, 4) if b else None


def _rsi(close: pd.Series, length: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] else math.inf
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, length: int = 14) -> float:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean().iloc[-1]


def summarize_symbol(df: pd.DataFrame) -> dict:
    df = df.sort_values("timestamp").reset_index(drop=True)
    close = df["close"]
    last = close.iloc[-1]

    def ret(bars: int) -> float | None:
        return _pct(last, close.iloc[-1 - bars]) if len(close) > bars else None

    log_ret = np.log(close).diff().dropna()
    week = log_ret.iloc[-168:]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    vol24 = df["volume"].iloc[-24:].sum()
    avg_vol24 = df["volume"].mean() * 24
    atr = _atr(df)
    hi, lo = df["high"].max(), df["low"].min()

    return {
        "bars": len(df),
        "first_bar": df["timestamp"].iloc[0].isoformat(),
        "last_bar": df["timestamp"].iloc[-1].isoformat(),
        "last_close": _sig(last),
        "return_pct": {"1h": ret(1), "24h": ret(24), "7d": ret(168), "window": _pct(last, close.iloc[0])},
        "realized_vol_annual_pct": _sig(week.std() * math.sqrt(24 * 365) * 100, 4) if len(week) > 1 else None,
        "atr14": _sig(atr),
        "atr14_pct": _sig(atr / last * 100, 4),
        "sma20": _sig(sma20),
        "sma50": _sig(sma50),
        "close_vs_sma20_pct": _pct(last, sma20),
        "sma20_vs_sma50_pct": _pct(sma20, sma50),
        "rsi14": _sig(_rsi(close), 4),
        "volume_24h_vs_avg": _sig(vol24 / avg_vol24, 4) if avg_vol24 else None,
        "window_high": _sig(hi),
        "window_low": _sig(lo),
        "pct_from_window_high": _pct(last, hi),
        "recent_closes": [_sig(c) for c in close.iloc[-RECENT_CLOSES:]],
    }


def summarize(data: pd.DataFrame) -> dict:
    """Summary keyed by symbol, plus the timestamp of the newest bar."""
    if data.empty:
        raise ValueError("no market data to summarize")
    symbols = {sym: summarize_symbol(group) for sym, group in data.groupby("symbol", sort=True)}
    return {
        "timeframe": "1h",
        "data_as_of": data["timestamp"].max().isoformat(),
        "symbols": symbols,
    }
