"""Fetch OHLCV candles for Bybit USDT linear perpetuals."""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from unitrader.timeutil import parse_duration

log = logging.getLogger(__name__)

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
PAGE_LIMIT = 1000  # Bybit maximum candles per request

# Bybit interval codes, keyed by bar length in seconds.
_BYBIT_INTERVALS = {
    60: "1", 180: "3", 300: "5", 900: "15", 1800: "30",
    3600: "60", 7200: "120", 14400: "240", 21600: "360", 43200: "720",
    86400: "D", 604800: "W",
}

COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "turnover"]


class MarketDataError(RuntimeError):
    """Raised when the exchange returns an error or malformed data."""


def _bybit_interval(timeframe: str) -> tuple[str, int]:
    seconds = parse_duration(timeframe)
    if seconds not in _BYBIT_INTERVALS:
        raise ValueError(f"unsupported timeframe for Bybit: {timeframe!r}")
    return _BYBIT_INTERVALS[seconds], seconds


def fetch_symbol(
    symbol: str,
    lookback: str = "30d",
    timeframe: str = "1h",
    *,
    session: requests.Session | None = None,
    now_ms: int | None = None,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Fetch closed candles for one symbol covering ``lookback``, oldest first."""
    interval, bar_seconds = _bybit_interval(timeframe)
    bar_ms = bar_seconds * 1000
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    # Only keep bars that have fully closed.
    last_open = (now_ms // bar_ms) * bar_ms - bar_ms
    start = last_open - parse_duration(lookback) * 1000 + bar_ms

    http = session or requests.Session()
    rows: dict[int, list] = {}
    end = last_open
    while end >= start:
        resp = http.get(
            BYBIT_KLINE_URL,
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "limit": PAGE_LIMIT,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("retCode") != 0:
            raise MarketDataError(f"{symbol}: Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
        page = payload.get("result", {}).get("list") or []
        if not page:
            break
        for row in page:
            ts = int(row[0])
            if start <= ts <= last_open:
                rows[ts] = row
        oldest = min(int(r[0]) for r in page)
        # Don't trust page size as an end marker; walk back until we reach start.
        if oldest <= start:
            break
        end = oldest - bar_ms

    if not rows:
        raise MarketDataError(f"{symbol}: no candles returned")

    df = pd.DataFrame([rows[ts] for ts in sorted(rows)], columns=COLUMNS[1:])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in COLUMNS[2:]:
        df[col] = df[col].astype("float64")
    df.insert(0, "symbol", symbol)
    return df


def fetch_market_data(
    symbols: list[str],
    lookback: str = "30d",
    timeframe: str = "1h",
    *,
    session: requests.Session | None = None,
    now_ms: int | None = None,
) -> pd.DataFrame:
    """Fetch candles for every symbol and return one long-format frame.

    A symbol that fails is logged and skipped so one bad symbol does not
    sink the whole batch. Raises MarketDataError only if every symbol fails.
    """
    http = session or requests.Session()
    frames, failed = [], []
    for symbol in symbols:
        try:
            frames.append(fetch_symbol(symbol, lookback, timeframe, session=http, now_ms=now_ms))
        except (requests.RequestException, MarketDataError, ValueError) as exc:
            log.warning("failed to fetch %s: %s", symbol, exc)
            failed.append(symbol)
    if not frames:
        raise MarketDataError(f"all symbols failed: {', '.join(failed)}")
    if failed:
        log.warning("ingested %d/%d symbols; missing: %s", len(frames), len(symbols), ", ".join(failed))
    return pd.concat(frames, ignore_index=True)


BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"


def fetch_last_prices(
    symbols: list[str], *, session: requests.Session | None = None, timeout: float = 5.0
) -> dict[str, float]:
    """Latest traded price per symbol. Symbols that fail are omitted."""
    http = session or requests.Session()
    prices = {}
    for symbol in symbols:
        try:
            resp = http.get(BYBIT_TICKERS_URL, params={"category": "linear", "symbol": symbol}, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("retCode") != 0:
                raise MarketDataError(f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
            prices[symbol] = float(payload["result"]["list"][0]["lastPrice"])
        except (requests.RequestException, MarketDataError, KeyError, IndexError, ValueError) as exc:
            log.warning("no live price for %s: %s", symbol, exc)
    return prices
