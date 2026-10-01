"""Runtime configuration, overridable through environment variables."""

import os

# Top Bybit USDT linear perpetuals by liquidity.
DEFAULT_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


universe = _env_list("UNITRADER_UNIVERSE", DEFAULT_UNIVERSE)
state_dir = os.environ.get("UNITRADER_STATE_DIR", "data")
interval = os.environ.get("UNITRADER_INTERVAL", "1h")
lookback = os.environ.get("UNITRADER_LOOKBACK", "30d")
