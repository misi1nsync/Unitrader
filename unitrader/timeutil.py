"""Parse duration strings such as "15m", "1h", "30d"."""

import re

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str) -> int:
    """Return the number of seconds in a duration like "1h" or "30d"."""
    match = re.fullmatch(r"\s*(\d+)\s*([smhdw])\s*", text)
    if not match:
        raise ValueError(f"invalid duration: {text!r} (expected e.g. '15m', '1h', '30d')")
    value, unit = match.groups()
    seconds = int(value) * _UNITS[unit]
    if seconds <= 0:
        raise ValueError(f"duration must be positive: {text!r}")
    return seconds
