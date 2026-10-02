"""In-process event bus used by @loop(trigger=...)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

log = logging.getLogger(__name__)

_handlers: dict[str, list[Callable[[], object]]] = defaultdict(list)


def subscribe(event: str, handler: Callable[[], object]) -> None:
    if handler not in _handlers[event]:
        _handlers[event].append(handler)


def emit(event: str) -> int:
    """Run every handler for ``event`` in order; return how many ran.

    A failing handler is logged and does not stop the others or the emitter.
    """
    handlers = list(_handlers.get(event, ()))
    for handler in handlers:
        try:
            handler()
        except Exception:
            log.exception("handler %r for %r failed", getattr(handler, "__name__", handler), event)
    return len(handlers)


def clear() -> None:
    _handlers.clear()
