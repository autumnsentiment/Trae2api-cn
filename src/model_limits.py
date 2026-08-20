"""Shared safety limits for model completion requests."""

from __future__ import annotations

import os
from typing import Any, Optional


DEFAULT_MAX_COMPLETION_TOKENS = 131072


def _configured_limit() -> int:
    raw = os.environ.get("TRAE_MAX_COMPLETION_TOKENS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_COMPLETION_TOKENS
    return value if value > 0 else DEFAULT_MAX_COMPLETION_TOKENS


def clamp_max_completion_tokens(value: Any, model: Optional[str] = None) -> Any:
    """Clamp positive completion-token values to the upstream-safe limit."""

    del model
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if value <= 0:
        return value
    return min(int(value), _configured_limit())
