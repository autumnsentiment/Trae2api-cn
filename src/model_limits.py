"""Shared safety limits for model completion requests."""

from __future__ import annotations

import os
from typing import Any, Optional


# Trae SOLO CN agent-remote models cap single responses at 64K tokens
# (report: solo_agent_remote max_tokens=64000). Keep the local clamp below
# that ceiling so a client asking for 131072 cannot push an upstream 4xx.
DEFAULT_MAX_COMPLETION_TOKENS = 64000

_AGENT_MODEL_MAX_TOKENS = 64000
_AGENT_MAX_MODELS = {
    "glm-5.1",
    "glm-5.2",
    "glm-5.3",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-pro-official",
    "deepseek-v4-flash-official",
}


def _configured_limit(model: Optional[str] = None) -> int:
    raw = os.environ.get("TRAE_MAX_COMPLETION_TOKENS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_COMPLETION_TOKENS
    if value <= 0:
        value = DEFAULT_MAX_COMPLETION_TOKENS
    normalized = str(model or "").strip().lower()
    if normalized.startswith("trae/"):
        normalized = normalized[5:]
    if normalized in _AGENT_MAX_MODELS:
        return min(value, _AGENT_MODEL_MAX_TOKENS)
    return value


def clamp_max_completion_tokens(value: Any, model: Optional[str] = None) -> Any:
    """Clamp positive completion-token values to the upstream-safe limit."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if value <= 0:
        return value
    return min(int(value), _configured_limit(model))
