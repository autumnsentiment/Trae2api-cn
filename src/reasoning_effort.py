"""Map OpenAI/Anthropic thinking-strength parameters to Trae's native levels.

The Trae client stores the level on ``custom_model``: ``reasoning_effort_level``
when the model object already carries that key, otherwise ``reasoning_effort``
(never both).  Levels are ``light`` / ``high`` / ``extra_high`` and the value
is clamped to the options advertised in ``reasoning_effort_config``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

LEVEL_ORDER = ("light", "high", "extra_high")

_ALIASES = {
    "none": "light",
    "minimal": "light",
    "min": "light",
    "low": "light",
    "light": "light",
    "medium": "high",
    "mid": "high",
    "normal": "high",
    "default": "high",
    "high": "high",
    "xhigh": "extra_high",
    "x-high": "extra_high",
    "extra_high": "extra_high",
    "extra-high": "extra_high",
    "very_high": "extra_high",
    "max": "extra_high",
    "maximum": "extra_high",
}


def normalize_level(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return _ALIASES.get(text, "")


def _budget_level(budget: Any) -> str:
    try:
        tokens = int(budget)
    except (TypeError, ValueError):
        return ""
    if tokens <= 0:
        return ""
    if tokens < 4096:
        return "light"
    if tokens < 16384:
        return "high"
    return "extra_high"


def requested_level(options: Optional[Mapping[str, Any]]) -> str:
    """Return the requested native level from chat/responses/anthropic options."""

    if not isinstance(options, Mapping):
        return ""
    for key in ("reasoning_effort", "reasoningEffort", "thinking_effort"):
        level = normalize_level(options.get(key))
        if level:
            return level
    reasoning = options.get("reasoning")
    if isinstance(reasoning, Mapping):
        level = normalize_level(reasoning.get("effort"))
        if level:
            return level
    thinking = options.get("thinking")
    if isinstance(thinking, Mapping):
        level = normalize_level(thinking.get("effort") or thinking.get("level"))
        if level:
            return level
        level = _budget_level(thinking.get("budget_tokens") or thinking.get("budgetTokens"))
        if level:
            return level
    return ""


def supported_levels(custom_model: Optional[Mapping[str, Any]]) -> list[str]:
    if not isinstance(custom_model, Mapping):
        return []
    config = custom_model.get("reasoning_effort_config")
    if not isinstance(config, Mapping) or not config.get("support_thinking"):
        return []
    options = config.get("options") or []
    levels = [normalize_level(item) for item in options if normalize_level(item)]
    return [level for level in LEVEL_ORDER if level in levels]


def clamp_level(level: str, supported: list[str]) -> str:
    """Highest supported level that is <= the requested one (else the lowest)."""

    if not level or not supported:
        return ""
    rank = LEVEL_ORDER.index(level)
    eligible = [item for item in supported if LEVEL_ORDER.index(item) <= rank]
    return eligible[-1] if eligible else supported[0]


def apply_reasoning_effort(
    custom_model: Optional[Mapping[str, Any]],
    options: Optional[Mapping[str, Any]],
) -> tuple[Optional[dict[str, Any]], str]:
    """Return ``(custom_model_copy, applied_level)``.

    The model object is returned unchanged (and level ``""``) when the caller
    did not ask for a strength or the model does not support thinking levels.
    """

    if not isinstance(custom_model, Mapping):
        return (dict(custom_model) if custom_model else custom_model), ""
    level = clamp_level(requested_level(options), supported_levels(custom_model))
    if not level:
        return dict(custom_model), ""
    updated = dict(custom_model)
    if "reasoning_effort_level" in updated:
        updated["reasoning_effort_level"] = level
        updated.pop("reasoning_effort", None)
    else:
        updated["reasoning_effort"] = level
        updated.pop("reasoning_effort_level", None)
    return updated, level