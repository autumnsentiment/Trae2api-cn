"""Shared safety limits for model completion requests."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Optional


# Trae SOLO CN agent-remote models cap single responses at 64K tokens
# (report: solo_agent_remote max_tokens=64000). Keep the local clamp below
# that ceiling so a client asking for 131072 cannot push an upstream 4xx.
DEFAULT_MAX_COMPLETION_TOKENS = 64000

# The fallback order is only a preference among descriptors already returned
# by Trae.  It is deliberately not a capability allow-list: a model name does
# not prove that the selected account/function can read images.
MULTIMODAL_FALLBACK_ORDER = (
    "kimi-k2.6",
    "kimi-k3.1-pro",
    "Doubao-Seed-2.1-Pro",
    "qwen3.8-max",
    "glm-5v-turbo",
    "minimax-m3",
    "kimi-k3",
    "Doubao-Seed-Code",
)


def _explicit_bool(value: Any) -> Optional[bool]:
    """Parse a descriptor flag while preserving an absent/unknown value."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def _mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
        try:
            return _positive_int(json.loads(value))
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        parsed_values = [
            parsed
            for item in value
            if (parsed := _positive_int(item)) is not None
        ]
        return max(parsed_values) if parsed_values else None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def descriptor_multimodal(descriptor: Any) -> Optional[bool]:
    """Return the live Trae image capability, or ``None`` when unspecified.

    Catalog revisions have used top-level ``multimodal`` and
    ``support_read_image`` as well as ``features.multimodal.enable``.  An
    explicit false must win over a true value in a less authoritative nested
    field; otherwise the first field present is used.  No model-name heuristic
    is consulted here.
    """

    if not isinstance(descriptor, Mapping):
        return None
    for key in ("multimodal", "support_read_image"):
        value = _explicit_bool(descriptor.get(key))
        if value is not None:
            return value
    features = _mapping(descriptor.get("features"))
    if features is not None:
        nested = features.get("multimodal")
        if isinstance(nested, Mapping):
            value = _explicit_bool(nested.get("enable"))
        else:
            value = _explicit_bool(nested)
        if value is not None:
            return value
    display_config = _mapping(descriptor.get("display_config"))
    if display_config is not None:
        value = _explicit_bool(display_config.get("multimodal"))
        if value is not None:
            return value
    return None


def is_multimodal_model(
    model: Optional[str] = None, descriptor: Any = None
) -> bool:
    """Whether the *selected descriptor* supports image input.

    ``model`` is retained for API compatibility and diagnostics only.  It is
    intentionally never used as a capability signal; callers must pass the
    descriptor returned for the bound account/function/tier.
    """

    return descriptor_multimodal(descriptor) is True


def descriptor_reasoning_config(descriptor: Any) -> Optional[dict[str, Any]]:
    """Return the current function/tier's reasoning descriptor, if any."""

    if not isinstance(descriptor, Mapping):
        return None
    config = _mapping(
        descriptor.get("reasoning_effort_config")
        or descriptor.get("reasoningEffortConfig")
    )
    if config is None:
        return None
    normalized = dict(config)
    support = _explicit_bool(
        normalized.get("support_thinking")
        if "support_thinking" in normalized
        else normalized.get("supportThinking")
    )
    if support is not None:
        normalized["support_thinking"] = support
    if "options" not in normalized:
        options = normalized.get("reasoning_effort_options")
        if options is None:
            options = normalized.get("reasoningEffortOptions")
        if options is not None:
            normalized["options"] = options
    options = normalized.get("options")
    if isinstance(options, str) and options.lstrip().startswith("["):
        try:
            parsed_options = json.loads(options)
        except (TypeError, ValueError):
            parsed_options = None
        if isinstance(parsed_options, list):
            normalized["options"] = parsed_options
    return normalized


def descriptor_context_limits(descriptor: Any) -> dict[str, Any]:
    """Extract only limits explicitly present in a live model descriptor.

    This function never manufactures 1M/936K/64K values.  It is used by both
    Remote and native transports so an absent max profile stays absent.
    """

    if not isinstance(descriptor, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "max_mode",
        "context_window_size",
        "context_window_sizes",
        "context_window_tokens",
        "prompt_max_tokens",
        "max_tokens",
        "max_turn",
        "max_turns",
    ):
        if key in descriptor and descriptor.get(key) is not None:
            result[key] = descriptor.get(key)
    features = _mapping(descriptor.get("features"))
    feature_windows = (
        _mapping(features.get("context_windows"))
        if features is not None
        else None
    )
    if feature_windows is not None:
        result["context_windows"] = dict(feature_windows)
    return result


def _descriptor_details(descriptor: Any) -> list[Mapping[str, Any]]:
    """Return the provider details carried by a live catalog descriptor."""

    if not isinstance(descriptor, Mapping):
        return []
    details = descriptor.get("_traecode_model_detail_list")
    if not isinstance(details, list):
        details = descriptor.get("model_detail_list")
    if not isinstance(details, list):
        return []
    return [
        item
        for item in details
        if isinstance(item, Mapping)
        and str(item.get("model_name") or "").strip()
    ]


def _detail_with_suffix(
    descriptor: Any, suffix: str
) -> Optional[Mapping[str, Any]]:
    suffix = str(suffix or "").strip().casefold()
    if not suffix:
        return None
    return next(
        (
            detail
            for detail in _descriptor_details(descriptor)
            if str(detail.get("model_name") or "").strip().casefold().endswith(
                suffix
            )
        ),
        None,
    )


def _window_value(value: Any, profile: str) -> Optional[int]:
    """Read one context-window value without treating a list as a scalar."""

    profile = str(profile or "dev").strip().casefold()
    mapping = _mapping(value)
    if mapping is not None:
        keys = (
            ("dev", "default", "base", "min")
            if profile == "dev"
            else ("max", "maximum", "max_context")
        )
        for key in keys:
            parsed = _positive_int(mapping.get(key))
            if parsed is not None:
                return parsed
        # Some catalog revisions put the values under a nested data object.
        data = _mapping(mapping.get("data"))
        if data is not None:
            return _window_value(data, profile)
        return None
    if isinstance(value, (list, tuple)):
        values = [
            parsed
            for item in value
            if (parsed := _positive_int(item)) is not None
        ]
        if not values:
            return None
        # ``context_window_tokens.max`` is occasionally [dev, max].  The
        # last value is the Max window; a dev list uses its first value.
        return values[-1] if profile == "max" else values[0]
    return _positive_int(value)


def descriptor_context_window(
    descriptor: Any, profile: str = "dev"
) -> Optional[int]:
    """Return the live context window for ``dev`` or ``max`` mode."""

    if not isinstance(descriptor, Mapping):
        return None
    profile = str(profile or "dev").strip().casefold()
    limits = descriptor_context_limits(descriptor)
    candidates: list[int] = []
    for key in ("context_window_tokens", "context_window_size", "context_window_sizes"):
        value = _window_value(limits.get(key), profile)
        if value is not None:
            candidates.append(value)
    windows = _mapping(limits.get("context_windows"))
    if windows is not None:
        data = _mapping(windows.get("data"))
        if data is not None:
            for key in (
                ("dev", "default", "context_window")
                if profile == "dev"
                else ("max", "max_context"),
            ):
                value = _window_value(data.get(key), profile)
                if value is not None:
                    candidates.append(value)
    if candidates:
        # There may be both token and size representations.  The largest
        # explicit value is the least surprising when they agree; for dev it
        # also handles a descriptor that only exposes one representation.
        return max(candidates)
    return None


def descriptor_output_tokens(
    descriptor: Any, profile: str = "dev"
) -> Optional[int]:
    """Return the output ceiling advertised by a descriptor/profile."""

    if not isinstance(descriptor, Mapping):
        return None
    profile = str(profile or "dev").strip().casefold()
    detail = _detail_with_suffix(
        descriptor, "__max" if profile == "max" else "__dev"
    )
    if detail is not None:
        value = _positive_int(detail.get("max_tokens"))
        if value is not None:
            return value
    return _positive_int(descriptor.get("max_tokens"))


def descriptor_max_profile(descriptor: Any) -> dict[str, Any]:
    """Return a verified Max profile, or an empty mapping.

    The profile is intentionally derived only from live descriptor fields.
    In particular, this helper never invents a 1M context or a completion
    budget for a model whose catalog did not provide them.
    """

    if not isinstance(descriptor, Mapping):
        return {}
    max_detail = _detail_with_suffix(descriptor, "__max")
    max_context = descriptor_context_window(descriptor, "max")
    if max_context is None:
        max_context = descriptor_max_context(descriptor)
    max_tokens = (
        _positive_int(max_detail.get("max_tokens"))
        if max_detail is not None
        else None
    ) or descriptor_output_tokens(descriptor, "max")
    prompt_max_tokens = _positive_int(descriptor.get("prompt_max_tokens"))
    if prompt_max_tokens is None and max_context and max_tokens:
        prompt_max_tokens = max_context - max_tokens
        if prompt_max_tokens <= 0:
            prompt_max_tokens = None
    if not max_context or not max_tokens:
        return {}
    if max_detail is None and _explicit_bool(descriptor.get("max_mode")) is not True:
        return {}
    if max_context < 1_000_000:
        return {}
    dev_context = descriptor_context_window(descriptor, "dev")
    if dev_context is not None and max_context <= dev_context:
        return {}
    return {
        "model_name": str(max_detail.get("model_name") or "").strip()
        if max_detail is not None
        else "",
        "context": int(max_context),
        "output": int(max_tokens),
        "prompt": int(prompt_max_tokens) if prompt_max_tokens else None,
        "base_context": int(dev_context) if dev_context else None,
    }


def descriptor_max_context(descriptor: Any) -> Optional[int]:
    """Return the explicitly advertised max context for this descriptor."""

    limits = descriptor_context_limits(descriptor)
    candidates: list[int] = []
    for key in ("context_window_size", "context_window_sizes"):
        value = limits.get(key)
        value_map = _mapping(value)
        if value_map is not None:
            value = value_map.get("max") or value_map.get("max_context")
        parsed = _positive_int(value)
        if parsed is not None:
            candidates.append(parsed)
    tokens = _mapping(limits.get("context_window_tokens"))
    if tokens is not None:
        parsed = _positive_int(tokens.get("max"))
        if parsed is not None:
            candidates.append(parsed)
    windows = _mapping(limits.get("context_windows"))
    if windows is not None:
        data = _mapping(windows.get("data"))
        if data is not None:
            for key in ("max_context", "max_context_list"):
                parsed = _positive_int(data.get(key))
                if parsed is not None:
                    candidates.append(parsed)
    return max(candidates) if candidates else None


def descriptor_supports_max_mode(descriptor: Any) -> bool:
    """Return true only for a live descriptor with a verified 1M Max profile."""

    return bool(descriptor_max_profile(descriptor))


def descriptor_access_identities(descriptor: Any) -> tuple[int, ...] | None:
    """Return the live access identity allow-list, or ``None`` if absent.

    Trae advertises ``features.access.data.identity_list`` on each model
    descriptor.  An empty/malformed list is not interpreted as an allow-list:
    only a concrete integer array can prove that a pay identity is denied.
    """

    if not isinstance(descriptor, Mapping):
        return None
    candidates: list[Any] = []
    for key in ("identity_list", "identityList"):
        if key in descriptor:
            candidates.append(descriptor.get(key))
    for container_key in ("features", "feature", "access"):
        container = _mapping(descriptor.get(container_key))
        if container is None:
            continue
        for key in ("identity_list", "identityList"):
            if key in container:
                candidates.append(container.get(key))
        features = _mapping(container.get("features"))
        if features is not None:
            for key in ("identity_list", "identityList"):
                if key in features:
                    candidates.append(features.get(key))
        access = _mapping(container.get("access"))
        if access is None:
            access = features
        if access is not None:
            for key in ("identity_list", "identityList"):
                if key in access:
                    candidates.append(access.get(key))
            data = _mapping(access.get("data"))
            if data is not None:
                for key in ("identity_list", "identityList"):
                    if key in data:
                        candidates.append(data.get(key))
    features = _mapping(descriptor.get("features"))
    access = _mapping(features.get("access")) if features is not None else None
    data = _mapping(access.get("data")) if access is not None else None
    if data is not None:
        for key in ("identity_list", "identityList"):
            if key in data:
                candidates.append(data.get(key))

    raw = next((value for value in candidates if value not in (None, "")), None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, (list, tuple)):
        return None
    identities: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        identities.append(parsed)
    return tuple(identities) if identities else None


def descriptor_denies_pay_identity(
    descriptor: Any, pay_identity: Any
) -> bool:
    """Whether a live descriptor explicitly excludes this pay identity.

    Unknown identity evidence is never treated as denial.  This keeps catalog
    revisions without an access list backwards compatible while allowing the
    relay to fail fast when both sides are known.
    """

    if pay_identity is None or isinstance(pay_identity, bool):
        return False
    if isinstance(pay_identity, str):
        text_identity = pay_identity.strip()
        if text_identity.casefold() == "free":
            pay_identity = 0
    try:
        identity = int(str(pay_identity).strip())
    except (TypeError, ValueError):
        return False
    identities = descriptor_access_identities(descriptor)
    return identities is not None and identity not in identities


def configured_multimodal_model() -> str:
    return str(os.environ.get("TRAE_MULTIMODAL_MODEL", "")).strip()


def _configured_limit(model: Optional[str] = None, descriptor: Any = None) -> int:
    raw = os.environ.get("TRAE_MAX_COMPLETION_TOKENS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_COMPLETION_TOKENS
    if value <= 0:
        value = DEFAULT_MAX_COMPLETION_TOKENS
    if isinstance(descriptor, Mapping):
        raw_max = descriptor.get("max_tokens")
        try:
            max_tokens = int(raw_max)
        except (TypeError, ValueError):
            max_tokens = 0
        if max_tokens > 0:
            return min(value, max_tokens)
    return value


def clamp_max_completion_tokens(
    value: Any, model: Optional[str] = None, descriptor: Any = None
) -> Any:
    """Clamp positive completion-token values to the upstream-safe limit."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if value <= 0:
        return value
    return min(int(value), _configured_limit(model, descriptor))
