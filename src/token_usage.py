"""Token detail aliases shared by the public protocol translators."""

import math
from typing import Any, Iterator, Mapping


_NESTED_USAGE_KEYS = (
    "usage",
    "token_usage",
    "tokenUsage",
    "metrics",
    "data",
    "cache",
    "prompt_cache",
    "promptCache",
)


def _usage_sources(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Iterator[Mapping[str, Any]]:
    """Yield usage-shaped mappings without letting wrapper frames hide details.

    Trae has emitted both a direct ``token_usage`` object and a wrapped form
    (``{data: {usage: ...}}``) across endpoint generations.  Keep the direct
    object first so an explicit zero remains authoritative, then inspect only
    known usage/cache wrappers instead of recursively walking arbitrary model
    output.
    """

    if not isinstance(value, Mapping) or depth > 3:
        return
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)
    try:
        yield value
        for key in _NESTED_USAGE_KEYS:
            nested = value.get(key)
            if isinstance(nested, Mapping):
                yield from _usage_sources(nested, depth=depth + 1, seen=seen)
    finally:
        seen.remove(marker)


def usage_details(usage: Mapping[str, Any]) -> dict[str, int]:
    """Preserve reported details, including explicit zero, without estimates."""

    details: dict[str, int] = {}
    fields = (
        (
            "cached_tokens",
            (
                "cached_tokens",
                "cachedTokens",
                "cache_read_tokens",
                "cacheReadTokens",
                "cache_read_input_tokens",
                "cacheReadInputTokens",
                "prompt_cache_hit_tokens",
                "promptCacheHitTokens",
                "prompt_cache_read_tokens",
                "promptCacheReadTokens",
                "cached_input_tokens",
                "input_cached_tokens",
                "cache_read",
                "read_tokens",
                "cache_hits",
                "hit_tokens",
            ),
            ("prompt_tokens_details", "input_tokens_details"),
        ),
        (
            "cache_write_tokens",
            (
                "cache_write_tokens",
                "cacheWriteTokens",
                "cache_creation_input_tokens",
                "cacheCreationInputTokens",
                "prompt_cache_write_tokens",
                "promptCacheWriteTokens",
                "prompt_cache_miss_tokens",
                "promptCacheMissTokens",
                "cache_write",
                "write_tokens",
                "cache_misses",
                "miss_tokens",
            ),
            ("prompt_tokens_details", "input_tokens_details"),
        ),
        (
            "reasoning_tokens",
            ("reasoning_tokens", "reasoningTokens"),
            ("completion_tokens_details", "output_tokens_details"),
        ),
    )
    for field, aliases, containers in fields:
        sources: list[Mapping[str, Any]] = []
        for source in _usage_sources(usage):
            if source not in sources:
                sources.append(source)
        for source in list(sources):
            for key in containers:
                nested = source.get(key)
                if isinstance(nested, Mapping) and nested not in sources:
                    sources.extend(_usage_sources(nested))
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for alias in aliases:
                value = source.get(alias)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and (not isinstance(value, float) or math.isfinite(value))
                ):
                    details[field] = max(0, int(value))
                    break
            if field in details:
                break
    return details
