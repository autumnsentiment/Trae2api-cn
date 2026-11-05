"""Recover missing cache counters from the same upstream turn's bill."""

import asyncio
import logging
import math
import os
from collections.abc import Mapping

from . import trae_client
from .sse import _map_usage, _merge_usage, _normalize_event_name
from .token_usage import usage_details

logger = logging.getLogger(__name__)


def _lookup_timeout() -> float:
    try:
        value = float(os.environ.get('TRAE_CACHE_USAGE_LOOKUP_TIMEOUT_SECONDS', '3'))
        return max(0.0, min(value, 10.0)) if math.isfinite(value) else 3.0
    except ValueError:
        return 3.0


async def enrich_usage(usage, token: str, turn_id: str, *, match_completion=False):
    """Supplement cache details only when the isolated turn bill matches."""
    details = usage_details(usage or {})
    timeout = _lookup_timeout()
    if (not token or not turn_id or not usage or usage.get("prompt_tokens", 0) <= 0
            or details.get("cached_tokens") or details.get("cache_write_tokens")
            or timeout <= 0):
        return usage
    deadline = asyncio.get_running_loop().time() + timeout
    for delay in (0.0, 0.25, 0.75):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= delay:
            break
        if delay:
            await asyncio.sleep(delay)
        try:
            bill = await asyncio.wait_for(
                trae_client.fetch_session_usage(turn_id, token),
                timeout=deadline - asyncio.get_running_loop().time(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("cache usage lookup unavailable: %s", type(exc).__name__)
            break
        if not isinstance(bill, Mapping):
            continue
        bill_details = usage_details(bill)
        prompt = usage["prompt_tokens"]
        if (bill.get("prompt_tokens") != prompt
                or match_completion and ("completion_tokens" not in bill
                                         or bill["completion_tokens"] != usage.get("completion_tokens"))
                or not any(key in bill_details for key in ("cached_tokens", "cache_write_tokens"))
                or bill_details.get("cached_tokens", 0)
                + bill_details.get("cache_write_tokens", 0) > prompt):
            continue
        # Explicit zero is valid. Repeated input is not evidence of a hit.
        recovered = dict(usage)
        recovered["prompt_tokens_details"] = {
            key: bill_details[key]
            for key in ("cached_tokens", "cache_write_tokens")
            if key in bill_details
        }
        return recovered
    return usage


def legacy_usage_enricher(response_wrapper, upstream_metadata):
    """Capture only an adapter-verified per-request billing identity."""
    turn_id = getattr(response_wrapper, "usage_turn_id", "")
    token = getattr(response_wrapper, "auth_token", "")
    if not isinstance(turn_id, str) or not turn_id or not isinstance(token, str) or not token:
        return None
    async def enrich(usage):
        upstream_metadata["usage_turn_id"] = turn_id
        return await enrich_usage(usage, token, turn_id, match_completion=True)

    return enrich


async def enrich_remote_events(events, token: str, turn_id: str):
    """Emit verified cache counters before the public terminal usage frame.

    Since October 27, Remote can report zero cache counters while its bill
    reports a hit. Use only the bound turn's bill with matching input totals;
    keep the original response when billing is unavailable or still settling.
    """
    usage = None
    source_closed = False

    async def close_source():
        nonlocal source_closed
        if not source_closed:
            source_closed = True
            close = getattr(events, 'aclose', None)
            if close is not None:
                await close()

    try:
        async for event, data in events:
            name = _normalize_event_name(event)
            if name == 'error':
                await close_source()
                yield event, data
                return
            if isinstance(data, Mapping):
                if name == 'token_usage':
                    usage = _merge_usage(usage, _map_usage(data.get('usage') or data))
                elif isinstance(data.get('usage'), dict):
                    usage = _merge_usage(usage, _map_usage(data['usage']))
                elif name == 'done':
                    usage = _merge_usage(usage, _map_usage(dict(data)))
            if name == 'done':
                # Consumers stop at done without advancing this generator.
                await close_source()
                recovered = await enrich_usage(usage, token, turn_id)
                if recovered is not usage:
                    yield 'token_usage', recovered
                yield event, data
                return
            yield event, data
    finally:
        await close_source()
