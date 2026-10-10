"""
sse.py - parse Trae upstream events into OpenAI-compatible responses.

Tool calls are always serialized back to the API client. This module never
executes a tool, command, or filesystem operation.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncIterator, Optional

from .cli_client import extract_result_text as _cli_extract_text
from .cli_client import complete_tool_call_signature as _complete_tool_call_signature
from .cli_client import deduplicate_tool_calls as _deduplicate_tool_calls
from .cli_client import extract_tool_calls as _extract_tool_calls
from .cli_client import extract_usage as _cli_extract_usage
from .cli_client import normalize_tool_call as _normalize_tool_call
from .cli_client import strip_tool_call_blocks as _strip_tool_call_blocks
from .cli_client import tool_call_signature as _tool_call_signature

logger = logging.getLogger(__name__)

THINK_OPEN = " thinking\n\n"
THINK_CLOSE = "\n response\n\n"

# A synchronous httpx.Response is still used by the raw/IDE compatibility
# clients.  Reading its iterator directly inside an async generator blocks
# uvicorn's event loop whenever Trae pauses between SSE frames.  Keep the
# blocking read in a worker thread and use an SSE comment as a wire-safe
# keepalive while waiting.  Comments are ignored by OpenAI-compatible clients
# and therefore never become assistant text or Responses events.
_STREAM_HEARTBEAT = object()
_STREAM_HEARTBEAT_LINE = ": relay-keepalive\n\n"


def _stream_heartbeat_seconds() -> float:
    try:
        value = float(os.environ.get("SSE_HEARTBEAT_SECONDS", "1"))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, value)


def _next_sync_item(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def _iter_stream_lines(response) -> AsyncIterator[Any]:
    """Consume sync or async upstream lines without blocking the event loop."""

    if hasattr(response, "__aiter__"):
        async for line in response:
            yield line
        return

    source = response.iter_lines() if hasattr(response, "iter_lines") else response
    iterator = iter(source)
    heartbeat = _stream_heartbeat_seconds()
    read_task = None
    try:
        while True:
            if read_task is None:
                read_task = asyncio.create_task(
                    asyncio.to_thread(_next_sync_item, iterator)
                )
            if heartbeat > 0:
                try:
                    has_item, line = await asyncio.wait_for(
                        asyncio.shield(read_task), heartbeat
                    )
                except asyncio.TimeoutError:
                    yield _STREAM_HEARTBEAT
                    continue
            else:
                has_item, line = await read_task
            read_task = None
            if not has_item:
                return
            yield line
    finally:
        if read_task is not None and not read_task.done():
            read_task.cancel()


class EmptyUpstreamResponse(RuntimeError):
    """Raised before any chunks are emitted when an upstream turn is empty."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        usage: Optional[dict] = None,
        observed_model_event: bool = False,
        account_quota_exhausted: bool = False,
    ):
        super().__init__(message)
        self.retryable = bool(retryable)
        self.usage = dict(usage) if isinstance(usage, dict) else None
        self.observed_model_event = bool(observed_model_event)
        # A quota response is account-scoped.  The dispatcher may rotate to
        # another credential, but must never replay it through the same
        # account's Work fallback after a streaming Agent attempt.
        self.account_quota_exhausted = bool(account_quota_exhausted)


class IncompleteUpstreamResponse(EmptyUpstreamResponse):
    """Raised when an upstream stream ends without a terminal event.

    Trae can emit a cumulative response snapshot with a ``stop_reason`` before
    it has sent the actual terminal SSE event.  Treating that snapshot as the
    end of the stream silently drops the remaining answer.  This exception is
    an ``EmptyUpstreamResponse`` subclass so existing retry paths can recover
    from a truncated upstream attempt.
    """


class RepeatedCompletedToolResponse(EmptyUpstreamResponse):
    """Raised when the only upstream output repeats an already completed call."""


class InvalidNativeToolArguments(EmptyUpstreamResponse):
    """Raised when native tool fragments do not form valid final JSON."""


class ModelProviderMismatch(RuntimeError):
    """Raised when Trae reports a provider different from the requested model."""


_EVENT_NAME_ALIASES = {
    "keepalive": "keepalive",
    "modelconfig": "model_config",
    "planitem": "plan_item",
    "requestwaitinqueue": "request_wait_in_queue",
    "responsedone": "done",
    "streamdone": "done",
    "tokenusage": "token_usage",
}


def _normalize_event_name(value: Any) -> str:
    """Normalize raw, remote, and native event spellings to snake_case."""

    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return _EVENT_NAME_ALIASES.get(text.replace("_", ""), text)


def _payload_event_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("event", "event_type", "eventType", "type"):
        value = payload.get(key)
        if value not in (None, ""):
            return _normalize_event_name(value)
    return ""


def _usage_turn_id(payload: Any) -> str:
    """Extract Trae's billable user-message id without exposing it downstream."""

    if not isinstance(payload, dict):
        return ""
    for key in (
        "reply_to_message_id",
        "replyToMessageId",
        "user_message_id",
        "userMessageId",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value).strip()
    for key in ("model_config", "modelConfig", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            value = _usage_turn_id(nested)
            if value:
                return value
    return ""


def _capture_upstream_metadata(target: Any, payload: Any) -> None:
    if not isinstance(target, dict):
        return
    turn_id = _usage_turn_id(payload)
    if turn_id:
        target["usage_turn_id"] = turn_id


def _provider_model_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    direct = (
        payload.get("provider_model_name")
        or payload.get("providerModelName")
        or payload.get("model_provider_name")
        or payload.get("modelProviderName")
    )
    if direct:
        return str(direct).strip()
    timing = payload.get("timing_cost") or payload.get("timingCost")
    if isinstance(timing, dict):
        value = (
            timing.get("provider_model_name")
            or timing.get("providerModelName")
            or timing.get("model_provider_name")
            or timing.get("modelProviderName")
        )
        if value:
            return str(value).strip()
    model_config = payload.get("model_config") or payload.get("modelConfig")
    if isinstance(model_config, dict):
        value = (
            model_config.get("provider_model_name")
            or model_config.get("providerModelName")
            or model_config.get("model_provider_name")
            or model_config.get("modelProviderName")
            or model_config.get("model_name")
            or model_config.get("modelName")
        )
        if value:
            return str(value).strip()
    event_name = _payload_event_name(payload)
    if event_name == "model_config":
        value = payload.get("model_name") or payload.get("modelName")
        if value:
            return str(value).strip()
    for key in ("provider_model",):
        value = payload.get(key)
        if value:
            return str(value).strip()
    timing_data = payload.get("data")
    if isinstance(timing_data, dict):
        for key in (
            "provider_model_name",
            "providerModelName",
            "model_provider_name",
            "modelProviderName",
        ):
            value = timing_data.get(key)
            if value:
                return str(value).strip()
        if event_name == "model_config":
            value = timing_data.get("model_name") or timing_data.get("modelName")
            if value:
                return str(value).strip()
    return ""


def _is_account_quota_error(payload: Any) -> bool:
    """Return whether an upstream error represents account quota exhaustion."""

    if not isinstance(payload, dict):
        return False
    code = str(
        payload.get("code")
        or payload.get("error_code")
        or payload.get("errorCode")
        or ""
    ).strip().lower()
    text = " ".join(
        str(payload.get(key) or "")
        for key in ("message", "error", "detail", "reason")
    ).lower()
    return (
        code == "4008"
        or "requests have exceeded the quota" in text
        or "quota exceeded" in text
        or "exceeded the quota" in text
    )


def _model_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("trae/"):
        text = text[5:]
    # Runtime ids append build variants after a double underscore. They do not
    # represent a different public model family.
    text = text.split("__", 1)[0].replace("_", "-")
    # The raw gateway prefixes Alibaba-hosted variants with ``ali-``
    # (DeepSeek, Kimi, ...). It is a hosting marker, not a different model.
    if text.startswith("ali-"):
        text = text[4:]
    # Provider deployments may append ``Official`` and/or a release date while
    # retaining the same DeepSeek public model identity.
    deepseek = re.fullmatch(
        r"(deepseek-v4-(?:pro|flash))(?:-(?:official|\d{4,8}))*", text
    )
    if deepseek:
        return deepseek.group(1)
    if text.endswith("-official"):
        text = text[: -len("-official")]
    return text


_OPAQUE_ENDPOINT_RE = re.compile(r"^ep-\d{8,}-[a-z0-9]+$", re.IGNORECASE)


def _is_opaque_provider_id(value: Any) -> bool:
    """Return True for provider ids that do not name a model family.

    ``timing_events`` may report a Volcengine endpoint id (``ep-2026...-x``)
    or an encrypted base64 token instead of a model name.  Comparing those
    against the requested family produces false mismatches (and 502 loops).
    """

    text = str(value or "").strip()
    if not text:
        return True
    if _OPAQUE_ENDPOINT_RE.match(text):
        return True
    if any(ch in text for ch in "/+=") and not text.lower().startswith("trae/"):
        return True
    if len(text) >= 24 and re.fullmatch(r"[A-Za-z0-9_\-]+", text):
        has_sep = "-" in text or "_" in text or "." in text
        if not has_sep:
            return True
    return False


def _check_provider_model(requested: str, actual: str) -> None:
    if not actual or not requested:
        return
    if _is_opaque_provider_id(actual):
        return
    if str(os.environ.get("TRAE_STRICT_MODEL_MATCH", "true")).strip().lower() in {
        "0", "false", "no", "off"
    }:
        return
    # The public API accepts aliases (for example ``gpt-4o`` and
    # ``claude-sonnet-4``) that Trae maps to a concrete remote config. Compare
    # the provider against that effective config rather than the alias text.
    effective_requested = requested
    try:
        from .trae_client import convert_model_name

        effective_requested = convert_model_name(str(requested or "")) or requested
    except Exception:
        pass
    requested_family = _model_family(effective_requested)
    if requested_family in {"", "auto", "work", "auto-work", "solo-work"}:
        return
    actual_family = _model_family(actual)
    if requested_family == actual_family:
        return
    raise ModelProviderMismatch(
        f"Trae selected provider model {actual!r} for requested model {requested!r}"
    )


def make_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:20]}"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    total = 0.0
    for ch in text:
        total += 1.5 if ord(ch) > 0x2000 else 0.25
    return max(1, int(total + 0.999))


def openai_chunk(
    prefix_id: str,
    model: str,
    delta: dict,
    finish_reason=None,
    usage=None,
    error=None,
    provider_model_name: Optional[str] = None,
):
    chunk = {
        "id": prefix_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    if error is not None:
        chunk["error"] = error
    if provider_model_name:
        chunk["provider_model_name"] = provider_model_name
    return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"


def openai_completion(
    prefix_id: str,
    model: str,
    content: Optional[str],
    finish_reason: str = "stop",
    usage=None,
    tool_calls: Optional[list[dict]] = None,
    provider_model_name: Optional[str] = None,
    reasoning_content: Optional[str] = None,
):
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        public_calls = []
        for call in tool_calls:
            public_call = {
                key: value
                for key, value in call.items()
                if key != "index" and not key.startswith("_")
            }
            function = public_call.get("function")
            if isinstance(function, dict):
                function = dict(function)
                if function.get("arguments") == "":
                    function["arguments"] = "{}"
                public_call["function"] = function
            public_calls.append(public_call)
        message["tool_calls"] = public_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    resp = {
        "id": prefix_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    text = content or ""
    resp["usage"] = usage if usage is not None else {
        "prompt_tokens": 0,
        "completion_tokens": estimate_tokens(text),
        "total_tokens": estimate_tokens(text),
    }
    if provider_model_name:
        resp["provider_model_name"] = provider_model_name
    return resp


def _map_usage(usage: Any) -> Optional[dict]:
    if not isinstance(usage, dict):
        return None

    def number(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return max(0, int(value))
        return 0

    def optional_number(*keys: str) -> Optional[int | float]:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, value)
        return None

    prompt = number(
        "prompt_tokens",
        "input_tokens",
        "input_token",
        "inputTokens",
        "inputToken",
    )
    completion = number(
        "completion_tokens",
        "output_tokens",
        "output_token",
        "outputTokens",
        "outputToken",
    )
    total = number(
        "total_tokens", "total_token", "totalTokens", "totalToken"
    ) or prompt + completion
    mapped = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    credits = optional_number(
        "credits_consumed",
        "consumed_credits",
        "credit_cost",
        "credits_cost",
        "credits_float",
        "creditsFloat",
    )
    if credits is not None:
        mapped["credits_consumed"] = credits
    # Trae's raw stream reports prompt-cache hits (report 9.1:
    # cache_read_tokens / cache_write_tokens). Surface the read count in the
    # OpenAI-standard slot so a caller can see its cache savings.
    cached = optional_number(
        "cache_read_tokens",
        "cacheReadTokens",
        "cached_tokens",
        "cachedTokens",
        "cache_read_input_tokens",
    )
    if cached is not None:
        mapped["prompt_tokens_details"] = {"cached_tokens": int(cached)}
    reasoning = optional_number("reasoning_tokens", "reasoningTokens")
    if reasoning is not None:
        mapped["completion_tokens_details"] = {"reasoning_tokens": int(reasoning)}
    return mapped


def _payload_finish_reason(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for key in ("finish_reason", "stop_reason", "finishReason", "stopReason"):
        value = data.get(key)
        if value:
            return str(value)
    message = data.get("message")
    if isinstance(message, dict):
        for key in (
            "finish_reason",
            "stop_reason",
            "finishReason",
            "stopReason",
        ):
            value = message.get(key)
            if value:
                return str(value)
        response_meta = message.get("response_meta")
        if isinstance(response_meta, dict):
            for key in (
                "finish_reason",
                "stop_reason",
                "finishReason",
                "stopReason",
            ):
                value = response_meta.get(key)
                if value:
                    return str(value)
    return None


def _finish_reason(value: Any, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    reason = str(value or "stop").strip().lower()
    if reason in {"tool_calls", "tool-calls", "function_call", "tool_use", "tool"}:
        return "stop"
    if reason in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "max_token",
        "token_limit",
        # Trae reports an over-budget prompt/context as its own reason
        # (report: finish_reason=model_context_window_exceeded). Mapping it to
        # ``stop`` would tell the client a truncated answer is complete.
        "model_context_window_exceeded",
        "context_window_exceeded",
        "context_length_exceeded",
    }:
        return "length"
    if reason in {"content_filter", "content-filter", "safety"}:
        return "content_filter"
    if reason in {"stop", "end_turn", "eos", "done", "finished", "complete"}:
        return "stop"
    return "stop"


def _required_tool_label(tool_choice: Any) -> Optional[str]:
    if tool_choice == "required":
        return "one of the declared tools"
    if isinstance(tool_choice, dict):
        # Accept both Chat Completions' nested shape and the flat
        # Responses/TraeWork shape (``{type: function, name: ...}``).
        function = tool_choice.get("function")
        if not isinstance(function, dict):
            function = tool_choice
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    return None


def _call_function_name(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    function = call.get("function")
    if not isinstance(function, dict):
        function = call
    name = function.get("name") if isinstance(function, dict) else ""
    return str(name or "").strip()


def _required_tool_error(
    tool_choice: Any,
    has_tool_calls: bool,
    tool_calls: Any = None,
) -> Optional[dict]:
    required = _required_tool_label(tool_choice)
    if not required:
        return None
    if tool_choice == "required":
        satisfied = has_tool_calls
    elif tool_calls is None:
        # Keep the legacy boolean-only contract for callers that do not retain
        # the call list.  Callers with the accumulator should pass its calls so
        # a named choice can be checked precisely.
        satisfied = has_tool_calls
    else:
        satisfied = any(
            _call_function_name(call) == required
            for call in tool_calls
            if isinstance(call, dict)
        )
    if satisfied:
        return None
    return {
        "message": f"Trae upstream did not return the required tool call: {required}",
        "type": "api_error",
    }


def _ensure_required_tool_call(
    tool_choice: Any,
    has_tool_calls: bool,
    tool_calls: Any = None,
) -> None:
    error = _required_tool_error(tool_choice, has_tool_calls, tool_calls)
    if error:
        raise RuntimeError(error["message"])


class ToolCallAccumulator:
    """Merge native fragments or cumulative calls into OpenAI tool deltas."""

    def __init__(self, max_calls: Optional[int] = None):
        self._calls: dict[str, dict] = {}
        self._order: list[str] = []
        self._indexes: dict[str, int] = {}
        self._max_calls = max_calls

    @property
    def has_calls(self) -> bool:
        return bool(self._calls)

    def _existing_key(self, call: dict) -> Optional[str]:
        key = call.get("id")
        if key in self._calls:
            return key
        function = call.get("function") or {}
        requested_index = call.get("index")
        # Trae native stream: tool-call argument fragments. The first item of
        # each call carries a real id + name; follow-up fragments carry no id
        # and no name (normalize_tool_call then synthesizes a per-fragment
        # content-hash id, which must NOT be used as the merge key). The
        # upstream `index` is the stable cross-frame call ordinal (0, 1, ...),
        # so index-based merge is the correct path when present.
        if call.get("_synthetic_id") or not function.get("name"):
            if isinstance(requested_index, int):
                for existing_key, existing_index in self._indexes.items():
                    if existing_index != requested_index:
                        continue
                    existing_name = (self._calls[existing_key].get("function") or {}).get(
                        "name"
                    )
                    if not function.get("name") or function.get("name") == existing_name:
                        return existing_key
            # No usable index: nameless delta fragments are ordered, so merge
            # into the latest open delta call (upstream order == merge order).
            if call.get("_synthetic_id") and not function.get("name"):
                last_key = self._order[-1] if self._order else None
                if last_key is not None:
                    last = self._calls.get(last_key) or {}
                    last_function = last.get("function") or {}
                    if (
                        last.get("_arguments_mode") == "delta"
                        and last_function.get("name")
                    ):
                        return last_key
                return None
        if isinstance(requested_index, int) and (
            call.get("_synthetic_id") or not function.get("name")
        ):
            for existing_key, existing_index in self._indexes.items():
                if existing_index != requested_index:
                    continue
                existing_name = (self._calls[existing_key].get("function") or {}).get(
                    "name"
                )
                if not function.get("name") or function.get("name") == existing_name:
                    return existing_key
        return None

    @staticmethod
    def _complete_signature(call: Any) -> str:
        """Return a canonical signature only when arguments are valid JSON."""

        return _complete_tool_call_signature(call)

    @classmethod
    def _closed_state(cls, existing: dict, current: dict) -> str:
        """Classify a complete frame for an already closed logical call."""

        previous_signature = cls._complete_signature(existing)
        current_signature = cls._complete_signature(current)
        if not previous_signature or not current_signature:
            return ""
        if previous_signature == current_signature:
            return "replay"
        same_index = (
            existing.get("_explicit_index") is True
            and current.get("_explicit_index") is True
            and existing.get("index") == current.get("index")
        )
        if (
            existing.get("_explicit_id") is True
            or current.get("_explicit_id") is True
            or same_index
        ):
            return "conflict"
        return ""

    def prepare(self, calls: Any) -> list[dict]:
        if not isinstance(calls, list):
            return []
        prepared: list[dict] = []
        for fallback_index, raw in enumerate(calls):
            call = _normalize_tool_call(raw, index=fallback_index)
            if not call:
                continue
            function = call.get("function") or {}
            existing_key = self._existing_key(call)
            if existing_key is not None:
                existing = self._calls[existing_key]
                existing_function = existing.get("function") if existing else None
                if not function.get("name") and isinstance(existing_function, dict):
                    function["name"] = existing_function["name"]
                call["id"] = existing_key
                call["_synthetic_id"] = existing.get("_synthetic_id") is True
            prepared.append(call)
        return prepared

    def add(self, calls: Any) -> list[dict]:
        if not isinstance(calls, list):
            return []
        deltas: list[dict] = []
        for fallback_index, raw in enumerate(calls):
            call = _normalize_tool_call(raw, index=fallback_index)
            if not call:
                continue
            function = call["function"]
            call_id = call["id"]
            key = self._existing_key(call) or call_id
            if key not in self._calls and not function.get("name"):
                # 匹配不到任何已有调用的无 name 帧：只有以 { [ 开头才是新调用
                # 起点；孤立的 } ] 空串是上游并发流的遗留闭合帧，直接丢弃。
                first = str(function.get("arguments") or "").lstrip()[:1]
                if first not in ("{", "["):
                    import logging as _lg
                    _lg.getLogger("trae-cn-relay").warning(
                        "[junk-drop] orphan fragment %r",
                        str(function.get("arguments") or "")[:30],
                    )
                    continue
            existing = self._calls.get(key)
            if existing is not None:
                call_id = key
                call["id"] = key
            if not function.get("name"):
                existing_function = existing.get("function") if existing else None
                if not isinstance(existing_function, dict) or not existing_function.get(
                    "name"
                ):
                    continue
                function["name"] = existing_function["name"]
            if existing is not None:
                closed_state = self._closed_state(existing, call)
                if closed_state == "replay":
                    logger.debug(
                        "dropping repeated complete tool call id=%s",
                        existing.get("id"),
                    )
                    continue
                if closed_state == "conflict":
                    logger.warning(
                        "dropping conflicting complete tool call id=%s",
                        existing.get("id"),
                    )
                    continue
            requested_index = call.get("index")
            if key not in self._calls and call.get("_synthetic_id"):
                current_signature = self._complete_signature(call)
                if (
                    current_signature
                    and call.get("_explicit_index") is not True
                    and any(
                        existing_call.get("_synthetic_id") is True
                        and existing_call.get("_explicit_index") is not True
                        and self._complete_signature(existing_call) == current_signature
                        for existing_call in self._calls.values()
                    )
                ):
                    logger.debug(
                        "dropping repeated idless complete tool call signature=%s",
                        current_signature,
                    )
                    continue
                for existing_key in self._order:
                    existing = self._calls[existing_key]
                    if not existing.get("_synthetic_id"):
                        continue
                    existing_function = existing.get("function") or {}
                    if existing_function.get("name") != function.get("name"):
                        continue
                    both_indexed = (
                        existing.get("_explicit_index") is True
                        and call.get("_explicit_index") is True
                    )
                    same_index = both_indexed and self._indexes[existing_key] == requested_index
                    if both_indexed and not same_index:
                        continue
                    previous_args = str(existing_function.get("arguments") or "")
                    current_args = str(function.get("arguments") or "")
                    cumulative = current_args.startswith(previous_args) or previous_args.startswith(
                        current_args
                    )
                    if same_index or cumulative:
                        key = existing_key
                        call_id = existing["id"]
                        call["id"] = call_id
                        break
            if key not in self._calls:
                if self._max_calls is not None and len(self._order) >= self._max_calls:
                    continue
                index = requested_index if isinstance(requested_index, int) else len(self._order)
                self._calls[key] = call
                self._order.append(key)
                self._indexes[key] = index
                deltas.append({
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": function["arguments"],
                    },
                })
                continue

            previous = self._calls[key]["function"]
            current_name = function["name"]
            current_args = function["arguments"]
            previous_args = previous["arguments"]
            function_delta: dict[str, str] = {}
            if current_name and current_name != previous["name"]:
                previous["name"] = current_name
                function_delta["name"] = current_name
            if call.get("_arguments_mode") == "delta":
                # Trae 并发工具流会把多个调用的参数串在同一个 index 下连续发。
                # 当前 key 的 arguments 已是合法 JSON 时，新片段属于【新调用】，
                # 不是续片——闭合后新开，避免 `{"a":1}{"b":2}` 拼接。
                try:
                    json.loads(previous_args)
                    prev_closed = True
                except Exception:
                    prev_closed = False
                if prev_closed and current_args:
                    # 只有新片段以 { 或 [ 开头才是真正的新调用起点；
                    # 孤立的 } / ] / 空帧是上游遗留的闭合帧，忽略不拼。
                    first = current_args.lstrip()[:1]
                    if first in ("{", "["):
                        if (
                            existing.get("_explicit_id") is True
                            or call.get("_explicit_id") is True
                            or (
                                existing.get("_explicit_index") is True
                                and call.get("_explicit_index") is True
                                and existing.get("index") == call.get("index")
                            )
                        ):
                            logger.warning(
                                "dropping new delta object for closed logical call id=%s",
                                existing.get("id"),
                            )
                            continue
                        new_key = call_id + ":" + str(len(self._order))
                        self._calls[new_key] = {
                            "id": new_key,
                            "type": "function",
                            "function": {
                                "name": current_name or previous["name"],
                                "arguments": current_args,
                            },
                            "_synthetic_id": True,
                            "_arguments_mode": "delta",
                            "_source_index": call.get("_source_index", 0),
                        }
                        self._order.append(new_key)
                        self._indexes[new_key] = len(self._order) - 1
                        deltas.append(
                            {
                                "index": self._indexes[new_key],
                                "id": new_key,
                                "type": "function",
                                "function": {
                                    "name": current_name or previous["name"],
                                    "arguments": current_args,
                                },
                            }
                        )
                        continue
                    # 遗留闭合/空帧：丢弃，不污染已完整的前一个调用。
                    continue
                argument_delta = current_args
                previous["arguments"] += current_args
                if argument_delta:
                    function_delta["arguments"] = argument_delta
            elif current_args != previous_args:
                if current_args.startswith(previous_args):
                    argument_delta = current_args[len(previous_args):]
                    previous["arguments"] = current_args
                elif previous_args.startswith(current_args):
                    argument_delta = ""
                else:
                    # Some upstreams send true argument deltas rather than a
                    # cumulative string. Preserve that behavior for the client.
                    argument_delta = current_args
                    previous["arguments"] += current_args
                if argument_delta:
                    function_delta["arguments"] = argument_delta
            if function_delta:
                deltas.append({
                    "index": self._indexes[key],
                    "function": function_delta,
                })
        return deltas

    def calls(self) -> list[dict]:
        return [self._calls[key] for key in self._order]


    def set_calls(self, calls: list[dict]) -> None:
        """Replace the aggregated calls (used after repair/split)."""
        self._calls = {}
        self._order = []
        self._indexes = {}
        for idx, c in enumerate(calls):
            cid = str(c.get("id") or f"call_repair_{idx}")
            self._calls[cid] = c
            self._order.append(cid)
            call_index = c.get("index")
            if not isinstance(call_index, int):
                call_index = idx
            else:
                c.setdefault("index", call_index)
            self._indexes[cid] = call_index


def _repair_json_arguments(arguments: str) -> str:
    """Best-effort repair of a truncated native tool arguments string.

    Trae's native stream occasionally drops the final fragment(s) of a tool
    call, leaving arguments like ``{"command": "git status"`` without the
    closing ``}`` (or with a dangling quote). Never raise on malformed input;
    return the original string when we cannot repair confidently.
    """
    if not arguments:
        return arguments
    try:
        json.loads(arguments)
        return arguments
    except Exception:
        pass
    repaired = arguments.rstrip()
    stack = []
    in_str = False
    esc = False
    for ch in repaired:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_str:
        repaired += '"'
    closing = {"{": "}", "[": "]"}
    for opener in reversed(stack):
        repaired += closing[opener]
    try:
        json.loads(repaired)
        return repaired
    except Exception:
        return arguments


def _split_json_stream(text: str):
    """从 {..}{..} 拼接串里切出一个个合法 JSON 对象（raw_decode）。"""
    out = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        if text[i] not in "{[":
            return None
        try:
            obj, end = json.JSONDecoder().raw_decode(text, i)
        except Exception:
            return None
        out.append(obj)
        i = end
    return out


def _ensure_native_tool_arguments(
    calls: list[dict], usage: Any = None
) -> list[dict]:
    """Validate/repair/split malformed native delta tool arguments."""
    repaired_any = False
    kept: list[dict] = []
    for call in calls:
        if call.get("_arguments_mode") != "delta":
            kept.append(call)
            continue
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not arguments:
            kept.append(call)
            continue
        try:
            json.loads(arguments)
            kept.append(call)
            continue
        except (TypeError, ValueError):
            pass
        parts = _split_json_stream(arguments)
        if parts is not None:
            if len(parts) > 1:
                logger.warning(
                    "[split] call=%s split %d objects",
                    function.get("name") or "?",
                    len(parts),
                )
            seen_signatures: set[str] = set()
            for idx, obj in enumerate(parts):
                new_call = dict(call)
                new_fn = dict(function)
                new_fn["arguments"] = json.dumps(obj, ensure_ascii=False)
                new_call["function"] = new_fn
                signature = _tool_call_signature(new_call)
                if signature and signature in seen_signatures:
                    logger.warning(
                        "[split-drop] repeated object in one native stream call=%s",
                        function.get("name") or "?",
                    )
                    repaired_any = True
                    continue
                if signature:
                    seen_signatures.add(signature)
                if idx > 0:
                    new_call["id"] = f"{call.get('id')}:{idx}"
                kept.append(new_call)
            repaired_any = True
            continue
        fixed = _repair_json_arguments(arguments)
        if fixed != arguments:
            try:
                json.loads(fixed)
                function["arguments"] = fixed
                repaired_any = True
                logger.warning(
                    "[repair] call=%s %r -> %r",
                    function.get("name") or "?",
                    str(arguments)[:80],
                    str(fixed)[:80],
                )
                kept.append(call)
                continue
            except Exception:
                pass
        stripped = arguments.strip()
        if stripped in ("}", "]", "", "{}", "[]"):
            logger.warning("[drop] orphan closing fragment %r", stripped[:30])
            continue
        logger.warning(
            "[tool-drop] dropping unusable call %s args=%r",
            function.get("name") or "?",
            str(arguments)[:200],
        )
        repaired_any = True
    if repaired_any:
        logger.warning("[repair] repaired/split/dropped unusable tool call(s) this turn")
    return kept


def _calls_from_payload(data: Any) -> list[dict]:
    if not isinstance(data, dict):
        return []
    calls = _extract_tool_calls(data)
    tool_info = data.get("tool_call_info")
    if isinstance(tool_info, dict):
        item = dict(tool_info)
        item.setdefault("id", tool_info.get("tool_call_id") or data.get("id"))
        call = _normalize_tool_call(item, index=len(calls))
        # ``extract_tool_calls`` now understands top-level ``tool_call_info``
        # natively.  Keep this compatibility fallback for older/partial
        # payloads, but do not append the same call a second time when the
        # native extractor already returned it (missing ids otherwise produce
        # two synthetic ids and two client-visible invocations).
        call_signature = _tool_call_signature(call) if call else ""
        already_present = bool(
            call
            and any(
                str(existing.get("id") or "") == str(call.get("id") or "")
                or (
                    call_signature
                    and call_signature == _tool_call_signature(existing)
                )
                for existing in calls
                if isinstance(existing, dict)
            )
        )
        if call and not already_present:
            calls.append(call)
    return _deduplicate_tool_calls(calls)


def _tool_names(tools: Any) -> Optional[set[str]]:
    """Return normalized caller tool names.

    ``None`` means the caller did not provide a tools field (legacy translator
    callers keep the old pass-through behavior); an empty set means tools were
    explicitly disabled for this request.
    """
    if tools is None:
        return None
    if isinstance(tools, dict):
        tools = list(tools.values())
    if not isinstance(tools, list):
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        candidate = function if isinstance(function, dict) else tool
        name = candidate.get("name") if isinstance(candidate, dict) else None
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _filter_tool_calls(
    calls: Any,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
) -> list[dict]:
    if not isinstance(calls, list):
        return []
    names = _tool_names(allowed_tools)
    if tool_choice == "none" or not names:
        # ``names is None`` means the caller omitted an explicit catalog.  It
        # must still honor an explicit ``tool_choice=none``; an empty set means
        # tools were explicitly disabled and therefore suppresses calls too.
        if tool_choice == "none" or names == set():
            return []
        # No catalog + auto/required is a valid TraeWork custom ingress. Keep
        # parsed protocol calls and apply the remaining policy below.
    selected: Optional[str] = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if not isinstance(fn, dict):
            fn = tool_choice
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            selected = fn["name"].strip()
    filtered: list[dict] = []
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = _call_function_name(call)
        if names is not None and name not in names:
            continue
        if selected and name != selected:
            continue
        call_id = str(call.get("id") or "")
        is_delta = call.get("_arguments_mode") == "delta"
        if not is_delta:
            if call_id and call_id in seen:
                continue
            if call_id:
                seen.add(call_id)
        filtered.append(call)
        if parallel_tool_calls is False:
            break
    return filtered


def _suppress_completed_tool_calls(
    calls: Any, completed_tool_signatures: Any = None
) -> list[dict]:
    """Drop exact repeats of a tool call whose result is already in history.

    The upstream occasionally echoes the serialized assistant/tool history as
    a fresh request (often with a new id).  Comparing only ids is insufficient;
    compare the canonical name/arguments signature and keep the caller's
    explicit new-user-message escape hatch in ``completed_tool_signatures``.
    """

    if not isinstance(calls, list):
        return []
    protected = {
        str(value)
        for value in (completed_tool_signatures or [])
        if isinstance(value, str) and value
    }
    if not protected:
        return [call for call in calls if isinstance(call, dict)]
    filtered: list[dict] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        signature = _tool_call_signature(call)
        if signature and signature in protected:
            logger.warning(
                "suppressing repeated completed tool call: %s",
                (call.get("function") or {}).get("name", ""),
            )
            continue
        filtered.append(call)
    return filtered


def _contains_completed_tool_repeat(
    calls: Any, completed_tool_signatures: Any = None
) -> bool:
    if not isinstance(calls, list):
        return False
    protected = {
        str(value)
        for value in (completed_tool_signatures or [])
        if isinstance(value, str) and value
    }
    if not protected:
        return False
    return any(
        isinstance(call, dict)
        and (signature := _tool_call_signature(call))
        and signature in protected
        for call in calls
    )


def _filter_for_accumulator(
    accumulator: ToolCallAccumulator,
    calls: Any,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    completed_tool_signatures: Any = None,
) -> list[dict]:
    prepared = accumulator.prepare(
        _suppress_completed_tool_calls(calls, completed_tool_signatures)
    )
    return _filter_tool_calls(
        prepared,
        allowed_tools,
        tool_choice,
        parallel_tool_calls,
    )


def _emit_tool_deltas(
    prefix_id: str,
    model: str,
    accumulator: ToolCallAccumulator,
    calls: list[dict],
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    completed_tool_signatures: Any = None,
):
    calls = _filter_for_accumulator(
        accumulator,
        calls,
        allowed_tools,
        tool_choice,
        parallel_tool_calls,
        completed_tool_signatures,
    )
    return [
        openai_chunk(prefix_id, model, {"tool_calls": [delta]})
        for delta in accumulator.add(calls)
    ]


def _visible_text(text: Any) -> str:
    return _strip_tool_call_blocks(text) if isinstance(text, str) else ""


def _hold_incomplete_tool_block(
    text: str, *, hold_escape_prefix: bool = True
) -> str:
    """Keep a split XML tool block out of visible text until it is complete."""
    lowered = text.lower()
    pairs = (
        ("<opencode_tool_call>", "</opencode_tool_call>"),
        ("<tool_use>", "</tool_use>"),
        ("<tool_call", "</tool_call>"),
        ("<tool_cell", "</tool_cell>"),
        ("<invoke", "</invoke>"),
        ("<bash>", "</bash>"),
        ("<read>", "</read>"),
        ("<write>", "</write>"),
        ("<edit>", "</edit>"),
        ("<glob>", "</glob>"),
        ("<grep>", "</grep>"),
        ("<task>", "</task>"),
    )
    earliest = len(text)
    for opening, closing in pairs:
        start = lowered.find(opening)
        if start >= 0 and lowered.find(closing, start + len(opening)) < 0:
            earliest = min(earliest, start)
    last_lt = lowered.rfind("<")
    if last_lt >= 0:
        suffix = lowered[last_lt:]
        if any(opening.startswith(suffix) for opening, _ in pairs):
            earliest = min(earliest, last_lt)
    if hold_escape_prefix:
        # An escaped wait frame can split inside its leading backslash run,
        # before the first ``<`` makes the protocol prefix recognizable.
        escape_prefix = re.search(r"\\+[ \t\r\n]*$", text)
        if escape_prefix:
            earliest = min(earliest, escape_prefix.start())
    return text[:earliest]


class ProtocolTextAccumulator:
    """Merge cumulative or incremental text while hiding serialized tool calls."""

    def __init__(self):
        self.raw = ""
        self.visible = ""

    def _result(self, *, final: bool = False) -> tuple[str, list[dict]]:
        visible = _hold_incomplete_tool_block(
            _visible_text(self.raw), hold_escape_prefix=not final
        )
        delta = _cli_text_delta(self.visible, visible)
        self.visible = visible
        calls = _extract_tool_calls({"response": self.raw}) if self.raw else []
        return delta, calls

    def finalize(self) -> tuple[str, list[dict]]:
        """Release a harmless trailing escape run at logical text EOF."""

        return self._result(final=True)

    def add_delta(self, value: Any) -> tuple[str, list[dict]]:
        text = value if isinstance(value, str) else ""
        if text:
            self.raw += text
        return self._result()

    def add_snapshot(self, value: Any) -> tuple[str, list[dict]]:
        text = value if isinstance(value, str) else ""
        if text:
            # Cumulative snapshots can arrive out of order.  A shorter
            # prefix/suffix is an old snapshot, not a new assistant reply.
            if self.raw.startswith(text):
                return self._result()
            if self.raw.endswith(text) and len(text) < len(self.raw):
                return self._result()
            self.raw = text
        return self._result()

    def add(self, value: Any) -> tuple[str, list[dict]]:
        text = value if isinstance(value, str) else ""
        if text:
            if text.startswith(self.raw):
                self.raw = text
            elif not self.raw.startswith(text):
                common = _common_prefix_length(self.raw, text)
                shorter = min(len(self.raw), len(text))
                if common >= 4 or (shorter and common * 2 >= shorter):
                    self.raw = text
                else:
                    self.raw += text
        return self._result()


class ThinkingTracker:
    """Merge Trae reasoning_content and response into visible text."""

    def __init__(self):
        self.started = False
        self.ended = False

    def merge(self, reasoning: str, response: str) -> str:
        parts = []
        if reasoning:
            if not self.started:
                parts.append(THINK_OPEN + reasoning)
                self.started = True
                self.ended = False
            else:
                parts.append(reasoning)
        if response:
            if self.started and not self.ended:
                parts.append(THINK_CLOSE + response)
                self.started = False
                self.ended = True
            else:
                parts.append(response)
        return "".join(parts)


_EMBEDDED_THINK_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thinking)\b[^>]*>(?P<body>[\s\S]*?)</(?P=tag)>",
    flags=re.IGNORECASE,
)


def _split_embedded_reasoning(text: Any) -> tuple[str, str]:
    """Split optional ``<think>`` markup from ordinary assistant text."""

    if not isinstance(text, str) or not text:
        return "", ""
    matches = list(_EMBEDDED_THINK_BLOCK_RE.finditer(text))
    if not matches:
        return "", text
    reasoning = "\n".join(
        match.group("body") for match in matches if match.group("body")
    )
    visible = _EMBEDDED_THINK_BLOCK_RE.sub("", text)
    return reasoning, visible


def parse_ide_sse_line(data: str) -> Optional[str]:
    data = data.strip()
    if not data.startswith("data:"):
        return None
    return data[5:].strip()


def _cli_text_delta(previous: str, current: str) -> str:
    if not current:
        return ""
    if current.startswith(previous):
        return current[len(previous):]
    if previous.startswith(current):
        return ""
    common = _common_prefix_length(previous, current)
    return current[common:]


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _text_suffix_prefix_overlap(left: str, right: str, minimum: int = 4) -> int:
    """Return the longest suffix(left)/prefix(right) overlap."""

    limit = min(len(left), len(right))
    for size in range(limit, minimum - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _merge_plan_message_text(plan_text: str, message_text: str) -> str:
    """Merge remote plan content and message text without replaying either."""

    plan = plan_text or ""
    message = message_text or ""
    if not plan:
        return message
    if not message:
        return plan
    if plan == message or plan.startswith(message) or plan.endswith(message):
        return plan
    if message.startswith(plan):
        return message
    overlap = _text_suffix_prefix_overlap(plan, message)
    if overlap:
        return plan + message[overlap:]
    return plan.rstrip() + "\n\n" + message.lstrip()


async def translate_ide_stream(
    response,
    model: str,
    forward_usage: bool = True,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    fail_on_empty: bool = False,
    completed_tool_signatures: Any = None,
    require_terminal: bool = True,
    upstream_metadata: Optional[dict] = None,
    include_reasoning: bool = False,
):
    """Translate /api/ide chat or llm_raw_chat SSE into OpenAI SSE."""
    prefix_id = make_id()
    tracker = ThinkingTracker()
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)
    completion_bytes = 0
    content_count = 0
    pending_event = None
    last_queue_position = None
    reasoning_text = ProtocolTextAccumulator()
    response_text = ProtocolTextAccumulator()
    final_usage = None
    final_reason = "stop"
    provider_model_name = ""
    saw_completed_repeat = False
    saw_terminal = False
    terminal_event_pending = False
    started = not fail_on_empty
    pending_queue_chunks: list[str] = []

    if started:
        yield openai_chunk(prefix_id, model, {"role": "assistant"})

    async for line in _iter_stream_lines(response):
        if line is _STREAM_HEARTBEAT:
            # The public dispatcher already emitted a parseable start frame.
            # Keep the wire active while Trae is still waiting for its first
            # token/tool frame as well as during later model pauses.
            yield _STREAM_HEARTBEAT_LINE
            continue
        if line is None:
            continue
        raw = line.strip()
        if not raw:
            continue
        if raw.lower().startswith("event:"):
            pending_event = _normalize_event_name(raw[6:].strip())
            # Keep the event line itself as a terminal marker.  The normal SSE
            # form carries a final data frame after ``event: done``; retaining
            # this pending bit also handles a clean close immediately after
            # the event line without misclassifying it as truncation.
            terminal_event_pending = pending_event == "done"
            continue
        if not raw.lower().startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload.upper() == "[DONE]":
            saw_terminal = True
            break
        try:
            obj = json.loads(payload)
        except Exception:
            if terminal_event_pending:
                saw_terminal = True
                break
            continue
        if not isinstance(obj, dict):
            if terminal_event_pending:
                saw_terminal = True
                break
            continue
        event_type = pending_event or _payload_event_name(obj)
        pending_event = None
        terminal_event_pending = False
        _capture_upstream_metadata(upstream_metadata, obj)
        if event_type == "error":
            raise RuntimeError(
                str(obj.get("message") or obj.get("error") or "Trae raw upstream returned an error event")
            )
        if event_type == "request_wait_in_queue" or obj.get("position") is not None:
            position = obj.get("position", 0)
            if position != last_queue_position:
                queue_chunk = openai_chunk(
                    prefix_id,
                    model,
                    {"content": f"排队中，当前位置：{position}\n"},
                )
                if started:
                    yield queue_chunk
                    content_count += 1
                else:
                    pending_queue_chunks.append(queue_chunk)
                last_queue_position = position
            continue
        if event_type == "token_usage":
            final_usage = _map_usage(obj.get("usage") or obj)
            continue

        reported_provider = _provider_model_name({**obj, "event": event_type})
        if reported_provider:
            provider_model_name = reported_provider
            _check_provider_model(model, reported_provider)

        reasoning = _payload_reasoning_text(obj)
        raw_response = obj.get("response") if isinstance(obj.get("response"), str) else ""
        embedded_reasoning, visible_response = _split_embedded_reasoning(raw_response)
        reasoning_delta, reasoning_calls = reasoning_text.add(reasoning)
        if embedded_reasoning:
            embedded_delta, embedded_calls = reasoning_text.add(embedded_reasoning)
            reasoning_delta += embedded_delta
            reasoning_calls.extend(embedded_calls)
        response_delta, response_calls = response_text.add(visible_response)
        calls = _calls_from_payload(obj)
        calls.extend(reasoning_calls)
        calls.extend(response_calls)
        if _contains_completed_tool_repeat(calls, completed_tool_signatures):
            saw_completed_repeat = True

        tool_chunks = list(
            _emit_tool_deltas(
                prefix_id,
                model,
                tool_calls,
                calls,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            )
        )
        if tool_chunks and not started:
            started = True
            yield openai_chunk(prefix_id, model, {"role": "assistant"})
            for pending in pending_queue_chunks:
                yield pending
            pending_queue_chunks.clear()
        for chunk in tool_chunks:
            yield chunk

        text_delta = response_delta
        if text_delta:
            if not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
                for pending in pending_queue_chunks:
                    yield pending
                pending_queue_chunks.clear()
            completion_bytes += len(text_delta.encode("utf-8"))
            content_count += 1
            yield openai_chunk(prefix_id, model, {"content": text_delta})

        if obj.get("usage"):
            final_usage = _map_usage(obj.get("usage"))
        if obj.get("finish_reason"):
            final_reason = str(obj.get("finish_reason"))
        # Trae sometimes attaches a finish_reason/stop_reason to an
        # intermediate cumulative snapshot.  It is metadata, not a terminal
        # boundary.  Only an explicit done event or the [DONE] sentinel above
        # terminates the upstream stream; otherwise consuming the rest is
        # required to avoid truncating the answer.
        if event_type == "done":
            saw_terminal = True
            final_reason = _payload_finish_reason(obj) or final_reason
            break

    reasoning_delta, reasoning_calls = reasoning_text.finalize()
    response_delta, response_calls = response_text.finalize()
    final_calls = reasoning_calls + response_calls
    if _contains_completed_tool_repeat(final_calls, completed_tool_signatures):
        saw_completed_repeat = True
    final_tool_chunks = list(
        _emit_tool_deltas(
            prefix_id,
            model,
            tool_calls,
            final_calls,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            completed_tool_signatures,
        )
    )
    if final_tool_chunks and not started:
        started = True
        yield openai_chunk(prefix_id, model, {"role": "assistant"})
        for pending in pending_queue_chunks:
            yield pending
        pending_queue_chunks.clear()
    for chunk in final_tool_chunks:
        yield chunk

    final_text_delta = response_delta
    if final_text_delta:
        if not started:
            started = True
            yield openai_chunk(prefix_id, model, {"role": "assistant"})
            for pending in pending_queue_chunks:
                yield pending
            pending_queue_chunks.clear()
        completion_bytes += len(final_text_delta.encode("utf-8"))
        content_count += 1
        yield openai_chunk(prefix_id, model, {"content": final_text_delta})

    reasoning_summary = (
        compact_reasoning_text(reasoning_text.raw) if include_reasoning else ""
    )
    if reasoning_summary:
        if not started:
            started = True
            yield openai_chunk(prefix_id, model, {"role": "assistant"})
            for pending in pending_queue_chunks:
                yield pending
            pending_queue_chunks.clear()
        yield openai_chunk(
            prefix_id,
            model,
            {"reasoning_content": reasoning_summary},
        )

    if terminal_event_pending:
        saw_terminal = True
    if not saw_terminal and require_terminal:
        observed_model_event = bool(
            content_count
            or tool_calls.has_calls
            or final_usage is not None
            or provider_model_name
            or saw_completed_repeat
            or (
                isinstance(upstream_metadata, dict)
                and upstream_metadata.get("usage_turn_id")
            )
        )
        raise IncompleteUpstreamResponse(
            "Trae raw upstream ended before its terminal event",
            retryable=not observed_model_event,
            usage=final_usage,
            observed_model_event=observed_model_event,
        )
    tool_calls.set_calls(
        _ensure_native_tool_arguments(tool_calls.calls(), final_usage)
    )
    if content_count == 0 and not tool_calls.has_calls and saw_completed_repeat:
        raise RepeatedCompletedToolResponse(
            "Trae upstream repeated only already completed tool calls",
            retryable=False,
            usage=final_usage,
            observed_model_event=True,
        )
    required_error = _required_tool_error(
        tool_choice, tool_calls.has_calls, tool_calls.calls()
    )
    if required_error:
        yield openai_chunk(
            prefix_id,
            model,
            {},
            finish_reason="stop",
            error=required_error,
        )
        yield "data: [DONE]\n\n"
        return
    if (
        fail_on_empty
        and content_count == 0
        and not tool_calls.has_calls
        and not reasoning_summary
    ):
        observed_model_event = bool(
            final_usage is not None
            or provider_model_name
            or (
                isinstance(upstream_metadata, dict)
                and upstream_metadata.get("usage_turn_id")
            )
        )
        raise EmptyUpstreamResponse(
            "Trae raw upstream returned no text or tool call",
            retryable=not observed_model_event,
            usage=final_usage,
            observed_model_event=observed_model_event,
        )

    if content_count == 0 and not tool_calls.has_calls and not reasoning_summary:
        yield openai_chunk(prefix_id, model, {"content": "(trae upstream returned an empty response)"})
    usage = final_usage
    if forward_usage and usage is None:
        estimated = completion_bytes // 4 + (1 if completion_bytes else 0)
        usage = {"prompt_tokens": 0, "completion_tokens": estimated, "total_tokens": estimated}
    yield openai_chunk(
        prefix_id,
        model,
        {},
        finish_reason=_finish_reason(final_reason, tool_calls.has_calls),
        usage=usage if forward_usage else None,
        provider_model_name=provider_model_name,
    )
    yield "data: [DONE]\n\n"


# Meta-narration the remote agent prepends to ``thought`` before the real
# answer. The upstream ships reasoning and reply in one field, so a prompt
# directive alone cannot keep this out of the visible answer.
_NARRATION_SENTENCE_RE = re.compile(
    r"""^[ \t]*(?:
        (?:the\s+)?user(?:'s)?\s+(?:wants?|asks?|is\s+asking|input|query|question|
            has\s+sent|request(?:s|ed)?|needs?)\b
      | (?:so\s+|ok(?:ay)?[,.]?\s+|now[,.]?\s+)?let(?:'s|\s+me)\b
      | i\s+(?:need\s+to|should|will|'ll|must|can)\b
      | (?:this|that)\s+(?:is|does\s?n?o?t?|require)\b.{0,60}?
            (?:question|request|tool|skill|exploration|change)
      | (?:no|nothing)\s+(?:tools?|skill|codebase|file)\b
      | simple\s+(?:factual|conceptual|knowledge)\s+question\b
      | (?:this\s+is\s+a\s+)?(?:straightforward|simple|direct)\s+
            (?:question|request|task)\b
      | (?:the\s+)?(?:simplest|easiest|best|standard)\s+way\s+(?:is|would\s+be)\b
      | (?:i'?ll\s+|i\s+will\s+)?answer\s+(?:in|with|concisely|directly|briefly)\b
      | (?:here'?s|here\s+is)\s+(?:the|a)\s+(?:answer|solution|approach)\b
    # A narration sentence can run straight into the answer without a space
    # ("...two sentences.TCP is slower"), so do not require whitespace after
    # the terminator.
    )[^\n]*?(?:[.!?]|\n|$)""",
    re.IGNORECASE | re.VERBOSE,
)


# A cumulative snapshot can be cut mid-narration ("The", "The user wa"). Those
# prefixes do not match the sentence pattern yet, so streaming them out would
# leave a dangling fragment that no later frame can retract.
_NARRATION_PREFIX_RE = re.compile(
    r"""^[ \t]*(?:
        t(?:h(?:e(?:\s+u(?:s(?:e(?:r(?:'?s?)?)?)?)?)?)?)?
      | l(?:e(?:t(?:'?s?|\s+m(?:e)?)?)?)?
      | i(?:\s+(?:n(?:e(?:e(?:d)?)?)?|s(?:h(?:o(?:u(?:l(?:d)?)?)?)?)?|w(?:i(?:l(?:l)?)?)?|m(?:u(?:s(?:t)?)?)?))?
      | s(?:i(?:m(?:p(?:l(?:e)?)?)?)?)?
      | n(?:o(?:\s+t(?:o(?:o(?:l(?:s)?)?)?)?)?)?
      | t(?:h(?:i(?:s)?)?)?
    )[ \t]*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _concise_reasoning_enabled() -> bool:
    return str(os.environ.get("TRAE_VERBOSE_REASONING", "")).strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def strip_reasoning_narration(text: str, *, hold_incomplete: bool = False) -> str:
    """Drop leading meta-narration sentences from a cumulative answer snapshot.

    Only a prefix is removed, so the result stays stable as the snapshot grows
    (important for incremental streaming deltas).

    With ``hold_incomplete`` the function returns ``""`` while the snapshot is
    still nothing but narration.  A streaming caller must hold that text back:
    once a delta is on the wire it cannot be retracted, and the answer that
    follows would otherwise arrive behind the narration it was meant to replace.
    Without the flag an all-narration snapshot is returned unchanged, which is
    what a non-streaming caller wants as a last resort so the turn is not empty.
    """

    if not text or not _concise_reasoning_enabled():
        return text
    if hold_incomplete and _NARRATION_PREFIX_RE.match(text):
        # Still inside a possible narration opener; wait for the full sentence.
        return ""
    remainder = text
    removed = False
    for _ in range(6):
        match = _NARRATION_SENTENCE_RE.match(remainder)
        if not match or match.end() == 0:
            break
        candidate = remainder[match.end():]
        if not candidate.strip():
            # Nothing but narration so far.
            return "" if hold_incomplete else text
        remainder = candidate
        removed = True
    if not removed:
        return text
    if not remainder.strip():
        return text
    # The answer follows the dropped sentence, often after a space or newline.
    return remainder.lstrip(" \t\n")


_REASONING_NODE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\r?\n+")
_REASONING_PRIORITY_RE = re.compile(
    r"(?:结论|结果|原因|关键|方案|修复|验证|完成|失败|成功|风险|因此|最终|"
    r"conclusion|result|because|therefore|fix(?:ed)?|verified|success|fail(?:ed)?|risk)",
    re.IGNORECASE,
)


def compact_reasoning_text(
    text: Any, *, max_chars: int = 800, max_lines: int = 6
) -> str:
    """Return a few reasoning checkpoints instead of the model's full trace.

    This is intentionally extractive and bounded.  It removes serialized tool
    protocol first, splits long paragraphs into sentence-sized checkpoints,
    de-duplicates cumulative snapshots, and retains a small set of salient or
    final nodes.  Tool calls are parsed from the unmodified upstream payload by
    the callers before this presentation-only summary is produced.
    """

    if not isinstance(text, str) or not text.strip() or max_chars <= 0 or max_lines <= 0:
        return ""
    cleaned = _visible_text(text)
    cleaned = re.sub(
        r"<(?:think|thinking)\b[^>]*>|</(?:think|thinking)>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = strip_reasoning_narration(
        cleaned.strip(), hold_incomplete=False
    ).strip()
    if not cleaned:
        return ""

    nodes: list[str] = []
    seen: set[str] = set()
    for raw in _REASONING_NODE_SPLIT_RE.split(cleaned):
        node = re.sub(r"[ \t]+", " ", raw).strip()
        if not node:
            continue
        # A single generated line can contain a very long working trace.  A
        # bounded node prevents one paragraph from defeating the summary cap.
        if len(node) > 240:
            node = node[:237].rstrip() + "..."
        fingerprint = re.sub(r"\s+", " ", node).casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        nodes.append(node)
    if not nodes:
        return ""

    if len(nodes) > max_lines:
        priority = [
            index for index, node in enumerate(nodes) if _REASONING_PRIORITY_RE.search(node)
        ]
        selected: list[int] = []
        # Preserve one orienting node, then prefer conclusions and the newest
        # checkpoints.  Sorting restores the original reasoning order.
        selected.append(0)
        for index in priority:
            if index not in selected:
                selected.append(index)
            if len(selected) >= max_lines:
                break
        for index in range(len(nodes) - 1, -1, -1):
            if index not in selected:
                selected.append(index)
            if len(selected) >= max_lines:
                break
        nodes = [nodes[index] for index in sorted(selected[:max_lines])]

    result = "\n".join(nodes)
    if len(result) > max_chars:
        result = result[: max_chars - 3].rstrip() + "..."
    return result


def _payload_reasoning_text(data: Any) -> str:
    """Extract a reasoning field without treating it as assistant content."""

    if not isinstance(data, dict):
        return ""
    for key in ("reasoning_content", "reasoningContent", "thought", "thinking"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for nested_key in ("content", "text", "summary"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested:
                    return nested
    message = data.get("message")
    if isinstance(message, dict):
        nested = _payload_reasoning_text(message)
        if nested:
            return nested
        content = message.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if str(block.get("type") or "").lower() not in {
                    "reasoning",
                    "thinking",
                    "reasoning_text",
                }:
                    continue
                value = block.get("text") or block.get("content") or block.get("value")
                if isinstance(value, str):
                    parts.append(value)
            if parts:
                return "\n".join(parts)
    return ""


def _web_plan_has_split_reasoning(data: dict) -> bool:
    """Current remote plan items carry reasoning in ``reasoning_content``.

    In that layout ``thought`` is the visible answer snapshot (for example
    ``thought="pong", reasoning_content=""``).  Older variants omit the
    ``reasoning_content`` key and use ``thought`` as the trace.
    """

    return isinstance(data, dict) and any(
        key in data for key in ("reasoning_content", "reasoningContent")
    )


def _web_plan_reasoning(data: dict) -> str:
    if _web_plan_has_split_reasoning(data):
        value = data.get("reasoning_content")
        if not isinstance(value, str) or not value:
            value = data.get("reasoningContent")
        return value if isinstance(value, str) else ""
    return _payload_reasoning_text(data)


def _web_plan_content(data: dict) -> str:
    content = _web_message_text({"content": data.get("content")})
    if content:
        return content
    if _web_plan_has_split_reasoning(data):
        thought = data.get("thought")
        if isinstance(thought, str):
            return thought
    return ""


def _web_plan_text(data: dict, *, hold_incomplete: bool = False) -> str:
    """Return the visible text of a remote ``plan_item`` event.

    The remote SSE carries the reasoning trace in ``thought`` /
    ``reasoning_content`` and the actual reply in ``content`` (report: plan_item
    data = {id, thought, content}).  Both are cumulative snapshots, so append
    the reply after the trace instead of dropping it; a plan item that only
    carries ``content`` otherwise reads as an empty upstream response.
    """

    thought = data.get("thought") or data.get("reasoning_content") or ""
    if not isinstance(thought, str):
        thought = ""
    thought = strip_reasoning_narration(thought, hold_incomplete=hold_incomplete)
    content = _web_message_text({"content": data.get("content")})
    if not content:
        return thought
    if not thought:
        return content
    if content in thought:
        return thought
    return f"{thought}\n\n{content}"


def _web_finish_summary(data: dict) -> str:
    tci = data.get("tool_call_info") or data.get("toolCallInfo") or {}
    if isinstance(tci, dict) and tci.get("name") == "finish":
        params = tci.get("params") or {}
        if isinstance(params, dict):
            return str(params.get("summary") or "")
    return ""


def _web_message_text(data: dict) -> str:
    """Extract visible assistant text from remote message/content events."""
    if not isinstance(data, dict):
        return ""
    for key in ("message", "agent_message", "assistant_message"):
        nested = data.get(key)
        if isinstance(nested, dict):
            data = nested
            break
    content = data.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if str(block.get("type") or "").lower() in {
                    "reasoning",
                    "thinking",
                    "reasoning_text",
                }:
                    continue
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    for key in ("text", "response", "answer", "output"):
        value = data.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for block in value:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            if parts:
                return "".join(parts)
    return ""

async def translate_web_events(
    event_iter,
    model: str,
    forward_usage: bool = True,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    completed_tool_signatures: Any = None,
    fail_on_empty: bool = False,
    include_reasoning: bool = False,
):
    prefix_id = make_id()
    reasoning_order: list[str] = []
    reasoning_states: dict[str, ProtocolTextAccumulator] = {}
    plan_visible_states: dict[str, ProtocolTextAccumulator] = {}
    streamed_text = ""
    plan_streamed_text = ""
    message_visible_text = ""
    message_text = ProtocolTextAccumulator()
    usage = None
    error_event = None
    final_summary = ""
    final_reason = "stop"
    provider_model_name = ""
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)
    started = not fail_on_empty
    # True once a plan item used the legacy layout where ``thought`` is the
    # only text surface.  Only that layout may fall back to reasoning text.
    legacy_plan_reasoning = False

    if started:
        yield openai_chunk(prefix_id, model, {"role": "assistant"})
    async for event, data in event_iter:
        event = _normalize_event_name(event)
        if not isinstance(data, dict):
            data = {}
        # The remote Trae SSE emits explicit ``heartbeat`` events while the
        # agent is thinking.  Keep those frames alive on the public OpenAI SSE
        # stream; silently dropping them makes clients such as zcode assume the
        # turn ended and replay the previous tool request.
        if event in {"heartbeat", "keepalive", "ping"}:
            yield _STREAM_HEARTBEAT_LINE
            continue
        if event == "error":
            error_event = data
            break
        reported_provider = _provider_model_name({**data, "event": event})
        if reported_provider:
            provider_model_name = reported_provider
            _check_provider_model(model, reported_provider)
        if event == "token_usage":
            usage = _map_usage(data.get("usage") or data)
            continue
        if event == "plan_item":
            emitted_tool = False
            pid = str(data.get("id") or f"plan-{len(reasoning_order)}")
            plan_reasoning = _web_plan_reasoning(data)
            if not _web_plan_has_split_reasoning(data) and plan_reasoning:
                legacy_plan_reasoning = True
            reasoning_state = reasoning_states.setdefault(
                pid, ProtocolTextAccumulator()
            )
            if pid not in reasoning_order:
                reasoning_order.append(pid)
            _, reasoning_calls = reasoning_state.add(plan_reasoning)
            visible_state = plan_visible_states.setdefault(
                pid, ProtocolTextAccumulator()
            )
            # ``thought`` is a cumulative snapshot.  Filter leading
            # meta-narration before feeding it to the public accumulator, while
            # retaining the raw snapshot above for compact reasoning and tool
            # extraction.  The explicit ``content`` field is appended as the
            # actual answer when present.
            # Only the explicit ``content`` field is public answer text.  The
            # adjacent ``thought`` snapshot remains in ``reasoning_state`` and
            # is compacted once at the end of the turn.
            visible_snapshot = _web_plan_content(data)
            content_delta, content_calls = visible_state.add(visible_snapshot)
            plan_calls = _calls_from_payload(data)
            plan_calls.extend(reasoning_calls)
            plan_calls.extend(content_calls)
            plan_chunks = list(_emit_tool_deltas(
                prefix_id,
                model,
                tool_calls,
                plan_calls,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            ))
            if plan_chunks and not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
            for chunk in plan_chunks:
                emitted_tool = True
                yield chunk
            if emitted_tool and allowed_tools is not None:
                break
            if content_delta:
                if not started:
                    started = True
                    yield openai_chunk(prefix_id, model, {"role": "assistant"})
                plan_streamed_text += content_delta
                merged_text = _merge_plan_message_text(
                    plan_streamed_text, message_visible_text
                )
                public_delta = _cli_text_delta(streamed_text, merged_text)
                if public_delta:
                    streamed_text += public_delta
                    yield openai_chunk(prefix_id, model, {"content": public_delta})
            summary = _web_finish_summary(data)
            if summary:
                final_summary = summary
        if event in {"message", "assistant_message", "response", "text", "output"}:
            text = _web_message_text(data)
            event_reasoning = _payload_reasoning_text(data)
            embedded_reasoning, text = _split_embedded_reasoning(text)
            if event_reasoning or embedded_reasoning:
                reasoning_key = "__messages__"
                reasoning_state = reasoning_states.setdefault(
                    reasoning_key, ProtocolTextAccumulator()
                )
                if reasoning_key not in reasoning_order:
                    reasoning_order.append(reasoning_key)
                if event_reasoning:
                    reasoning_state.add(event_reasoning)
                if embedded_reasoning:
                    reasoning_state.add(embedded_reasoning)
            if text:
                _, message_calls = message_text.add_snapshot(text)
                for chunk in _emit_tool_deltas(
                    prefix_id,
                    model,
                    tool_calls,
                    message_calls,
                    allowed_tools,
                    tool_choice,
                    parallel_tool_calls,
                    completed_tool_signatures,
                ):
                    if not started:
                        started = True
                        yield openai_chunk(prefix_id, model, {"role": "assistant"})
                    yield chunk
                message_visible_text = message_text.visible
                merged_text = _merge_plan_message_text(
                    plan_streamed_text, message_visible_text
                )
                message_delta = _cli_text_delta(streamed_text, merged_text)
                if message_delta:
                    if not started:
                        started = True
                        yield openai_chunk(prefix_id, model, {"role": "assistant"})
                    streamed_text += message_delta
                    yield openai_chunk(prefix_id, model, {"content": message_delta})
        if event == "done":
            final_reason = _payload_finish_reason(data) or final_reason
            break

    # Finalize every protocol accumulator so a harmless trailing backslash is
    # released while incomplete tool markup remains hidden from presentation.
    for state in reasoning_states.values():
        _, calls = state.finalize()
        for chunk in _emit_tool_deltas(
            prefix_id,
            model,
            tool_calls,
            calls,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            completed_tool_signatures,
        ):
            if not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
            yield chunk
    for state in plan_visible_states.values():
        content_delta, calls = state.finalize()
        for chunk in _emit_tool_deltas(
            prefix_id,
            model,
            tool_calls,
            calls,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            completed_tool_signatures,
        ):
            if not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
            yield chunk
        if content_delta:
            if not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
            plan_streamed_text += content_delta
            merged_text = _merge_plan_message_text(
                plan_streamed_text, message_visible_text
            )
            public_delta = _cli_text_delta(streamed_text, merged_text)
            if public_delta:
                streamed_text += public_delta
                yield openai_chunk(prefix_id, model, {"content": public_delta})

    message_delta, message_calls = message_text.finalize()
    for chunk in _emit_tool_deltas(
        prefix_id,
        model,
        tool_calls,
        message_calls,
        allowed_tools,
        tool_choice,
        parallel_tool_calls,
        completed_tool_signatures,
    ):
        if not started:
            started = True
            yield openai_chunk(prefix_id, model, {"role": "assistant"})
        yield chunk
    message_visible_text = message_text.visible
    merged_text = _merge_plan_message_text(
        plan_streamed_text, message_visible_text
    )
    if message_delta or merged_text:
        public_delta = _cli_text_delta(streamed_text, merged_text)
        if public_delta:
            if not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
            streamed_text += public_delta
            yield openai_chunk(prefix_id, model, {"content": public_delta})

    if final_summary:
        summary_calls = _extract_tool_calls({"response": final_summary})
        if summary_calls:
            if not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
            for chunk in _emit_tool_deltas(
                prefix_id,
                model,
                tool_calls,
                summary_calls,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            ):
                yield chunk
        # The finish summary is another reasoning surface, so apply the same
        # narration filter used for plan thoughts. Without this a narrated
        # summary arrives *after* the answer that already streamed.
        final_summary = strip_reasoning_narration(
            _visible_text(final_summary).strip(), hold_incomplete=True
        ).strip()
    reasoning_raw = "\n".join(
        reasoning_states[key].raw for key in reasoning_order if reasoning_states[key].raw
    )
    reasoning_summary = compact_reasoning_text(reasoning_raw)
    if include_reasoning and reasoning_summary:
        if not started:
            started = True
            yield openai_chunk(prefix_id, model, {"role": "assistant"})
        yield openai_chunk(
            prefix_id,
            model,
            {"reasoning_content": reasoning_summary},
        )

    # Do not present a web agent's remote tool result as the external client's
    # tool result. The API client owns execution and the following turn.
    if not tool_calls.has_calls:
        if not streamed_text and final_summary:
            if not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
            streamed_text += final_summary
            yield openai_chunk(prefix_id, model, {"content": final_summary})
        elif final_summary:
            summary = final_summary.rstrip()
            streamed_head = streamed_text.strip()
            if streamed_head and summary.startswith(streamed_head):
                # The answer snapshot is a prefix of the finish summary (the
                # last snapshot was cut short); emit only the missing tail.
                tail = summary[len(streamed_head):]
                if tail:
                    streamed_text += tail
                    yield openai_chunk(prefix_id, model, {"content": tail})
            elif not streamed_text.rstrip().endswith(summary):
                if not started:
                    started = True
                    yield openai_chunk(prefix_id, model, {"role": "assistant"})
                streamed_text += "\n\n" + final_summary
                yield openai_chunk(prefix_id, model, {"content": "\n\n" + final_summary})

    if (
        not streamed_text
        and not tool_calls.has_calls
        and reasoning_summary
        and not include_reasoning
        and not legacy_plan_reasoning
    ):
        # Split layout with reasoning but no answer: keep the trace in the
        # dedicated reasoning channel instead of presenting it as content.
        if not started:
            started = True
            yield openai_chunk(prefix_id, model, {"role": "assistant"})
        yield openai_chunk(
            prefix_id, model, {"reasoning_content": reasoning_summary}
        )
        include_reasoning = True
    if (
        not streamed_text
        and not tool_calls.has_calls
        and reasoning_summary
        and not include_reasoning
    ):
        # Older remote variants place the final answer only in ``thought``.
        # Return a bounded fallback instead of the complete trace or an empty
        # response; explicit thinking requests receive it in the dedicated
        # reasoning channel above.
        if reasoning_summary.strip():
            if not started:
                started = True
                yield openai_chunk(prefix_id, model, {"role": "assistant"})
            streamed_text += reasoning_summary
            yield openai_chunk(prefix_id, model, {"content": reasoning_summary})

    saw_reasoning_output = bool(include_reasoning and reasoning_summary)
    required_error = _required_tool_error(
        tool_choice, tool_calls.has_calls, tool_calls.calls()
    )
    if required_error and not error_event:
        yield openai_chunk(
            prefix_id,
            model,
            {},
            finish_reason="stop",
            error=required_error,
        )
        yield "data: [DONE]\n\n"
        return
    if fail_on_empty and not streamed_text and not tool_calls.has_calls and not saw_reasoning_output:
        quota_exhausted = _is_account_quota_error(error_event)
        # ``model_config`` is emitted before a Trae 4008 error.  It identifies
        # the selected provider, but it is not model output and must not make
        # an account-quota failure look non-retryable.  A retry is safe only
        # while this turn has produced no billable usage or visible reasoning.
        observed_model_event = bool(
            usage is not None
            or reasoning_raw
            or (provider_model_name and not quota_exhausted)
        )
        message = "Trae remote upstream returned no text or tool call"
        if error_event:
            message = (
                f"Trae remote upstream error: {error_event.get('code', '')}: "
                f"{error_event.get('message', '')}"
            )
        raise EmptyUpstreamResponse(
            message,
            retryable=not observed_model_event,
            usage=usage,
            observed_model_event=observed_model_event,
            account_quota_exhausted=quota_exhausted,
        )

    if error_event:
        yield openai_chunk(
            prefix_id,
            model,
            {},
            finish_reason="stop",
            error={
                "message": f"trae {error_event.get('code', '')}: {error_event.get('message', '')}",
                "type": "api_error",
            },
        )
    else:
        if required_error:
            yield openai_chunk(
                prefix_id,
                model,
                {},
                finish_reason="stop",
                error=required_error,
            )
            yield "data: [DONE]\n\n"
            return
        if not tool_calls.has_calls and not streamed_text and not saw_reasoning_output:
            yield openai_chunk(
                prefix_id,
                model,
                {"content": "(trae upstream returned an empty response)"},
            )
        yield openai_chunk(
            prefix_id,
            model,
            {},
            finish_reason=_finish_reason(final_reason, tool_calls.has_calls),
            usage=usage if forward_usage else None,
            provider_model_name=provider_model_name,
        )
    yield "data: [DONE]\n\n"


async def translate_cli_stream(
    event_iter,
    model: str,
    forward_usage: bool = True,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    completed_tool_signatures: Any = None,
    include_reasoning: bool = False,
):
    """Translate Trae CLI JSON/text output without executing returned calls."""
    prefix_id = make_id()
    text_state = ProtocolTextAccumulator()
    reasoning_state = ProtocolTextAccumulator()
    usage = None
    saw_output = False
    final_reason = "stop"
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)

    yield openai_chunk(prefix_id, model, {"role": "assistant"})
    async for event in event_iter:
        event_type = _normalize_event_name(event.type)
        if event_type == "error":
            if not saw_output and not tool_calls.has_calls:
                raise RuntimeError(event.error or "Trae CLI failed")
            yield openai_chunk(
                prefix_id,
                model,
                {},
                finish_reason=_finish_reason(final_reason, tool_calls.has_calls),
                error={"message": event.error or "Trae CLI failed", "type": "api_error"},
            )
            yield "data: [DONE]\n\n"
            return
        if event_type == "text":
            embedded_reasoning, visible_text = _split_embedded_reasoning(event.text or "")
            _, reasoning_calls = reasoning_state.add_delta(embedded_reasoning)
            text_delta, text_calls = text_state.add_delta(visible_text)
            text_calls.extend(reasoning_calls)
            for chunk in _emit_tool_deltas(
                prefix_id,
                model,
                tool_calls,
                text_calls,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            ):
                saw_output = True
                yield chunk
            if text_delta:
                saw_output = True
                yield openai_chunk(prefix_id, model, {"content": text_delta})
            continue
        if event_type != "json" or not event.data:
            continue
        result = event.data
        result_usage = _cli_extract_usage(result)
        if result_usage:
            usage = result_usage
        result_reason = _payload_finish_reason(result)
        if result_reason:
            final_reason = result_reason
        _, reasoning_calls = reasoning_state.add(_payload_reasoning_text(result))
        text_delta, text_calls = text_state.add_snapshot(_cli_extract_text(result))
        calls = _extract_tool_calls(result)
        calls.extend(reasoning_calls)
        calls.extend(text_calls)
        for chunk in _emit_tool_deltas(
            prefix_id,
            model,
            tool_calls,
            calls,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            completed_tool_signatures,
        ):
            saw_output = True
            yield chunk
        if text_delta:
            saw_output = True
            yield openai_chunk(prefix_id, model, {"content": text_delta})
        # A finish_reason on a CLI JSON snapshot is not a reliable terminal
        # marker: the CLI may emit cumulative snapshots with that field set
        # before the final snapshot.  Keep consuming until the process/stream
        # reaches EOF so later text and tool arguments are not truncated.

    text_delta, text_calls = text_state.finalize()
    _, reasoning_calls = reasoning_state.finalize()
    text_calls.extend(reasoning_calls)
    for chunk in _emit_tool_deltas(
        prefix_id,
        model,
        tool_calls,
        text_calls,
        allowed_tools,
        tool_choice,
        parallel_tool_calls,
        completed_tool_signatures,
    ):
        saw_output = True
        yield chunk
    if text_delta:
        saw_output = True
        yield openai_chunk(prefix_id, model, {"content": text_delta})

    reasoning_summary = (
        compact_reasoning_text(reasoning_state.raw) if include_reasoning else ""
    )
    if reasoning_summary:
        saw_output = True
        yield openai_chunk(
            prefix_id,
            model,
            {"reasoning_content": reasoning_summary},
        )

    required_error = _required_tool_error(
        tool_choice, tool_calls.has_calls, tool_calls.calls()
    )
    if required_error:
        yield openai_chunk(
            prefix_id,
            model,
            {},
            finish_reason="stop",
            error=required_error,
        )
        yield "data: [DONE]\n\n"
        return
    if not saw_output and not tool_calls.has_calls:
        yield openai_chunk(prefix_id, model, {"content": "(trae cli returned an empty response)"})
    yield openai_chunk(
        prefix_id,
        model,
        {},
        finish_reason=_finish_reason(final_reason, tool_calls.has_calls),
        usage=usage if forward_usage else None,
    )
    yield "data: [DONE]\n\n"


async def collect_nonstream_cli(
    event_iter,
    model: str,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    completed_tool_signatures: Any = None,
    include_reasoning: bool = False,
) -> dict:
    prefix_id = make_id()
    usage = None
    final_reason = "stop"
    text_state = ProtocolTextAccumulator()
    reasoning_state = ProtocolTextAccumulator()
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)
    async for event in event_iter:
        event_type = _normalize_event_name(event.type)
        if event_type == "error":
            raise RuntimeError(event.error or "Trae CLI failed")
        if event_type == "text" and event.text:
            embedded_reasoning, visible_text = _split_embedded_reasoning(event.text)
            _, reasoning_calls = reasoning_state.add_delta(embedded_reasoning)
            _, text_calls = text_state.add_delta(visible_text)
            text_calls.extend(reasoning_calls)
            tool_calls.add(
                _filter_for_accumulator(
                    tool_calls,
                    text_calls,
                    allowed_tools,
                    tool_choice,
                    parallel_tool_calls,
                    completed_tool_signatures,
                )
            )
        elif event_type == "json" and event.data:
            _, reasoning_calls = reasoning_state.add(
                _payload_reasoning_text(event.data)
            )
            _, text_calls = text_state.add_snapshot(
                _cli_extract_text(event.data)
            )
            calls = _extract_tool_calls(event.data)
            calls.extend(reasoning_calls)
            calls.extend(text_calls)
            tool_calls.add(
                _filter_for_accumulator(
                    tool_calls,
                    calls,
                    allowed_tools,
                    tool_choice,
                    parallel_tool_calls,
                    completed_tool_signatures,
                )
            )
            event_usage = _cli_extract_usage(event.data)
            if event_usage:
                usage = event_usage
            event_reason = _payload_finish_reason(event.data)
            if event_reason:
                final_reason = event_reason
                # Do not close/break on a snapshot finish_reason.  The CLI
                # stream has no uniformly reliable terminal event; EOF is the
                # authoritative boundary for non-stream collection.
    _, final_text_calls = text_state.finalize()
    _, final_reasoning_calls = reasoning_state.finalize()
    final_text_calls.extend(final_reasoning_calls)
    tool_calls.add(
        _filter_for_accumulator(
            tool_calls,
            final_text_calls,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            completed_tool_signatures,
        )
    )
    _ensure_required_tool_call(
        tool_choice, tool_calls.has_calls, tool_calls.calls()
    )
    content = text_state.visible.strip()
    reasoning_summary = (
        compact_reasoning_text(reasoning_state.raw) if include_reasoning else ""
    )
    if not content and not reasoning_summary and not tool_calls.has_calls:
        content = "(trae cli returned an empty response)"
    if usage is None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(content + reasoning_summary),
            "total_tokens": estimate_tokens(content + reasoning_summary),
        }
    return openai_completion(
        prefix_id,
        model,
        content or None,
        _finish_reason(final_reason, tool_calls.has_calls),
        usage,
        tool_calls.calls(),
        reasoning_content=reasoning_summary or None,
    )


async def collect_nonstream_ide(
    response,
    model: str,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    fail_on_empty: bool = False,
    completed_tool_signatures: Any = None,
    require_terminal: bool = True,
    upstream_metadata: Optional[dict] = None,
    include_reasoning: bool = False,
) -> dict:
    prefix_id = make_id()
    full = ""
    reasoning_text = ProtocolTextAccumulator()
    response_text = ProtocolTextAccumulator()
    finish_reason = "stop"
    usage = None
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)
    pending_event = None
    terminal_event_pending = False
    saw_terminal = False
    saw_completed_repeat = False
    provider_model_name = ""
    async for line in _iter_stream_lines(response):
        if line is _STREAM_HEARTBEAT:
            continue
        if not line:
            continue
        line = line.strip()
        if line.lower().startswith("event:"):
            pending_event = _normalize_event_name(line[6:].strip())
            terminal_event_pending = pending_event == "done"
            continue
        if not line.lower().startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload.upper() == "[DONE]":
            saw_terminal = True
            break
        try:
            obj = json.loads(payload)
        except Exception:
            if terminal_event_pending:
                saw_terminal = True
                break
            continue
        if not isinstance(obj, dict):
            if terminal_event_pending:
                saw_terminal = True
                break
            continue
        event_type = pending_event or _payload_event_name(obj)
        pending_event = None
        terminal_event_pending = False
        _capture_upstream_metadata(upstream_metadata, obj)
        if event_type == "error":
            raise RuntimeError(
                str(obj.get("message") or obj.get("error") or "Trae raw upstream returned an error event")
            )
        if event_type == "token_usage":
            usage = _map_usage(obj.get("usage") or obj)
            continue
        reported_provider = _provider_model_name({**obj, "event": event_type})
        if reported_provider:
            provider_model_name = reported_provider
            _check_provider_model(model, reported_provider)
        reasoning = _payload_reasoning_text(obj)
        raw_response = obj.get("response") if isinstance(obj.get("response"), str) else ""
        embedded_reasoning, visible_response = _split_embedded_reasoning(raw_response)
        reasoning_delta, reasoning_calls = reasoning_text.add(reasoning)
        if embedded_reasoning:
            embedded_delta, embedded_calls = reasoning_text.add(embedded_reasoning)
            reasoning_delta += embedded_delta
            reasoning_calls.extend(embedded_calls)
        response_delta, response_calls = response_text.add(visible_response)
        calls = _calls_from_payload(obj)
        calls.extend(reasoning_calls)
        calls.extend(response_calls)
        if _contains_completed_tool_repeat(calls, completed_tool_signatures):
            saw_completed_repeat = True
        tool_calls.add(
            _filter_for_accumulator(
                tool_calls,
                calls,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            )
        )
        if response_delta:
            full += response_delta
        if obj.get("finish_reason"):
            finish_reason = str(obj.get("finish_reason"))
        if obj.get("usage"):
            usage = _map_usage(obj.get("usage"))
        # See translate_ide_stream: finish_reason/stop_reason can be carried by
        # an intermediate cumulative snapshot, so only an explicit done event
        # ends this response.  [DONE] is handled above.
        if event_type == "done":
            saw_terminal = True
            finish_reason = _payload_finish_reason(obj) or finish_reason
            break
    reasoning_delta, reasoning_calls = reasoning_text.finalize()
    response_delta, response_calls = response_text.finalize()
    final_calls = reasoning_calls + response_calls
    if _contains_completed_tool_repeat(final_calls, completed_tool_signatures):
        saw_completed_repeat = True
    tool_calls.add(
        _filter_for_accumulator(
            tool_calls,
            final_calls,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            completed_tool_signatures,
        )
    )
    if response_delta:
        full += response_delta
    reasoning_summary = (
        compact_reasoning_text(reasoning_text.raw) if include_reasoning else ""
    )
    if terminal_event_pending:
        saw_terminal = True
    if not saw_terminal and require_terminal:
        observed_model_event = bool(
            full
            or reasoning_text.raw
            or tool_calls.has_calls
            or usage is not None
            or provider_model_name
            or saw_completed_repeat
            or (
                isinstance(upstream_metadata, dict)
                and upstream_metadata.get("usage_turn_id")
            )
        )
        raise IncompleteUpstreamResponse(
            "Trae raw upstream ended before its terminal event",
            retryable=not observed_model_event,
            usage=usage,
            observed_model_event=observed_model_event,
        )
    tool_calls.set_calls(_ensure_native_tool_arguments(tool_calls.calls(), usage))
    if not full and not reasoning_summary and not tool_calls.has_calls and saw_completed_repeat:
        raise RepeatedCompletedToolResponse(
            "Trae upstream repeated only already completed tool calls",
            retryable=False,
            usage=usage,
            observed_model_event=True,
        )
    _ensure_required_tool_call(
        tool_choice, tool_calls.has_calls, tool_calls.calls()
    )
    if fail_on_empty and not full and not reasoning_summary and not tool_calls.has_calls:
        observed_model_event = bool(
            usage is not None
            or provider_model_name
            or (
                isinstance(upstream_metadata, dict)
                and upstream_metadata.get("usage_turn_id")
            )
        )
        raise EmptyUpstreamResponse(
            "Trae raw upstream returned no text or tool call",
            retryable=not observed_model_event,
            usage=usage,
            observed_model_event=observed_model_event,
        )
    if not full and not reasoning_summary and not tool_calls.has_calls:
        full = "(trae upstream returned an empty response)"
    if usage is None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(full + reasoning_summary),
            "total_tokens": estimate_tokens(full + reasoning_summary),
        }
    return openai_completion(
        prefix_id,
        model,
        full or None,
        _finish_reason(finish_reason, tool_calls.has_calls),
        usage,
        tool_calls.calls(),
        provider_model_name=provider_model_name,
        reasoning_content=reasoning_summary or None,
    )


async def collect_nonstream_web(
    event_iter,
    model: str,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    completed_tool_signatures: Any = None,
    fail_on_empty: bool = False,
    include_reasoning: bool = False,
) -> dict:
    prefix_id = make_id()
    reasoning_order: list[str] = []
    reasoning_states: dict[str, ProtocolTextAccumulator] = {}
    plan_visible_states: dict[str, ProtocolTextAccumulator] = {}
    message_text = ProtocolTextAccumulator()
    usage = None
    error_event = None
    final_summary = ""
    final_reason = "stop"
    provider_model_name = ""
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)
    legacy_plan_reasoning = False
    async for event, data in event_iter:
        event = _normalize_event_name(event)
        if not isinstance(data, dict):
            data = {}
        if event in {"heartbeat", "keepalive", "ping"}:
            # Non-streaming callers do not need a wire frame, but consuming the
            # event is still important so the upstream connection remains
            # active until its terminal event.
            continue
        if event == "error":
            error_event = data
            break
        reported_provider = _provider_model_name({**data, "event": event})
        if reported_provider:
            provider_model_name = reported_provider
            _check_provider_model(model, reported_provider)
        if event == "token_usage":
            usage = _map_usage(data.get("usage") or data)
            continue
        if event == "plan_item":
            pid = str(data.get("id") or f"plan-{len(reasoning_order)}")
            reasoning_state = reasoning_states.setdefault(
                pid, ProtocolTextAccumulator()
            )
            if pid not in reasoning_order:
                reasoning_order.append(pid)
            plan_reasoning = _web_plan_reasoning(data)
            if not _web_plan_has_split_reasoning(data) and plan_reasoning:
                legacy_plan_reasoning = True
            _, reasoning_calls = reasoning_state.add(plan_reasoning)
            content_state = plan_visible_states.setdefault(
                pid, ProtocolTextAccumulator()
            )
            _, content_calls = content_state.add(
                _web_plan_content(data)
            )
            calls = list(_calls_from_payload(data))
            calls.extend(reasoning_calls)
            calls.extend(content_calls)
            calls = _filter_for_accumulator(
                tool_calls,
                calls,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            )
            tool_calls.add(calls)
            if calls and allowed_tools is not None:
                break
            summary = _web_finish_summary(data)
            if summary:
                final_summary = summary
        if event in {"message", "assistant_message", "response", "text", "output"}:
            text = _web_message_text(data)
            event_reasoning = _payload_reasoning_text(data)
            embedded_reasoning, text = _split_embedded_reasoning(text)
            if event_reasoning or embedded_reasoning:
                key = "__messages__"
                state = reasoning_states.setdefault(key, ProtocolTextAccumulator())
                if key not in reasoning_order:
                    reasoning_order.append(key)
                if event_reasoning:
                    state.add(event_reasoning)
                if embedded_reasoning:
                    state.add(embedded_reasoning)
            if text:
                _, message_calls = message_text.add_snapshot(text)
                filtered_calls = _filter_for_accumulator(
                    tool_calls,
                    message_calls,
                    allowed_tools,
                    tool_choice,
                    parallel_tool_calls,
                    completed_tool_signatures,
                )
                tool_calls.add(filtered_calls)
                if filtered_calls and allowed_tools is not None:
                    break
        if event == "done":
            final_reason = _payload_finish_reason(data) or final_reason
            break
    if error_event:
        raise RuntimeError(f"trae {error_event.get('code','')}: {error_event.get('message','')}")

    for state in reasoning_states.values():
        _, calls = state.finalize()
        tool_calls.add(
            _filter_for_accumulator(
                tool_calls,
                calls,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            )
        )
    for state in plan_visible_states.values():
        _, calls = state.finalize()
        tool_calls.add(
            _filter_for_accumulator(
                tool_calls,
                calls,
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            )
        )
    _, message_calls = message_text.finalize()
    tool_calls.add(
        _filter_for_accumulator(
            tool_calls,
            message_calls,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            completed_tool_signatures,
        )
    )
    content = "\n\n".join(
        state.visible.strip()
        for state in plan_visible_states.values()
        if state.visible.strip()
    )
    message_content = message_text.visible.strip()
    if message_content:
        content = _merge_plan_message_text(content, message_content)
    if final_summary:
        summary_calls = _extract_tool_calls({"response": final_summary})
        summary_calls = _filter_for_accumulator(
            tool_calls,
            summary_calls,
            allowed_tools,
            tool_choice,
            parallel_tool_calls,
            completed_tool_signatures,
        )
        tool_calls.add(summary_calls)
        # ``hold_incomplete`` also applies here: an all-narration summary is
        # dropped rather than appended after the answer. The plan thoughts above
        # already guarantee the turn is not empty.
        summary_text = _visible_text(final_summary).strip()
        filtered_summary = strip_reasoning_narration(
            summary_text, hold_incomplete=True
        ).strip()
        # Keep an all-narration raw summary only when there is no plan trace to
        # summarize.  Otherwise it would re-introduce the chain-of-thought that
        # this translator deliberately keeps out of ordinary content.
        has_plan_reasoning = any(state.raw for state in reasoning_states.values())
        final_summary = filtered_summary or (
            summary_text if not content and not has_plan_reasoning else ""
        )
    # A finish summary can carry the only client tool call. Extract it before
    # validating a required/named choice, otherwise a valid summary call is
    # reported as a missing tool and the non-stream request fails.
    _ensure_required_tool_call(
        tool_choice, tool_calls.has_calls, tool_calls.calls()
    )
    if not tool_calls.has_calls:
        if not content:
            content = final_summary
        elif final_summary and final_summary.strip().startswith(content.strip()):
            content = final_summary.strip()
        elif final_summary and not content.rstrip().endswith(final_summary.rstrip()):
            content = content.rstrip() + "\n\n" + final_summary
    reasoning_raw = "\n".join(
        reasoning_states[key].raw for key in reasoning_order if reasoning_states[key].raw
    )
    reasoning_summary = compact_reasoning_text(reasoning_raw)
    public_reasoning = reasoning_summary if include_reasoning else ""
    if (
        not content
        and not tool_calls.has_calls
        and reasoning_summary
        and not include_reasoning
        and not legacy_plan_reasoning
    ):
        # Split layout: never promote the reasoning trace into content.
        public_reasoning = reasoning_summary
    if (
        not content
        and not public_reasoning
        and not tool_calls.has_calls
        and reasoning_summary
        and not include_reasoning
    ):
        # Compatibility fallback for remote versions that return the final
        # answer only through plan_item.thought.  Keep it compact rather than
        # exposing the complete working trace.
        content = reasoning_summary
    if fail_on_empty and not content and not public_reasoning and not tool_calls.has_calls:
        quota_exhausted = _is_account_quota_error(error_event)
        # A provider/model metadata event is not output.  In particular,
        # Trae sends it immediately before a 4008 quota error; treating it as
        # activity prevents the dispatcher from rotating credentials.
        observed_model_event = bool(
            usage is not None
            or reasoning_raw
            or (provider_model_name and not quota_exhausted)
        )
        message = "Trae remote upstream returned no text or tool call"
        if error_event:
            message = (
                f"Trae remote upstream error: {error_event.get('code', '')}: "
                f"{error_event.get('message', '')}"
            )
        raise EmptyUpstreamResponse(
            message,
            retryable=not observed_model_event,
            usage=usage,
            observed_model_event=observed_model_event,
            account_quota_exhausted=quota_exhausted,
        )
    if not content and not public_reasoning and not tool_calls.has_calls:
        content = "(trae upstream returned an empty response)"
    if usage is None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(content + public_reasoning),
            "total_tokens": estimate_tokens(content + public_reasoning),
        }
    return openai_completion(
        prefix_id,
        model,
        content or None,
        _finish_reason(final_reason, tool_calls.has_calls),
        usage,
        tool_calls.calls(),
        provider_model_name=provider_model_name,
        reasoning_content=public_reasoning or None,
    )
