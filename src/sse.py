"""
sse.py - parse Trae upstream events into OpenAI-compatible responses.

Tool calls are always serialized back to the API client. This module never
executes a tool, command, or filesystem operation.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Optional

from .cli_client import extract_result_text as _cli_extract_text
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


class RepeatedCompletedToolResponse(EmptyUpstreamResponse):
    """Raised when the only upstream output repeats an already completed call."""


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
    return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"


def openai_completion(
    prefix_id: str,
    model: str,
    content: Optional[str],
    finish_reason: str = "stop",
    usage=None,
    tool_calls: Optional[list[dict]] = None,
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

    prompt = number("prompt_tokens", "input_tokens", "inputTokens")
    completion = number("completion_tokens", "output_tokens", "outputTokens")
    total = number("total_tokens", "totalTokens") or prompt + completion
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
    )
    if credits is not None:
        mapped["credits_consumed"] = credits
    return mapped


def _payload_finish_reason(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for key in ("finish_reason", "stop_reason"):
        value = data.get(key)
        if value:
            return str(value)
    message = data.get("message")
    if isinstance(message, dict):
        for key in ("finish_reason", "stop_reason"):
            value = message.get(key)
            if value:
                return str(value)
        response_meta = message.get("response_meta")
        if isinstance(response_meta, dict):
            for key in ("finish_reason", "stop_reason"):
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
        function = (
            tool_choice.get("function")
            if tool_choice.get("type") == "function"
            else tool_choice
        )
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    return None


def _required_tool_error(tool_choice: Any, has_tool_calls: bool) -> Optional[dict]:
    required = _required_tool_label(tool_choice)
    if not required or has_tool_calls:
        return None
    return {
        "message": f"Trae upstream did not return the required tool call: {required}",
        "type": "api_error",
    }


def _ensure_required_tool_call(tool_choice: Any, has_tool_calls: bool) -> None:
    error = _required_tool_error(tool_choice, has_tool_calls)
    if error:
        raise RuntimeError(error["message"])


class ToolCallAccumulator:
    """Deduplicate full/cumulative calls and emit OpenAI streaming deltas."""

    def __init__(self, max_calls: Optional[int] = None):
        self._calls: dict[str, dict] = {}
        self._order: list[str] = []
        self._indexes: dict[str, int] = {}
        self._max_calls = max_calls

    @property
    def has_calls(self) -> bool:
        return bool(self._calls)

    def prepare(self, calls: Any) -> list[dict]:
        if not isinstance(calls, list):
            return []
        prepared: list[dict] = []
        for fallback_index, raw in enumerate(calls):
            call = _normalize_tool_call(raw, index=fallback_index)
            if not call:
                continue
            function = call.get("function") or {}
            if not function.get("name"):
                existing_key = call.get("id")
                existing = self._calls.get(existing_key)
                requested_index = call.get("index")
                if existing is None and isinstance(requested_index, int):
                    for key, index in self._indexes.items():
                        if index == requested_index:
                            existing_key = key
                            existing = self._calls.get(key)
                            break
                existing_function = existing.get("function") if existing else None
                if isinstance(existing_function, dict) and existing_function.get("name"):
                    function["name"] = existing_function["name"]
                    call["id"] = existing_key
                    call["_synthetic_id"] = False
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
            key = call_id
            if not function.get("name"):
                existing = self._calls.get(key)
                requested_index = call.get("index")
                if existing is None and isinstance(requested_index, int):
                    for existing_key, existing_index in self._indexes.items():
                        if existing_index == requested_index:
                            key = existing_key
                            call_id = existing_key
                            call["id"] = existing_key
                            existing = self._calls.get(existing_key)
                            break
                existing_function = existing.get("function") if existing else None
                if not isinstance(existing_function, dict) or not existing_function.get(
                    "name"
                ):
                    continue
                function["name"] = existing_function["name"]
            requested_index = call.get("index")
            if call.get("_synthetic_id"):
                for existing_key in self._order:
                    existing = self._calls[existing_key]
                    if not existing.get("_synthetic_id"):
                        continue
                    existing_function = existing.get("function") or {}
                    if existing_function.get("name") != function.get("name"):
                        continue
                    same_index = (
                        existing.get("_explicit_index") is True
                        and call.get("_explicit_index") is True
                        and existing.get("_source_index")
                        == call.get("_source_index")
                    )
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
            if current_args != previous_args:
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


def _calls_from_payload(data: Any) -> list[dict]:
    if not isinstance(data, dict):
        return []
    calls = _extract_tool_calls(data)
    tool_info = data.get("tool_call_info")
    if isinstance(tool_info, dict):
        item = dict(tool_info)
        item.setdefault("id", tool_info.get("tool_call_id") or data.get("id"))
        call = _normalize_tool_call(item, index=len(calls))
        if call:
            calls.append(call)
    deduped: dict[str, dict] = {}
    for call in calls:
        deduped[call["id"]] = call
    return list(deduped.values())


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
    if names is None:
        return calls
    if tool_choice == "none" or not names:
        return []
    selected: Optional[str] = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if tool_choice.get("type") == "function" else tool_choice
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            selected = fn["name"].strip()
    filtered: list[dict] = []
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or name.strip() not in names:
            continue
        if selected and name.strip() != selected:
            continue
        call_id = str(call.get("id") or "")
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


def _hold_incomplete_tool_block(text: str) -> str:
    """Keep a split XML tool block out of visible text until it is complete."""
    lowered = text.lower()
    pairs = (
        ("<opencode_tool_call>", "</opencode_tool_call>"),
        ("<tool_use>", "</tool_use>"),
        ("<tool_call", "</tool_call>"),
        ("<tool_cell", "</tool_cell>"),
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
    return text[:earliest]


class ProtocolTextAccumulator:
    """Merge cumulative or incremental text while hiding serialized tool calls."""

    def __init__(self):
        self.raw = ""
        self.visible = ""

    def _result(self) -> tuple[str, list[dict]]:
        visible = _hold_incomplete_tool_block(_visible_text(self.raw))
        delta = _cli_text_delta(self.visible, visible)
        self.visible = visible
        calls = _extract_tool_calls({"response": self.raw}) if self.raw else []
        return delta, calls

    def add_delta(self, value: Any) -> tuple[str, list[dict]]:
        text = value if isinstance(value, str) else ""
        if text:
            self.raw += text
        return self._result()

    def add_snapshot(self, value: Any) -> tuple[str, list[dict]]:
        text = value if isinstance(value, str) else ""
        if text and not self.raw.startswith(text):
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


async def translate_ide_stream(
    response,
    model: str,
    forward_usage: bool = True,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    fail_on_empty: bool = False,
    completed_tool_signatures: Any = None,
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
    saw_completed_repeat = False
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
        if raw.startswith("event:"):
            pending_event = raw[6:].strip()
            continue
        if not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        event_type = obj.get("event") or pending_event
        pending_event = None
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

        reasoning = obj.get("reasoning_content") if isinstance(obj.get("reasoning_content"), str) else ""
        raw_response = obj.get("response") if isinstance(obj.get("response"), str) else ""
        reasoning_delta, reasoning_calls = reasoning_text.add(reasoning)
        response_delta, response_calls = response_text.add(raw_response)
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

        text_delta = tracker.merge(reasoning_delta, response_delta)
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
        # Trae sometimes attaches a finish_reason to an intermediate
        # cumulative snapshot.  Only an explicit done event, stop_reason, or
        # the [DONE] sentinel above terminates the upstream stream; otherwise
        # consuming the rest is required to avoid truncating the answer.
        if event_type == "done" or obj.get("stop_reason"):
            final_reason = str(obj.get("finish_reason") or obj.get("stop_reason") or final_reason)
            break

    if content_count == 0 and not tool_calls.has_calls and saw_completed_repeat:
        raise RepeatedCompletedToolResponse(
            "Trae upstream repeated only already completed tool calls"
        )
    if fail_on_empty and content_count == 0 and not tool_calls.has_calls:
        raise EmptyUpstreamResponse("Trae raw upstream returned no text or tool call")

    required_error = _required_tool_error(tool_choice, tool_calls.has_calls)
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
    if content_count == 0 and not tool_calls.has_calls:
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
    )
    yield "data: [DONE]\n\n"


def _web_plan_text(data: dict) -> str:
    return data.get("thought") or data.get("reasoning_content") or ""


def _web_finish_summary(data: dict) -> str:
    tci = data.get("tool_call_info") or {}
    if isinstance(tci, dict) and tci.get("name") == "finish":
        params = tci.get("params") or {}
        if isinstance(params, dict):
            return str(params.get("summary") or "")
    return ""


async def translate_web_events(
    event_iter,
    model: str,
    forward_usage: bool = True,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    completed_tool_signatures: Any = None,
):
    prefix_id = make_id()
    order: list[str] = []
    thoughts: dict[str, str] = {}
    streamed_text = ""
    usage = None
    error_event = None
    final_summary = ""
    final_reason = "stop"
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)

    yield openai_chunk(prefix_id, model, {"role": "assistant"})
    async for event, data in event_iter:
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
        if event == "token_usage":
            usage = _map_usage(data.get("usage") or data)
            continue
        if event == "plan_item":
            emitted_tool = False
            for chunk in _emit_tool_deltas(
                prefix_id,
                model,
                tool_calls,
                _calls_from_payload(data),
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            ):
                emitted_tool = True
                yield chunk
            if emitted_tool and allowed_tools is not None:
                break
            pid = str(data.get("id") or "")
            if pid:
                thought = _hold_incomplete_tool_block(
                    _visible_text(_web_plan_text(data))
                )
                if pid not in thoughts:
                    order.append(pid)
                previous = thoughts.get(pid, "")
                piece = ""
                if len(thought) >= len(previous):
                    thoughts[pid] = thought
                    piece = _cli_text_delta(previous, thought)
                if piece:
                    streamed_text += piece
                    yield openai_chunk(prefix_id, model, {"content": piece})
            summary = _web_finish_summary(data)
            if summary:
                final_summary = summary
        if event == "done":
            final_reason = _payload_finish_reason(data) or final_reason
            break

    # Do not present a web agent's remote tool result as the external client's
    # tool result. The API client owns execution and the following turn.
    if not tool_calls.has_calls:
        if not streamed_text and final_summary:
            yield openai_chunk(prefix_id, model, {"content": final_summary})
        elif final_summary:
            full = "".join(thoughts.get(item, "") for item in order)
            summary = final_summary.rstrip()
            if not full.rstrip().endswith(summary) and not streamed_text.rstrip().endswith(summary):
                yield openai_chunk(prefix_id, model, {"content": "\n\n" + final_summary})

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
        required_error = _required_tool_error(tool_choice, tool_calls.has_calls)
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
        yield openai_chunk(
            prefix_id,
            model,
            {},
            finish_reason=_finish_reason(final_reason, tool_calls.has_calls),
            usage=usage if forward_usage else None,
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
):
    """Translate Trae CLI JSON/text output without executing returned calls."""
    prefix_id = make_id()
    text_state = ProtocolTextAccumulator()
    usage = None
    saw_output = False
    final_reason = "stop"
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)

    yield openai_chunk(prefix_id, model, {"role": "assistant"})
    async for event in event_iter:
        if event.type == "error":
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
        if event.type == "text":
            text_delta, text_calls = text_state.add_delta(event.text or "")
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
        if event.type != "json" or not event.data:
            continue
        result = event.data
        result_usage = _cli_extract_usage(result)
        if result_usage:
            usage = result_usage
        result_reason = _payload_finish_reason(result)
        if result_reason:
            final_reason = result_reason
        text_delta, text_calls = text_state.add_snapshot(_cli_extract_text(result))
        calls = _extract_tool_calls(result)
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

    required_error = _required_tool_error(tool_choice, tool_calls.has_calls)
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
) -> dict:
    prefix_id = make_id()
    usage = None
    final_reason = "stop"
    text_state = ProtocolTextAccumulator()
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)
    async for event in event_iter:
        if event.type == "error":
            raise RuntimeError(event.error or "Trae CLI failed")
        if event.type == "text" and event.text:
            _, text_calls = text_state.add_delta(event.text)
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
        elif event.type == "json" and event.data:
            _, text_calls = text_state.add_snapshot(
                _cli_extract_text(event.data)
            )
            calls = _extract_tool_calls(event.data)
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
    _ensure_required_tool_call(tool_choice, tool_calls.has_calls)
    content = text_state.visible.strip()
    if not content and not tool_calls.has_calls:
        content = "(trae cli returned an empty response)"
    if usage is None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(content),
            "total_tokens": estimate_tokens(content),
        }
    return openai_completion(
        prefix_id,
        model,
        content or None,
        _finish_reason(final_reason, tool_calls.has_calls),
        usage,
        tool_calls.calls(),
    )


async def collect_nonstream_ide(
    response,
    model: str,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    fail_on_empty: bool = False,
    completed_tool_signatures: Any = None,
) -> dict:
    prefix_id = make_id()
    tracker = ThinkingTracker()
    full = ""
    reasoning_text = ProtocolTextAccumulator()
    response_text = ProtocolTextAccumulator()
    finish_reason = "stop"
    usage = None
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)
    pending_event = None
    saw_completed_repeat = False
    async for line in _iter_stream_lines(response):
        if line is _STREAM_HEARTBEAT:
            continue
        if not line:
            continue
        line = line.strip()
        if line.startswith("event:"):
            pending_event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        event_type = obj.get("event") or pending_event
        pending_event = None
        if event_type == "error":
            raise RuntimeError(
                str(obj.get("message") or obj.get("error") or "Trae raw upstream returned an error event")
            )
        reasoning = obj.get("reasoning_content") or ""
        raw_response = obj.get("response") or ""
        reasoning_delta, reasoning_calls = reasoning_text.add(reasoning)
        response_delta, response_calls = response_text.add(raw_response)
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
        if reasoning_delta or response_delta:
            full += tracker.merge(reasoning_delta, response_delta)
        if obj.get("finish_reason"):
            finish_reason = str(obj.get("finish_reason"))
        if obj.get("usage"):
            usage = _map_usage(obj.get("usage"))
        # See translate_ide_stream: finish_reason can be carried by an
        # intermediate cumulative snapshot, so only explicit done/stop_reason
        # ends this response.  [DONE] is handled above.
        if event_type == "done" or obj.get("stop_reason"):
            finish_reason = str(
                obj.get("finish_reason") or obj.get("stop_reason") or finish_reason
            )
            break
    if not full and not tool_calls.has_calls and saw_completed_repeat:
        raise RepeatedCompletedToolResponse(
            "Trae upstream repeated only already completed tool calls"
        )
    if fail_on_empty and not full and not tool_calls.has_calls:
        raise EmptyUpstreamResponse("Trae raw upstream returned no text or tool call")
    _ensure_required_tool_call(tool_choice, tool_calls.has_calls)
    if not full and not tool_calls.has_calls:
        full = "(trae upstream returned an empty response)"
    if usage is None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(full),
            "total_tokens": estimate_tokens(full),
        }
    return openai_completion(
        prefix_id,
        model,
        full or None,
        _finish_reason(finish_reason, tool_calls.has_calls),
        usage,
        tool_calls.calls(),
    )


async def collect_nonstream_web(
    event_iter,
    model: str,
    allowed_tools: Any = None,
    tool_choice: Any = None,
    parallel_tool_calls: Any = None,
    completed_tool_signatures: Any = None,
) -> dict:
    prefix_id = make_id()
    order: list[str] = []
    thoughts: dict[str, str] = {}
    usage = None
    error_event = None
    final_summary = ""
    final_reason = "stop"
    tool_calls = ToolCallAccumulator(1 if parallel_tool_calls is False else None)
    async for event, data in event_iter:
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
        if event == "token_usage":
            usage = _map_usage(data.get("usage") or data)
            continue
        if event == "plan_item":
            calls = _filter_for_accumulator(
                tool_calls,
                _calls_from_payload(data),
                allowed_tools,
                tool_choice,
                parallel_tool_calls,
                completed_tool_signatures,
            )
            tool_calls.add(calls)
            if calls and allowed_tools is not None:
                break
            pid = str(data.get("id") or "")
            if pid:
                thought = _hold_incomplete_tool_block(
                    _visible_text(_web_plan_text(data))
                )
                if pid not in thoughts:
                    order.append(pid)
                if len(thought) >= len(thoughts.get(pid, "")):
                    thoughts[pid] = thought
            summary = _web_finish_summary(data)
            if summary:
                final_summary = summary
        if event == "done":
            final_reason = _payload_finish_reason(data) or final_reason
            break
    if error_event:
        raise RuntimeError(f"trae {error_event.get('code','')}: {error_event.get('message','')}")
    _ensure_required_tool_call(tool_choice, tool_calls.has_calls)
    content = "".join(thoughts.get(item, "") for item in order)
    if not tool_calls.has_calls:
        if not content:
            content = final_summary
        elif final_summary and not content.rstrip().endswith(final_summary.rstrip()):
            content = content.rstrip() + "\n\n" + final_summary
    if not content and not tool_calls.has_calls:
        content = "(trae upstream returned an empty response)"
    if usage is None:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(content),
            "total_tokens": estimate_tokens(content),
        }
    return openai_completion(
        prefix_id,
        model,
        content or None,
        _finish_reason(final_reason, tool_calls.has_calls),
        usage,
        tool_calls.calls(),
    )
