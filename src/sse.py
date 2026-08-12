"""
sse.py - 解析 Trae 上游 SSE 并转换为 OpenAI 兼容格式

事件类型：
- request_wait_in_queue: 排队提示
- output: { response, reasoning_content, finish_reason }
- done: { finish_reason }
- token_usage / plan_item: OmniRoute 网页版 remote 事件
- Trae CLI: 子进程 JSON/text 流事件
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from .cli_client import extract_result_text as _cli_extract_text
from .cli_client import extract_usage as _cli_extract_usage

logger = logging.getLogger(__name__)

THINK_OPEN = " thinking\n\n"
THINK_CLOSE = "\n response\n\n"


def make_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:20]}"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    total = 0.0
    for ch in text:
        if ord(ch) > 0x2000:
            total += 1.5
        else:
            total += 0.25
    return max(1, int(total + 0.999))


def openai_chunk(prefix_id: str, model: str, delta: dict, finish_reason=None, usage=None, error=None):
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


def openai_completion(prefix_id: str, model: str, content: str, finish_reason: str = "stop", usage=None):
    resp = {
        "id": prefix_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        resp["usage"] = usage
    else:
        resp["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(content),
            "total_tokens": estimate_tokens(content),
        }
    return resp


class ThinkingTracker:
    """把 reasoning_content 和 response 合并成标准思维链文本"""

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
    """从一行 SSE 数据串提取 data 值"""
    data = data.strip()
    if not data.startswith("data:"):
        return None
    return data[5:].strip()


async def translate_ide_stream(response, model: str, forward_usage: bool = True):
    """把 /api/ide/v1/chat 的 SSE 翻译成 OpenAI SSE。

    response 是一个带 .iter_lines() 的 httpx 同步响应。
    """
    prefix_id = make_id()
    tracker = ThinkingTracker()
    completion_bytes = 0
    content_count = 0
    finished = False

    import httpx

    if isinstance(response, httpx.Response):
        stream_iter = response.iter_lines()
    else:
        stream_iter = response

    lines = stream_iter

    yield f"data: {json.dumps({'id': prefix_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"

    for line in lines:
        if line is None:
            continue
        raw = line.strip()
        if not raw:
            continue

        if raw.startswith("event: "):
            event = raw[7:].strip()
            # 下一行是 data
            continue

        if raw.startswith("data: "):
            event = None
            payload = raw[5:].strip()
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            event_type = obj.get("event") if isinstance(obj, dict) else None
            reasoning = obj.get("reasoning_content", "") if isinstance(obj, dict) else ""
            response_text = obj.get("response", "") if isinstance(obj, dict) else ""
            finish = obj.get("finish_reason", "") if isinstance(obj, dict) else ""
            if event_type == "request_wait_in_queue" or isinstance(obj, dict) and obj.get("position") is not None:
                pos = obj.get("position", 0)
                message = f"排队中，当前位置：{pos}\n"
                yield openai_chunk(prefix_id, model, {"content": message})
                continue
            if event_type == "done" or (obj and obj.get("stop_reason")):
                finish = obj.get("finish_reason") or obj.get("stop_reason") or "stop"
                finished = True
                yield openai_chunk(prefix_id, model, {}, finish_reason=finish)
                if forward_usage:
                    usage = obj.get("usage")
                    if usage:
                        usage = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                    else:
                        usage = {
                            "prompt_tokens": 0,
                            "completion_tokens": completion_bytes // 4 + 1,
                            "total_tokens": completion_bytes // 4 + 1,
                        }
                    yield openai_chunk(prefix_id, model, {}, finish_reason=finish, usage=usage)
                yield "data: [DONE]\n\n"
                return

            if event_type == "output" or isinstance(obj, dict) and (reasoning or response_text):
                delta = tracker.merge(reasoning, response_text)
                if delta:
                    completion_bytes += len(delta.encode("utf-8"))
                    content_count += 1
                    yield openai_chunk(prefix_id, model, {"content": delta})

    # 流结束兜底
    if not finished:
        if content_count == 0:
            yield openai_chunk(prefix_id, model, {"content": "(trae upstream returned an empty response)"})
        finish_reason = "stop"
        if forward_usage:
            yield openai_chunk(
                prefix_id, model, {},
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": completion_bytes // 4 + 1,
                    "total_tokens": completion_bytes // 4 + 1,
                },
            )
        yield "data: [DONE]\n\n"


def _web_plan_text(data: dict) -> str:
    """Best visible text from a Trae web plan_item event."""
    return data.get("thought") or data.get("reasoning_content") or ""


def _web_finish_summary(data: dict) -> str:
    """Extract the final assistant summary from a finish tool_call."""
    tci = data.get("tool_call_info") or {}
    if tci.get("name") == "finish":
        params = tci.get("params") or {}
        return str(params.get("summary") or "")
    return ""


async def translate_web_events(event_iter, model: str, forward_usage: bool = True):
    """把 OmniRoute 网页版 remote 会话的 plan_item/token_usage/done 事件转成 OpenAI SSE"""
    prefix_id = make_id()
    order: list[str] = []
    thoughts: dict[str, str] = {}
    sent = 0
    usage = None
    error_event = None
    final_summary = ""

    yield openai_chunk(prefix_id, model, {"role": "assistant"})

    async for event, data in event_iter:
        if event == "error":
            error_event = data
            break
        if event == "token_usage":
            usage = data
            continue
        if event == "plan_item":
            pid = data.get("id")
            if pid:
                thought = _web_plan_text(data)
                if pid not in thoughts:
                    order.append(pid)
                if len(thought) >= len(thoughts.get(pid, "")):
                    thoughts[pid] = thought
                finish_summary = _web_finish_summary(data)
                if finish_summary:
                    final_summary = finish_summary
                full = "".join(thoughts[p] for p in order)
                piece = full[sent:]
                sent = len(full)
                if piece:
                    yield openai_chunk(prefix_id, model, {"content": piece})
        if event == "done":
            break

    if sent == 0 and final_summary:
        yield openai_chunk(prefix_id, model, {"content": final_summary})
    elif final_summary:
        # if reasoning content was streamed, append summary as final delta
        full = "".join(thoughts.get(p, "") for p in order)
        if not full.rstrip().endswith(final_summary.rstrip()):
            yield openai_chunk(prefix_id, model, {"content": "\n\n" + final_summary})

    if error_event:
        yield openai_chunk(
            prefix_id, model, {},
            error={"message": f"trae {error_event.get('code', '')}: {error_event.get('message', '')}", "type": "api_error"},
        )
    else:
        if forward_usage and usage:
            yield openai_chunk(prefix_id, model, {}, finish_reason="stop", usage={
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "total_tokens": usage.get("total_tokens", 0) or 0,
            })
        else:
            yield openai_chunk(prefix_id, model, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


def _cli_text_delta(previous: str, current: str) -> str:
    if not current:
        return ""
    if current.startswith(previous):
        return current[len(previous):]
    return current


async def translate_cli_stream(event_iter, model: str, forward_usage: bool = True):
    """Translate Trae CLI JSON/text stream events into OpenAI SSE chunks."""
    prefix_id = make_id()
    last_text = ""
    usage = None
    saw_output = False

    yield openai_chunk(prefix_id, model, {"role": "assistant"})

    async for event in event_iter:
        if event.type == "error":
            if not saw_output:
                raise RuntimeError(event.error or "Trae CLI failed")
            yield openai_chunk(
                prefix_id, model, {},
                error={"message": event.error or "Trae CLI failed", "type": "api_error"},
            )
            yield "data: [DONE]\n\n"
            return
        if event.type == "text":
            if event.text:
                saw_output = True
                yield openai_chunk(prefix_id, model, {"content": event.text})
            continue
        if event.type != "json" or not event.data:
            continue
        result = event.data
        result_usage = _cli_extract_usage(result)
        if result_usage:
            usage = result_usage
        text = _cli_extract_text(result)
        delta = _cli_text_delta(last_text, text)
        if delta:
            saw_output = True
            last_text = text
            yield openai_chunk(prefix_id, model, {"content": delta})

    if not saw_output:
        yield openai_chunk(prefix_id, model, {"content": "(trae cli returned an empty response)"})
    if forward_usage and usage:
        yield openai_chunk(prefix_id, model, {}, finish_reason="stop", usage=usage)
    else:
        yield openai_chunk(prefix_id, model, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def collect_nonstream_cli(event_iter, model: str) -> dict:
    """Collect a complete Trae CLI response into a non-streaming completion."""
    prefix_id = make_id()
    parts: list[str] = []
    usage = None
    last_json_text = ""
    async for event in event_iter:
        if event.type == "error":
            raise RuntimeError(event.error or "Trae CLI failed")
        if event.type == "text" and event.text:
            parts.append(event.text)
        elif event.type == "json" and event.data:
            text = _cli_extract_text(event.data)
            if text:
                delta = _cli_text_delta(last_json_text, text)
                if delta:
                    parts.append(delta)
                last_json_text = text
            event_usage = _cli_extract_usage(event.data)
            if event_usage:
                usage = event_usage
    content = "".join(parts).strip()
    if not content:
        content = "(trae cli returned an empty response)"
    if not usage:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(content),
            "total_tokens": estimate_tokens(content),
        }
    return openai_completion(prefix_id, model, content, "stop", usage)


async def collect_nonstream_ide(response, model: str) -> dict:
    """非流式：收集 /api/ide/v1/chat 的完整 SSE"""
    prefix_id = make_id()
    tracker = ThinkingTracker()
    full = ""
    finish_reason = "stop"
    for line in response.iter_lines():
        if not line:
            continue
        line = line.strip()
        if line.startswith("event: "):
            continue
        if line.startswith("data: "):
            try:
                obj = json.loads(line[6:].strip())
            except Exception:
                continue
            reasoning = obj.get("reasoning_content") or ""
            response_text = obj.get("response") or ""
            if obj.get("response") or obj.get("reasoning_content"):
                full += tracker.merge(reasoning, response_text)
            if obj.get("finish_reason"):
                finish_reason = obj.get("finish_reason")
    if not full:
        full = "(trae upstream returned an empty response)"
    return openai_completion(prefix_id, model, full, finish_reason, {
        "prompt_tokens": 0,
        "completion_tokens": estimate_tokens(full),
        "total_tokens": estimate_tokens(full),
    })


async def collect_nonstream_web(event_iter, model: str) -> dict:
    """非流式：收集网页版 remote 会话事件"""
    prefix_id = make_id()
    order: list[str] = []
    thoughts: dict[str, str] = {}
    usage = None
    error_event = None
    final_summary = ""
    async for event, data in event_iter:
        if event == "error":
            error_event = data
            break
        if event == "token_usage":
            usage = data
            continue
        if event == "plan_item":
            pid = data.get("id")
            if pid:
                thought = _web_plan_text(data)
                if pid not in order:
                    order.append(pid)
                if len(thought) >= len(thoughts.get(pid, "")):
                    thoughts[pid] = thought
                finish_summary = _web_finish_summary(data)
                if finish_summary:
                    final_summary = finish_summary
        if event == "done":
            break
    if error_event:
        raise RuntimeError(f"trae {error_event.get('code','')}: {error_event.get('message','')}")
    content = "".join(thoughts.get(p, "") for p in order)
    if not content:
        content = final_summary
    if not content:
        content = "(trae upstream returned an empty response)"
    # append finish summary if not already included in content
    if final_summary and not content.rstrip().endswith(final_summary.rstrip()):
        content = content.rstrip() + "\n\n" + final_summary
    if usage:
        usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }
    else:
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_tokens(content),
            "total_tokens": estimate_tokens(content),
        }
    return openai_completion(prefix_id, model, content, "stop", usage)
