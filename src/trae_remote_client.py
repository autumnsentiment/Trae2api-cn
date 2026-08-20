"""9router-style Trae remote session transport.

The implementation follows 9router's provider executor, but keeps this
relay's existing CN endpoint and account store.  It only forwards the remote
session and parses SSE; tools remain owned by the API caller.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Mapping, Optional

import httpx

from . import auth, trae_client


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://trae-api-cn.mchost.guru/api/remote/v1"
DEFAULT_ORIGIN = "https://solo.trae.cn"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)


def base_url(options: Optional[Mapping[str, Any]] = None) -> str:
    options = options or {}
    configured = options.get("base_url") or options.get("baseURL")
    value = configured or os.environ.get("TRAE_WEB_BASE_URL") or DEFAULT_BASE_URL
    return str(value).rstrip("/")


def _provider_specific(options: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    options = options or {}
    value = options.get("provider_specific") or options.get("providerSpecificData")
    return dict(value) if isinstance(value, Mapping) else auth.get_psd()


def build_headers(
    token: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
    stream: bool = True,
) -> dict[str, str]:
    psd = _provider_specific(options)
    base = base_url(options).lower()
    intl = "trae.ai" in base and "trae-api-cn" not in base
    language_default = "en" if intl else "zh-CN"
    region_default = "US" if intl else "CN"
    origin_default = "https://solo.trae.ai" if intl else DEFAULT_ORIGIN
    origin = os.environ.get("TRAE_WEB_ORIGIN") or origin_default
    return {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "Content-Type": "application/json",
        "X-Trae-Client-Type": os.environ.get("TRAE_WEB_CLIENT_TYPE", "web"),
        "X-Preferenced-Language": str(
            psd.get("appLanguage") or os.environ.get("TRAE_WEB_LANGUAGE") or language_default
        ),
        "x-user-region": str(
            psd.get("userRegion") or os.environ.get("TRAE_WEB_USER_REGION") or region_default
        ),
        "Origin": origin,
        "Referer": origin.rstrip("/") + "/",
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/event-stream" if stream else "application/json",
    }


def resolve_mode(model: str) -> tuple[str, str, str]:
    value = (model or "").strip()
    lowered = value.lower()
    if lowered in {"work", "auto-work", "solo-work"}:
        return "work", "auto", ""
    if not value or lowered == "auto":
        return "code", "auto", ""
    return "code", "manual", trae_client.convert_model_name(value)


def common_params(
    psd: Mapping[str, Any],
    mode: str,
    session_id: str = "",
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> str:
    base = base_url(options).lower()
    intl = "trae.ai" in base and "trae-api-cn" not in base
    default_language = "en" if intl else "zh-CN"
    default_scope = "marscode-us" if intl else "marscode-cn"
    default_region = "US-East" if intl else "cn"
    params: dict[str, Any] = {
        "language": "en-us" if intl else "zh-cn",
        "app_language": psd.get("appLanguage") or default_language,
        "quality": "stable",
        "app_version": psd.get("appVersion") or "1.0.0.1229",
        "web_id": psd.get("webId") or "",
        "user_identity": psd.get("userIdentity") or "Free",
        "is_freshman": "0",
        "biz_user_id": psd.get("bizUserId") or "",
        "user_unique_id": psd.get("userUniqueId") or "",
        "scope": psd.get("scope") or default_scope,
        "tenant": psd.get("tenant") or "marscode",
        "region": psd.get("region") or default_region,
        "aiRegion": psd.get("aiRegion") or psd.get("region") or default_region,
        "is_privacy_mode": 0,
        "privacy_mode": "off",
        "solo_chat_mode": mode,
    }
    if session_id:
        params["biz_session_id"] = session_id
    return json.dumps(params, ensure_ascii=False)


def flatten_query(messages: list[dict[str, Any]]) -> str:
    return trae_client.flatten_query(messages)


async def create_session(
    client: httpx.AsyncClient,
    token: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> tuple[str, str]:
    if not token:
        raise RuntimeError("No Cloud-IDE-JWT token available")
    mode, strategy, model_name = resolve_mode(model)
    body = {
        "mode": mode,
        "environment_id": "default",
        "initial_message": {
            "chat_session_id": "",
            "content": [],
            "query": flatten_query(messages),
            "model_name": model_name,
            "agent_type": "solo_agent_remote",
            "model_selection_strategy": strategy,
            "common_params": common_params(
                _provider_specific(options), mode, options=options
            ),
        },
        "env": "remote",
        "auto_create_project": False,
        "origin": "web",
    }
    logger.info(
        "remote upstream request model=%s url=%s body_bytes=%d messages=%d query_chars=%d",
        model,
        f"{base_url(options)}/chat_sessions",
        len(json.dumps(body, ensure_ascii=False, separators=(",", ":"))),
        len(messages),
        len(str(body["initial_message"].get("query") or "")),
    )
    response = await client.post(
        f"{base_url(options)}/chat_sessions",
        headers=build_headers(token, options=options, stream=False),
        json=body,
        timeout=float(os.environ.get("TRAE_WEB_CONNECT_TIMEOUT", "60")),
    )
    text = response.text
    if response.status_code >= 400:
        raise RuntimeError(f"Trae remote create_session [{response.status_code}]: {text[:500]}")
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Trae remote invalid create_session response: {text[:500]}") from exc
    if payload.get("code") not in (None, 0) and not payload.get("data"):
        raise RuntimeError(f"Trae remote create_session: {payload}")
    data = payload.get("data") or payload
    session_id = str(data.get("chat_session_id") or "")
    message_id = str(data.get("message_id") or "")
    if not session_id or not message_id:
        raise RuntimeError(f"Trae remote create_session missing ids: {payload}")
    return session_id, message_id


async def stream_events(
    client: httpx.AsyncClient,
    token: str,
    session_id: str,
    message_id: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    url = f"{base_url(options)}/chat_sessions/{session_id}/events?reply_to_message_id={message_id}"
    timeout = httpx.Timeout(
        float(os.environ.get("TRAE_WEB_STREAM_TIMEOUT", os.environ.get("STREAM_TIMEOUT", "300"))),
        connect=float(os.environ.get("TRAE_WEB_CONNECT_TIMEOUT", "60")),
    )
    async with client.stream(
        "GET", url, headers=build_headers(token, options=options, stream=True), timeout=timeout
    ) as response:
        if response.status_code >= 400:
            body = await response.aread()
            raise RuntimeError(f"Trae remote events [{response.status_code}]: {body[:500]}")
        event_name: Optional[str] = None
        buffer = ""
        async for raw in response.aiter_bytes():
            buffer += raw.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line == "":
                    event_name = None
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    yield "done", {}
                    return
                try:
                    data = json.loads(payload)
                except Exception:
                    data = {"_raw": payload}
                if not isinstance(data, dict):
                    data = {"value": data}
                yield event_name or str(data.get("event") or "message"), data
                event_name = None
        if buffer.strip().startswith("data:"):
            payload = buffer.strip()[5:].strip()
            try:
                data = json.loads(payload)
            except Exception:
                data = {"_raw": payload}
            if isinstance(data, dict):
                yield "message", data


async def stop_session(
    client: httpx.AsyncClient,
    token: str,
    session_id: str,
    message_id: str,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> None:
    if not session_id or not message_id or not token:
        return
    try:
        await client.post(
            f"{base_url(options)}/chat_sessions/{session_id}/stop",
            headers=build_headers(token, options=options, stream=False),
            json={"chat_session_id": session_id, "user_message_id": message_id},
            timeout=10,
        )
    except Exception:
        return
