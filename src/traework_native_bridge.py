"""Optional Windows bridge for TraeWork's native ai-agent transport.

The production relay runs in a Linux container and must never try to load a
Windows PE DLL. This module treats the native runtime as a Windows helper
process. The helper owns DLL loading and AHA IPC; the relay sends the
request envelope that TraeWork's renderer sends and consumes an HTTP/SSE
response from the helper.

The helper URL is configurable. A real AHA framing/endpoint can be substituted
without changing the OpenAI route or translation code. No token or packet
content is written to logs by this module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import struct
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import httpx

from . import auth
from .cli_client import sanitize_assistant_history_messages


logger = logging.getLogger(__name__)

DEFAULT_BRIDGE_URL = "http://127.0.0.1:40006"
DEFAULT_BRIDGE_ENDPOINT = "/v1/traework/request_stream"
DEFAULT_HEALTH_ENDPOINT = "/healthz"
DEFAULT_MAX_FRAME_BYTES = 64 * 1024 * 1024


class NativeBridgeError(RuntimeError):
    """Base error raised by the optional native transport."""


class NativeBridgeUnavailable(NativeBridgeError):
    """The Windows helper or configured endpoint is unavailable."""


class NativeBridgeProtocolError(NativeBridgeError):
    """A helper returned an invalid HTTP or AHA packet."""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _split_command(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    try:
        parts = shlex.split(value, posix=sys.platform != "win32")
    except ValueError:
        parts = value.split()
    return tuple(item for item in parts if item)


@dataclass(frozen=True)
class NativeBridgeConfig:
    """Configuration for a local TraeWork native helper."""

    enabled: bool = False
    bridge_url: str = DEFAULT_BRIDGE_URL
    endpoint: str = DEFAULT_BRIDGE_ENDPOINT
    health_endpoint: str = DEFAULT_HEALTH_ENDPOINT
    timeout_seconds: float = 300.0
    connect_timeout_seconds: float = 5.0
    install_dir: str = ""
    helper_command: tuple[str, ...] = ()
    auto_start: bool = False
    allow_non_windows: bool = False
    bridge_token: str = field(default="", repr=False)
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES

    @classmethod
    def from_env(cls) -> "NativeBridgeConfig":
        install_dir = os.environ.get("TRAE_WORK_INSTALL_DIR", "").strip()
        helper_command = _split_command(
            os.environ.get("TRAEWORK_NATIVE_HELPER_COMMAND", "")
        )
        try:
            max_frame_bytes = int(
                os.environ.get(
                    "TRAEWORK_NATIVE_MAX_FRAME_BYTES",
                    str(DEFAULT_MAX_FRAME_BYTES),
                )
            )
        except (TypeError, ValueError):
            max_frame_bytes = DEFAULT_MAX_FRAME_BYTES
        return cls(
            enabled=_env_bool("TRAEWORK_NATIVE_ENABLED", False),
            bridge_url=(
                os.environ.get("TRAEWORK_NATIVE_BRIDGE_URL", DEFAULT_BRIDGE_URL).strip()
                or DEFAULT_BRIDGE_URL
            ).rstrip("/"),
            endpoint=(
                os.environ.get(
                    "TRAEWORK_NATIVE_BRIDGE_ENDPOINT", DEFAULT_BRIDGE_ENDPOINT
                ).strip()
                or DEFAULT_BRIDGE_ENDPOINT
            ),
            health_endpoint=(
                os.environ.get(
                    "TRAEWORK_NATIVE_HEALTH_ENDPOINT", DEFAULT_HEALTH_ENDPOINT
                ).strip()
                or DEFAULT_HEALTH_ENDPOINT
            ),
            timeout_seconds=_env_float("TRAEWORK_NATIVE_TIMEOUT_SECONDS", 300.0),
            connect_timeout_seconds=_env_float(
                "TRAEWORK_NATIVE_CONNECT_TIMEOUT_SECONDS", 5.0
            ),
            install_dir=install_dir,
            helper_command=helper_command,
            auto_start=_env_bool("TRAEWORK_NATIVE_AUTO_START", False),
            # Tests and an explicitly configured Wine host may opt in. The
            # normal relay remains Windows-only by default.
            allow_non_windows=_env_bool("TRAEWORK_NATIVE_ALLOW_NON_WINDOWS", False),
            bridge_token=os.environ.get("TRAEWORK_NATIVE_BRIDGE_TOKEN", "").strip(),
            max_frame_bytes=max(1024, max_frame_bytes),
        )

    @property
    def enabled_for_platform(self) -> bool:
        if self.allow_non_windows or sys.platform == "win32":
            return True
        # A Linux/Docker relay may proxy to a Windows helper over HTTP. This
        # does not load PE files in Linux; require an explicit non-default URL
        # so the normal raw/remote deployment remains unchanged.
        configured_url = os.environ.get("TRAEWORK_NATIVE_BRIDGE_URL", "").strip()
        return bool(
            (configured_url and configured_url.rstrip("/") != DEFAULT_BRIDGE_URL)
            or self.bridge_url != DEFAULT_BRIDGE_URL
        )

    def endpoint_url(self) -> str:
        return f"{self.bridge_url}/{self.endpoint.lstrip('/')}"

    def health_url(self) -> str:
        return f"{self.bridge_url}/{self.health_endpoint.lstrip('/')}"


def discover_install_dir(config: Optional[NativeBridgeConfig] = None) -> Optional[Path]:
    """Find a TraeWork install without probing or logging credentials."""

    config = config or NativeBridgeConfig.from_env()
    candidates: list[Path] = []
    if config.install_dir:
        candidates.append(Path(config.install_dir).expanduser())
    for env_name in ("TRAE_WORK_INSTALL_DIR", "TRAE_SOLO_INSTALL_DIR"):
        value = os.environ.get(env_name, "").strip()
        if value:
            candidates.append(Path(value).expanduser())
    # Do not scan the entire filesystem. These are conservative defaults;
    # users can point TRAE_WORK_INSTALL_DIR at another installation.
    for root_name in ("ProgramFiles", "LOCALAPPDATA"):
        root = os.environ.get(root_name, "").strip()
        if root:
            candidates.extend(
                [
                    Path(root) / "Trae",
                    Path(root) / "Trae SOLO CN",
                    Path(root) / "Programs" / "Trae",
                    Path(root) / "Programs" / "Trae SOLO CN",
                ]
            )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir():
            return candidate
    return None


def inspect_installation(path: Optional[Path]) -> dict[str, Any]:
    """Return non-sensitive native-runtime status for diagnostics/tests."""

    if path is None:
        return {"available": False, "reason": "install_dir_not_found"}
    agent = path / "resources" / "app" / "modules" / "ai-agent"
    if (path / "modules" / "ai-agent").is_dir():
        agent = path / "modules" / "ai-agent"
    elif path.name.lower() == "ai-agent":
        agent = path
    required = {
        "ai_agent_dll": agent / "ai_agent.dll",
        "sscronet_dll": agent / "sscronet.dll",
        "meta_json": agent / "meta.json",
        "start_bat": agent / "start.bat",
    }
    present = {name: item.is_file() for name, item in required.items()}
    return {
        "available": all(present.values()),
        "agent_dir": str(agent),
        "files": present,
        "missing": [name for name, found in present.items() if not found],
    }


def build_native_payload(
    messages: list[dict[str, Any]],
    model: str,
    *,
    stream: bool = True,
    options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the renderer/Ode payload consumed by TraeWork's AHA layer."""

    options = options or {}
    session_id = str(
        options.get("connect_session_id")
        or options.get("connectSessionId")
        or options.get("native_session_id")
        or options.get("session_id")
        or options.get("sessionId")
        or uuid.uuid4()
    )
    model_name = str(
        options.get("trae_native_model_name")
        or options.get("modelName")
        or options.get("model_name")
        or model
        or "auto"
    )

    data = dict(options.get("native_data") or {})
    native_messages = data.get("messages")
    if not isinstance(native_messages, list):
        native_messages = messages
    data["messages"] = sanitize_assistant_history_messages(native_messages)
    data.setdefault("model_name", model_name)
    data.setdefault("stream", bool(stream))
    for key in ("custom_model", "model_auto_selection"):
        value = options.get(key)
        if isinstance(value, Mapping):
            data.setdefault(key, dict(value))
    # Internal relay bookkeeping keys are intentionally never copied.
    for key in (
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "reasoning_effort",
        "response_format",
        "user",
    ):
        if key in options and options[key] is not None:
            data[key] = options[key]

    user_info = dict(options.get("native_user_info") or {})
    token = str(options.get("_auth_token") or options.get("auth_token") or "")
    if not token:
        token = auth.get_token()
    user_info.setdefault("token", token)
    user_info.setdefault(
        "user_id",
        str(
            options.get("_auth_user_id")
            or options.get("_billing_id")
            or auth.get_user_id()
            or ""
        ),
    )
    psd = dict(auth.get_psd())
    provider_specific = options.get("provider_specific") or options.get(
        "providerSpecificData"
    )
    if isinstance(provider_specific, Mapping):
        psd.update(provider_specific)
    for target in ("scope", "tenant_id", "loginScope", "email", "name"):
        if target not in user_info and psd.get(target):
            user_info[target] = psd[target]

    common_params = dict(options.get("native_common_params") or {})
    common_params.setdefault("ai_model_name", model_name)
    common_params.setdefault(
        "agent_type", str(options.get("agent_type") or "solo_lite")
    )
    common_params.setdefault(
        "shell_execute_strategy",
        str(options.get("shell_execute_strategy") or "ask"),
    )
    for key in (
        "request_traffic_type",
        "ab_force_vids",
        "ab_autotest_advanced_mode",
    ):
        if options.get(key) is not None:
            common_params.setdefault(key, options[key])
    streamlined = dict(options.get("native_streamlined_common_params") or {})
    streamlined.setdefault("ai_model_name", model_name)

    context = options.get("client_context") or options.get("clientContext")
    context = dict(context) if isinstance(context, Mapping) else {}
    client_info = dict(options.get("native_client_info") or {})
    client_info.setdefault("connect_session_id", session_id)
    client_info.setdefault(
        "client_type", str(options.get("client_type") or "solo_lite")
    )
    client_info.setdefault("is_solo_mode", True)
    client_info.setdefault(
        "workspace_folder",
        str(
            options.get("workspace_folder")
            or options.get("workspacePath")
            or context.get("workspace_path")
            or context.get("workspacePath")
            or ""
        ),
    )
    workspace_folder = str(client_info.get("workspace_folder") or "")
    client_info.setdefault(
        "workspace_folders", [workspace_folder] if workspace_folder else []
    )
    client_info.setdefault("original_workspace_folder", workspace_folder)
    client_info.setdefault(
        "terminal_info",
        context.get("terminal_context") or context.get("terminalContext") or [],
    )
    client_info.setdefault(
        "device_id",
        str(options.get("device_id") or options.get("deviceId") or ""),
    )
    client_info.setdefault(
        "workspace_id",
        str(options.get("workspace_id") or options.get("workspaceId") or ""),
    )
    client_info.setdefault(
        "version_code",
        str(options.get("version_code") or options.get("versionCode") or ""),
    )
    client_info.setdefault("icube_language", str(psd.get("appLanguage") or "zh-CN"))
    client_info.setdefault("user_timezone", str(options.get("user_timezone") or "Asia/Shanghai"))

    return {
        "service": "chat",
        "method": "chat",
        "data": data,
        "user_info": user_info,
        "common_params": common_params,
        "streamlined_common_params": streamlined,
        "client_info": client_info,
    }


def build_aha_packet(
    messages: list[dict[str, Any]],
    model: str,
    *,
    stream: bool = True,
    options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the renderer request envelope used by request/request_stream RPC."""

    options = options or {}
    params = build_native_payload(messages, model, stream=stream, options=options)
    session_id = str(
        params["client_info"].get("connect_session_id") or uuid.uuid4()
    )
    channel_id = str(
        options.get("native_channel_id")
        or options.get("channel_id")
        or uuid.uuid4()
    )
    # TransportManager.e() in TraeWork always wraps the payload in a
    # packet_type="request" envelope. The RPC method (request vs
    # request_stream) selects streaming; packet_type is not the RPC method.
    return {
        "packet_type": "request",
        "channel_id": channel_id,
        "session_id": session_id,
        "params": params,
    }


def encode_aha_frame(packet: Mapping[str, Any], *, byteorder: str = ">") -> bytes:
    """Encode one AHA frame using a four-byte JSON length prefix."""

    if byteorder not in (">", "<"):
        raise ValueError("byteorder must be '>' or '<'")
    payload = json.dumps(
        packet, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return struct.pack(f"{byteorder}I", len(payload)) + payload


class AhaFrameDecoder:
    """Incremental decoder for length-prefixed JSON AHA frames."""

    def __init__(
        self,
        *,
        byteorder: str = ">",
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ):
        if byteorder not in (">", "<"):
            raise ValueError("byteorder must be '>' or '<'")
        self.byteorder = byteorder
        self.max_frame_bytes = max(1024, int(max_frame_bytes))
        self._buffer = bytearray()

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[dict[str, Any]]:
        self._buffer.extend(bytes(chunk))
        frames: list[dict[str, Any]] = []
        while len(self._buffer) >= 4:
            length = struct.unpack(f"{self.byteorder}I", self._buffer[:4])[0]
            if length > self.max_frame_bytes:
                raise NativeBridgeProtocolError(
                    f"AHA frame exceeds configured limit ({length} bytes)"
                )
            if len(self._buffer) < 4 + length:
                break
            payload = bytes(self._buffer[4 : 4 + length])
            del self._buffer[: 4 + length]
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NativeBridgeProtocolError(
                    "AHA frame is not valid UTF-8 JSON"
                ) from exc
            if not isinstance(decoded, dict):
                raise NativeBridgeProtocolError(
                    "AHA frame must contain a JSON object"
                )
            frames.append(decoded)
        return frames

    def finish(self) -> None:
        if self._buffer:
            raise NativeBridgeProtocolError("incomplete AHA frame at end of stream")


@dataclass
class NativeChatResponse:
    """Streaming helper response and its owning synchronous HTTP client."""

    response: httpx.Response
    client: httpx.Client
    auth_token: str = ""

    def close(self) -> None:
        self.response.close()
        self.client.close()

    def __enter__(self) -> "NativeChatResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class TraeWorkNativeBridge:
    """HTTP adapter to a Windows helper that hosts TraeWork's DLLs."""

    def __init__(
        self,
        config: Optional[NativeBridgeConfig] = None,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ):
        self.config = config or NativeBridgeConfig.from_env()
        self._client_factory = client_factory
        self._helper_process: Optional[subprocess.Popen] = None

    def _ensure_platform(self) -> None:
        if not self.config.enabled_for_platform:
            raise NativeBridgeUnavailable(
                "TraeWork native transport requires a Windows helper; "
                "set TRAEWORK_NATIVE_BRIDGE_URL to a running helper or use raw/remote on Linux"
            )

    def _ensure_enabled(self) -> None:
        self._ensure_platform()
        if not self.config.enabled:
            raise NativeBridgeUnavailable(
                "TraeWork native transport is disabled; set TRAEWORK_NATIVE_ENABLED=true"
            )

    def _maybe_start_helper(self) -> None:
        if not self.config.auto_start or not self.config.helper_command:
            return
        if self._helper_process is not None and self._helper_process.poll() is None:
            return
        if not self.config.enabled_for_platform:
            return
        env = os.environ.copy()
        if self.config.install_dir:
            env.setdefault("TRAE_WORK_INSTALL_DIR", self.config.install_dir)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._helper_process = subprocess.Popen(
                list(self.config.helper_command),
                cwd=self.config.install_dir or None,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise NativeBridgeUnavailable(
                "could not start TraeWork native helper"
            ) from exc

    async def health(self) -> bool:
        self._ensure_enabled()
        self._maybe_start_helper()

        def check() -> bool:
            try:
                with self._client_factory(
                    timeout=self.config.connect_timeout_seconds
                ) as client:
                    response = client.get(self.config.health_url())
                    return 200 <= response.status_code < 300
            except (httpx.HTTPError, OSError):
                return False

        return await asyncio.to_thread(check)

    async def send_chat_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        stream: bool = True,
        options: Optional[Mapping[str, Any]] = None,
    ) -> NativeChatResponse:
        """Send one full TraeWork packet to the configured Windows helper."""

        self._ensure_enabled()
        self._maybe_start_helper()
        options = dict(options or {})
        packet = build_aha_packet(messages, model, stream=stream, options=options)
        token = str(
            options.get("_auth_token")
            or options.get("auth_token")
            or auth.get_token()
            or ""
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "X-Trae-Native-Bridge": "1",
            "X-Trae-Channel-Id": str(packet["channel_id"]),
        }
        if token:
            headers["Authorization"] = f"Cloud-IDE-JWT {token}"
        if self.config.bridge_token:
            headers["X-Trae-Native-Bridge-Token"] = self.config.bridge_token

        def open_stream() -> NativeChatResponse:
            client = self._client_factory(timeout=self.config.timeout_seconds)
            response: Optional[httpx.Response] = None
            try:
                request = client.build_request(
                    "POST",
                    self.config.endpoint_url(),
                    headers=headers,
                    json=packet,
                )
                response = client.send(request, stream=stream)
                if response.status_code < 200 or response.status_code >= 300:
                    body = response.read().decode("utf-8", errors="replace")[:500]
                    raise NativeBridgeProtocolError(
                        "TraeWork native helper returned HTTP "
                        f"{response.status_code}: {body}"
                    )
                return NativeChatResponse(response, client, token)
            except Exception:
                if response is not None:
                    response.close()
                client.close()
                raise

        try:
            return await asyncio.to_thread(open_stream)
        except NativeBridgeProtocolError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise NativeBridgeUnavailable(
                "TraeWork native helper request failed"
            ) from exc


async def send_native_chat_request(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool = True,
    options: Optional[Mapping[str, Any]] = None,
) -> NativeChatResponse:
    """Module-level compatibility wrapper used by main.py."""

    # main.py loads dotenv after importing modules, so read environment values
    # at request time instead of freezing them in a module-level singleton.
    return await TraeWorkNativeBridge().send_chat_request(
        messages, model, stream=stream, options=options
    )


__all__ = [
    "AhaFrameDecoder",
    "DEFAULT_BRIDGE_ENDPOINT",
    "DEFAULT_BRIDGE_URL",
    "NativeBridgeConfig",
    "NativeBridgeError",
    "NativeBridgeProtocolError",
    "NativeBridgeUnavailable",
    "NativeChatResponse",
    "TraeWorkNativeBridge",
    "build_aha_packet",
    "build_native_payload",
    "discover_install_dir",
    "encode_aha_frame",
    "inspect_installation",
    "send_native_chat_request",
]
