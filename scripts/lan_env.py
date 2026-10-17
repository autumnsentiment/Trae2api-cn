#!/usr/bin/env python3
"""LAN-mode helpers for ``start_lan.sh``.

The relay already listens on every interface, but a LAN deployment is only
"working" if it is also protected: the gateway spends the signed-in account's
own upstream quota, so a blank or guessable ``RELAY_API_KEYS`` lets any device
on the network drain it.  ``.env.example`` ships ``your-master-key`` as a
placeholder, which is exactly the kind of value that must never protect a
reachable port.

This module resolves the key a LAN launch should use, mints a random one when
none is configured, and persists it so clients keep working across restarts.
It is deliberately dependency-free (standard library only) so the launcher can
call it before the relay's own requirements are installed.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

# Values that must not be treated as a real LAN credential.  ``.env.example``
# ships the first one, so a plain ``cp .env.example .env`` would otherwise
# publish a trivially guessable key on every interface.
PLACEHOLDER_KEYS = {
    "",
    "your-master-key",
    "your-key",
    "changeme",
    "change-me",
    "master-key",
    "key",
    "key1",
    "test",
    "test-key",
    "sk-xxx",
}

ENV_KEY = "RELAY_API_KEYS"
KEY_PREFIX = "trae-lan-"


def generate_key() -> str:
    """Return a fresh URL-safe LAN key with a recognizable prefix."""

    return KEY_PREFIX + secrets.token_urlsafe(24)


def parse_keys(value: str) -> list[str]:
    """Split a comma-separated ``RELAY_API_KEYS`` value into usable keys."""

    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def is_placeholder(value: str) -> bool:
    """True when a configured key is empty or a known shipped default."""

    return str(value or "").strip().lower() in PLACEHOLDER_KEYS


def _unquote(value: str) -> str | None:
    """Apply the same quoting rules ``python-dotenv`` uses for one value.

    Returns ``None`` for an unterminated quote, which ``python-dotenv`` also
    refuses to parse.  Treating such a line as absent keeps the launcher from
    exporting a value the relay would never load.
    """

    value = value.strip()
    if value[:1] in ('"', "'"):
        quote = value[0]
        index = 1
        escaped = False
        while index < len(value):
            char = value[index]
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                break
            index += 1
        if index >= len(value):
            return None
        inner = value[1:index]
        if quote == "'":
            return inner
        return (
            inner.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )
    # Unquoted values end at the first `` #`` comment marker, matching dotenv.
    for index, char in enumerate(value):
        if char == "#" and index > 0 and value[index - 1] in " \t":
            return value[:index].rstrip()
    return value


def _env_line_name(line: str) -> str:
    """Return the variable name a ``.env`` line assigns, or an empty string."""

    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return ""
    name = stripped.split("=", 1)[0].strip()
    if name.startswith("export "):
        name = name[len("export ") :].strip()
    return name


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Return ``(name, value)`` for a ``.env`` assignment, else ``None``."""

    name = _env_line_name(line)
    if not name:
        return None
    value = _unquote(line.strip().split("=", 1)[1])
    if value is None:
        return None
    return name, value


def read_env_value(path: Path, key: str) -> str:
    """Read one ``KEY=value`` line without executing the rest of the file.

    Supports the forms ``python-dotenv`` accepts: an optional ``export``
    prefix, single or double quotes, and trailing ``# comments``.  The value
    must be read the same way the relay will read it, otherwise the launcher
    exports a string the server never accepts as its API key.
    """

    try:
        content = path.read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    for line in content.splitlines():
        parsed = _parse_env_line(line)
        if parsed is not None and parsed[0] == key:
            return parsed[1]
    return ""


def _format_env_value(value: str) -> str:
    """Quote a value only when writing it unquoted would change its meaning."""

    if value and not any(char in value for char in ' \t#"\'\\'):
        return value
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def upsert_env_value(path: Path, key: str, value: str) -> None:
    """Set ``KEY=value`` in ``.env`` while leaving every other line intact.

    An existing ``export`` prefix is preserved so a file that is also sourced
    by a shell keeps working.
    """

    try:
        lines = path.read_text("utf-8").splitlines() if path.exists() else []
    except (OSError, UnicodeDecodeError):
        lines = []
    out: list[str] = []
    seen = False
    for line in lines:
        if _env_line_name(line) == key:
            if not seen:
                seen = True
                prefix = "export " if line.strip().startswith("export ") else ""
                out.append(f"{prefix}{key}={_format_env_value(value)}")
            continue
        out.append(line)
    if not seen:
        out.append(f"{key}={_format_env_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", "utf-8")


def resolve_key(
    env_path: Path,
    explicit: str = "",
    generate: bool = True,
) -> tuple[str, str]:
    """Return ``(key, source)`` for a LAN launch.

    ``source`` is one of ``explicit`` (caller supplied it), ``existing`` (a real
    key already lived in ``.env``) or ``generated`` (a placeholder was replaced).
    An explicit key is returned untouched and never treated as a placeholder,
    because the operator asked for it on purpose.
    """

    if str(explicit or "").strip():
        return str(explicit).strip(), "explicit"

    configured = parse_keys(read_env_value(env_path, ENV_KEY))
    real = [item for item in configured if not is_placeholder(item)]
    if real:
        # Keep the whole list: the relay accepts any of these keys, so
        # exporting only the first would break clients using the others.
        return ",".join(real), "existing"
    if not generate:
        return "", "missing"
    return generate_key(), "generated"


def ensure_key(env_path: Path, explicit: str = "") -> tuple[str, str]:
    """Resolve the LAN key and persist it when it is new or needs cleanup.

    An explicit key is persisted too: a later restart without arguments would
    otherwise mint a fresh random key and silently break every client that was
    configured with the operator-supplied one.  An already-valid ``.env`` is
    left byte-for-byte alone, including its other keys, comments and quoting.
    """

    key, source = resolve_key(env_path, explicit)
    if not key:
        return key, source
    if source == "existing":
        configured = parse_keys(read_env_value(env_path, ENV_KEY))
        # A file that still lists a placeholder must be rewritten so the
        # guessable value stops being an accepted key, not merely ignored here.
        if not any(is_placeholder(item) for item in configured):
            return key, source
    upsert_env_value(env_path, ENV_KEY, key)
    return key, source


def _cmd_get(args: argparse.Namespace) -> int:
    """Print one ``.env`` value using the same parsing as the relay."""

    print(read_env_value(Path(args.env).expanduser(), args.key))
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    env_path = Path(args.env).expanduser()
    if args.save:
        key, source = ensure_key(env_path, args.key or "")
    else:
        key, source = resolve_key(env_path, args.key or "", generate=not args.no_generate)
    if args.json:
        print(json.dumps({"key": key, "source": source, "env": str(env_path)}))
    else:
        print(key)
    return 0 if key else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve the relay LAN API key")
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve", help="resolve (and optionally persist) the LAN key")
    resolve.add_argument("--env", default=".env", help="path to the relay .env file")
    resolve.add_argument("--key", default="", help="operator-supplied key (wins over .env)")
    resolve.add_argument("--save", action="store_true", help="write the resolved key into .env")
    resolve.add_argument(
        "--no-generate",
        action="store_true",
        help="do not mint a key when none is configured",
    )
    resolve.add_argument("--json", action="store_true", help="emit JSON instead of a bare key")
    resolve.set_defaults(func=_cmd_resolve)
    get = sub.add_parser("get", help="print one value from the relay .env")
    get.add_argument("--env", default=".env", help="path to the relay .env file")
    get.add_argument("--key", required=True, help="variable name to read")
    get.set_defaults(func=_cmd_get)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
