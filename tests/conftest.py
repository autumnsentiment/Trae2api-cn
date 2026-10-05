"""Process-wide test isolation.

The application intentionally persists credentials and account state under the
repository's ``.env`` and ``data/`` paths.  Test modules import ``src.main``
during collection, so a fixture that runs later is too late to prevent those
files from being read.  Keep all collection-time imports and app startup
inside a disposable, credential-free store instead.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


_ORIGINAL_ENV = os.environ.copy()
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="trae-relay-pytest-"))
_TEST_DATA = _TEST_ROOT / "data"
_TEST_DATA.mkdir(parents=True, exist_ok=True)
_TEST_ENV = _TEST_ROOT / ".env"
_TEST_ACCOUNTS = _TEST_DATA / "accounts.json"
_TEST_USAGE = _TEST_DATA / "usage_records.json"
_TEST_USAGE_STATS = _TEST_DATA / "usage_stats.json"
_FAKE_CLI = Path(__file__).resolve().parent / "fake" / "fake_cli.cmd"
_FAKE_CLI_SOURCE = _FAKE_CLI.with_name("fake_cli.py")
_FAKE_CLI_ORIGINAL: tuple[bytes, int] | None = None

# The fixture is intentionally named ``.cmd`` because Windows tests launch it
# directly.  GitHub Actions runs on Linux, where that file is neither
# executable nor a valid interpreter script.  Keep the same path used by all
# test modules, but install a disposable POSIX wrapper during collection.
if os.name != "nt" and _FAKE_CLI.exists() and _FAKE_CLI_SOURCE.exists():
    _FAKE_CLI_ORIGINAL = (_FAKE_CLI.read_bytes(), stat.S_IMODE(_FAKE_CLI.stat().st_mode))
    _FAKE_CLI.write_text(
        "#!/bin/sh\n"
        f'exec "{sys.executable}" "{_FAKE_CLI_SOURCE}" "$@"\n',
        encoding="utf-8",
    )
    _FAKE_CLI.chmod(_FAKE_CLI.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

# Prevent src.main's import-time dotenv.load_dotenv() from reading the
# developer's working-tree .env.  The original callable is restored when the
# pytest process finishes.
import dotenv

_ORIGINAL_LOAD_DOTENV = dotenv.load_dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False

# These defaults are deliberately credential-free.  Individual tests can
# override them with patch.dict or direct assignments as before.
_FAKE_CLI = str(Path(__file__).resolve().parent / "fake" / "fake_cli.cmd")
os.environ.update(
    {
        "TRAE_AUTH_SOURCE": "cli",
        "UPSTREAM_MODE": "cli",
        "TRAE_CLI_COMMAND": _FAKE_CLI,
        "TRAE_CLI_WORKDIR": str(_TEST_ROOT / "workspace"),
        "TRAE_CLI_PROMPT_MODE": "stdin",
        "TRAE_CLI_OUTPUT_MODE": "json",
        "TRAE_CLI_DISABLE_TOOLS": "false",
        # Keep the default test app unprotected.  Suites that exercise API-key
        # enforcement patch ``main.API_KEYS`` directly.
        "RELAY_API_KEYS": "",
        "TRAE_USAGE_RECORDS_PATH": str(_TEST_USAGE),
        "TRAE_USAGE_STATS_PATH": str(_TEST_USAGE_STATS),
    }
)

# Importing src.auth is safe: it does not load the account store until
# init_auth/_bootstrap_account_store is called.  Redirect both persistence
# paths before test modules import src.main.
from src import auth as _auth

_ORIGINAL_ACCOUNTS_PATH = _auth.ACCOUNTS_PATH
_ORIGINAL_ENV_PATH = _auth.ENV_PATH
_auth.ACCOUNTS_PATH = _TEST_ACCOUNTS
_auth.ENV_PATH = _TEST_ENV
_auth._accounts.clear()
_auth._active_account = ""
_auth._poll_enabled = False
_auth._settings.clear()
_auth._polling_mode = "round-robin"
_auth._rotation_cursor = 0


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Restore process globals and remove disposable test state."""

    if _FAKE_CLI_ORIGINAL is not None:
        content, mode = _FAKE_CLI_ORIGINAL
        _FAKE_CLI.write_bytes(content)
        _FAKE_CLI.chmod(mode)
    _auth.ACCOUNTS_PATH = _ORIGINAL_ACCOUNTS_PATH
    _auth.ENV_PATH = _ORIGINAL_ENV_PATH
    dotenv.load_dotenv = _ORIGINAL_LOAD_DOTENV

    os.environ.clear()
    os.environ.update(_ORIGINAL_ENV)
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
