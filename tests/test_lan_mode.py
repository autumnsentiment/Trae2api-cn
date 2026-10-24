"""LAN mode: key minting/persistence, the launcher, and the startup guard."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import lan_env

ROOT = Path(__file__).resolve().parent.parent


class LanEnvKeyTests(unittest.TestCase):
    def test_generated_key_is_prefixed_and_not_a_placeholder(self):
        key = lan_env.generate_key()
        self.assertTrue(key.startswith(lan_env.KEY_PREFIX))
        self.assertFalse(lan_env.is_placeholder(key))
        self.assertNotEqual(key, lan_env.generate_key())

    def test_placeholder_detection_covers_shipped_default(self):
        self.assertTrue(lan_env.is_placeholder("your-master-key"))
        self.assertTrue(lan_env.is_placeholder("  "))
        self.assertFalse(lan_env.is_placeholder("trae-lan-abc"))

    def test_parse_keys_trims_and_drops_empties(self):
        self.assertEqual(lan_env.parse_keys(" a ,, b "), ["a", "b"])
        self.assertEqual(lan_env.parse_keys(""), [])

    def test_explicit_key_wins_and_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("RELAY_API_KEYS=existing-real-key\n", encoding="utf-8")
            key, source = lan_env.ensure_key(env_path, "operator-key")
            self.assertEqual((key, source), ("operator-key", "explicit"))
            # The explicit key is saved so a later restart without arguments
            # reuses it instead of minting a new one and breaking clients.
            content = env_path.read_text("utf-8")
            self.assertIn("RELAY_API_KEYS=operator-key", content)
            self.assertNotIn("existing-real-key", content)
            reused, reused_source = lan_env.ensure_key(env_path)
            self.assertEqual((reused, reused_source), ("operator-key", "existing"))

    def test_existing_real_key_is_reused_without_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            original = "RELAY_API_KEYS=already-good\nPORT=8000\n"
            env_path.write_text(original, encoding="utf-8")
            key, source = lan_env.ensure_key(env_path)
            self.assertEqual((key, source), ("already-good", "existing"))
            self.assertEqual(env_path.read_text("utf-8"), original)

    def test_placeholder_is_replaced_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "# relay\nRELAY_API_KEYS=your-master-key\nPORT=8000\n",
                encoding="utf-8",
            )
            key, source = lan_env.ensure_key(env_path)
            self.assertEqual(source, "generated")
            self.assertTrue(key.startswith(lan_env.KEY_PREFIX))
            content = env_path.read_text("utf-8")
            self.assertIn(f"RELAY_API_KEYS={key}", content)
            self.assertNotIn("your-master-key", content)
            # Unrelated lines survive the rewrite.
            self.assertIn("# relay", content)
            self.assertIn("PORT=8000", content)

    def test_generated_key_is_stable_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            first, first_source = lan_env.ensure_key(env_path)
            second, second_source = lan_env.ensure_key(env_path)
            self.assertEqual(first_source, "generated")
            self.assertEqual(second_source, "existing")
            self.assertEqual(first, second)

    def test_resolve_can_refuse_to_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            key, source = lan_env.resolve_key(env_path, generate=False)
            self.assertEqual((key, source), ("", "missing"))

    def test_upsert_preserves_comments_and_other_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("# header\nA=1\nRELAY_API_KEYS=old\nB=2\n", encoding="utf-8")
            lan_env.upsert_env_value(env_path, "RELAY_API_KEYS", "new")
            content = env_path.read_text("utf-8")
            self.assertIn("RELAY_API_KEYS=new", content)
            self.assertNotIn("RELAY_API_KEYS=old", content)
            self.assertIn("A=1", content)
            self.assertIn("B=2", content)
            self.assertIn("# header", content)

    def test_cli_resolve_json_reports_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "lan_env.py"),
                    "resolve",
                    "--env",
                    str(env_path),
                    "--save",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["source"], "generated")
            self.assertTrue(payload["key"].startswith(lan_env.KEY_PREFIX))
            self.assertIn(f"RELAY_API_KEYS={payload['key']}", env_path.read_text("utf-8"))


class LanEnvParsingTests(unittest.TestCase):
    """``.env`` values must be read exactly the way ``python-dotenv`` reads them.

    The launcher exports the parsed value as an environment variable, which
    overrides ``.env`` inside the relay.  Any difference in quoting rules makes
    the relay accept a key the launcher never printed, or vice versa.
    """

    def _dotenv_value(self, text: str) -> str:
        from dotenv import dotenv_values

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(text + "\n", encoding="utf-8")
            return dotenv_values(env_path).get("RELAY_API_KEYS", "")

    def test_quoting_forms_match_dotenv(self):
        for text in (
            'RELAY_API_KEYS="quoted-key"',
            "RELAY_API_KEYS='single-key'",
            "export RELAY_API_KEYS=exported-key",
            "RELAY_API_KEYS=plain # comment",
            'RELAY_API_KEYS="a b"  # note',
            "RELAY_API_KEYS=abc#notcomment",
        ):
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory() as tmp:
                    env_path = Path(tmp) / ".env"
                    env_path.write_text(text + "\n", encoding="utf-8")
                    key, source = lan_env.ensure_key(env_path)
                self.assertEqual(source, "existing")
                self.assertEqual(key, self._dotenv_value(text))

    def test_multiple_real_keys_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("RELAY_API_KEYS=key-one,key-two\n", encoding="utf-8")
            key, source = lan_env.ensure_key(env_path)
            self.assertEqual((key, source), ("key-one,key-two", "existing"))
            self.assertEqual(
                env_path.read_text("utf-8"), "RELAY_API_KEYS=key-one,key-two\n"
            )

    def test_placeholder_is_dropped_from_a_mixed_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "RELAY_API_KEYS=your-master-key,real-key\n", encoding="utf-8"
            )
            key, source = lan_env.ensure_key(env_path)
            self.assertEqual((key, source), ("real-key", "existing"))
            # The guessable value must stop being accepted, not merely ignored.
            self.assertNotIn("your-master-key", env_path.read_text("utf-8"))

    def test_export_prefix_survives_a_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "export RELAY_API_KEYS=your-master-key\n", encoding="utf-8"
            )
            key, source = lan_env.ensure_key(env_path)
            self.assertEqual(source, "generated")
            content = env_path.read_text("utf-8")
            self.assertIn(f"export RELAY_API_KEYS={key}", content)
            self.assertNotIn("your-master-key", content)

    def test_unterminated_quote_is_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text('RELAY_API_KEYS="unterminated\n', encoding="utf-8")
            key, source = lan_env.ensure_key(env_path)
            self.assertEqual(source, "generated")
            self.assertTrue(key.startswith(lan_env.KEY_PREFIX))

    def test_get_subcommand_prints_parsed_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text('PORT="9123"\n', encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "lan_env.py"),
                    "get",
                    "--env",
                    str(env_path),
                    "--key",
                    "PORT",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(proc.stdout.strip(), "9123")


class LanLauncherTests(unittest.TestCase):
    """Exercise ``start_lan.sh`` without installing the relay's dependencies."""

    def _run(self, args, env=None):
        return subprocess.run(
            ["bash", str(ROOT / "start_lan.sh"), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def _make_project(self, tmp: str) -> Path:
        """Copy the launcher and its helper into a scratch project root.

        The relay's dependencies are not needed for a dry-run, so only the
        files ``start_lan.sh`` actually touches are copied.  The scratch ``.env``
        stays absent unless a test asks for it, which is how "a dry-run must not
        persist a key" is observed.
        """

        project = Path(tmp) / "proj"
        (project / "src").mkdir(parents=True)
        (project / "src" / "main.py").write_text("", encoding="utf-8")
        (project / "scripts").mkdir()
        (project / "scripts" / "lan_env.py").write_text(
            (ROOT / "scripts" / "lan_env.py").read_text("utf-8"),
            encoding="utf-8",
        )
        (project / "start_lan.sh").write_text(
            (ROOT / "start_lan.sh").read_text("utf-8"), encoding="utf-8"
        )
        return project

    def _run_in(self, project: Path, args):
        return subprocess.run(
            ["bash", str(project / "start_lan.sh"), *args],
            cwd=str(project),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_dry_run_reports_host_port_and_command(self):
        proc = self._run(["--dry-run"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("局域网 (LAN) 模式", proc.stdout)
        self.assertIn("HOST=0.0.0.0", proc.stdout)
        self.assertIn("--host 0.0.0.0", proc.stdout)
        self.assertIn("uvicorn src.main:app", proc.stdout)

    def test_dry_run_does_not_write_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._make_project(tmp)
            proc = self._run_in(project, ["--dry-run"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse((project / ".env").exists())

    def test_dry_run_uses_explicit_key(self):
        proc = self._run(["9001", "MyLanKey", "--dry-run"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("MyLanKey", proc.stdout)
        self.assertIn("命令行指定", proc.stdout)
        self.assertIn("--port 9001", proc.stdout)

    def test_invalid_port_is_rejected(self):
        proc = self._run(["notaport", "--dry-run"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("无效端口", proc.stdout)

    def test_extra_positional_argument_is_rejected(self):
        proc = self._run(["8000", "key", "extra", "--dry-run"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("多余的参数", proc.stdout)

    def test_saved_key_is_reused_on_second_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._make_project(tmp)
            first = self._run_in(project, ["--dry-run"])
            # A dry-run must not persist; emulate a real launch by resolving
            # with --save, then confirm the launcher reuses the saved value.
            subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts" / "lan_env.py"),
                    "resolve",
                    "--env",
                    str(project / ".env"),
                    "--save",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            saved = lan_env.read_env_value(project / ".env", "RELAY_API_KEYS")
            second = self._run_in(project, ["--dry-run"])
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn(saved, second.stdout)
            self.assertIn("沿用 .env", second.stdout)

    def test_port_falls_back_to_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._make_project(tmp)
            (project / ".env").write_text(
                "PORT=9123\nRELAY_API_KEYS=env-key\n", encoding="utf-8"
            )
            proc = self._run_in(project, ["--dry-run"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("--port 9123", proc.stdout)
            self.assertIn("9123/tcp", proc.stdout)

    def test_command_line_port_overrides_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._make_project(tmp)
            (project / ".env").write_text(
                "PORT=9123\nRELAY_API_KEYS=env-key\n", encoding="utf-8"
            )
            proc = self._run_in(project, ["7777", "--dry-run"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("--port 7777", proc.stdout)
            self.assertNotIn("--port 9123", proc.stdout)

    def test_invalid_env_port_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._make_project(tmp)
            (project / ".env").write_text(
                "PORT=notaport\nRELAY_API_KEYS=env-key\n", encoding="utf-8"
            )
            proc = self._run_in(project, ["--dry-run"])
            self.assertEqual(proc.returncode, 1)
            self.assertIn(".env 中的 PORT", proc.stdout)

    def test_dry_run_skips_port_occupancy_check(self):
        """A dry-run must not fail just because the port happens to be busy."""

        with socket.socket() as sock:
            sock.bind(("0.0.0.0", 0))
            sock.listen(1)
            busy_port = str(sock.getsockname()[1])
            proc = self._run([busy_port, "--dry-run"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("已被占用", proc.stdout)

    def test_occupied_port_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self._make_project(tmp)
            (project / ".venv").symlink_to(ROOT / ".venv")
            (project / ".env").write_text(
                "RELAY_API_KEYS=env-key\n", encoding="utf-8"
            )
            with socket.socket() as sock:
                sock.bind(("0.0.0.0", 0))
                sock.listen(1)
                busy_port = str(sock.getsockname()[1])
                proc = self._run_in(project, [busy_port])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("已被占用", proc.stdout)
            self.assertIn(busy_port, proc.stdout)


class LanStartupGuardTests(unittest.TestCase):
    """The startup warning must fire only for reachable hosts and weak keys."""

    def _warn_with(self, host, keys):
        """Call the guard under a patched host/key pair and return its warnings."""

        import logging

        from src import main as main_module

        logger = logging.getLogger("trae-cn-relay")
        with (
            patch.object(main_module, "HOST", host),
            patch.object(main_module, "API_KEYS", keys),
            patch.object(logger, "warning") as warn,
        ):
            main_module._warn_insecure_lan_key()
        messages = []
        for call in warn.call_args_list:
            template, *rest = call.args
            messages.append(str(template) % tuple(rest) if rest else str(template))
        return messages

    def test_loopback_host_never_warns(self):
        self.assertEqual(self._warn_with("127.0.0.1", []), [])
        self.assertEqual(self._warn_with("127.0.0.1", ["your-master-key"]), [])

    def test_reachable_host_without_keys_warns(self):
        warnings = self._warn_with("0.0.0.0", [])
        self.assertTrue(
            any("LAN mode without RELAY_API_KEYS" in line for line in warnings), warnings
        )

    def test_reachable_host_with_placeholder_warns(self):
        warnings = self._warn_with("0.0.0.0", ["your-master-key"])
        self.assertTrue(
            any("placeholder API key" in line for line in warnings), warnings
        )

    def test_reachable_host_with_real_key_is_quiet(self):
        self.assertEqual(self._warn_with("0.0.0.0", ["trae-lan-realrandomvalue"]), [])

    def test_lan_reachable_classification(self):
        from src import main as main_module

        for host, expected in (
            ("0.0.0.0", True),
            ("::", True),
            ("*", True),
            ("127.0.0.1", False),
            ("::1", False),
            ("192.168.1.10", True),
            ("", False),
        ):
            with self.subTest(host=host), patch.object(main_module, "HOST", host):
                self.assertEqual(main_module._is_lan_reachable(), expected)


if __name__ == "__main__":
    unittest.main()
