"""Release archives use tracked source, not the developer's runtime state."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_source import normalize_version


class ReleasePackagingTests(unittest.TestCase):
    def test_version_validation_accepts_release_and_prerelease_versions(self):
        for value, expected in (
            ("v1.0.2", "1.0.2"),
            ("1.2.3-rc.1", "1.2.3-rc.1"),
            ("0.0.0", "0.0.0"),
        ):
            self.assertEqual(normalize_version(value), expected)

    def test_version_validation_rejects_invalid_tags(self):
        for value in ("", "1", "01.0.2", "1.0.2/evil", "1.0.2-", "1.0.2-rc..1", "1.0.2-01"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_version(value)

    def test_packages_tracked_source_and_checksums_without_runtime_state(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "package_source.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print('relay')\n", encoding="utf-8")
            (root / ".env.example").write_text("PORT=8000\n", encoding="utf-8")
            (root / ".gitignore").write_text(".env\ndata/\n", encoding="utf-8")
            (root / ".env").write_text("PRIVATE_TOKEN=secret\n", encoding="utf-8")
            (root / "data").mkdir()
            (root / "data" / "accounts.json").write_text("secret", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Release Test", "-c",
                    "user.email=release@example.invalid", "commit", "-qm", "source",
                ],
                cwd=root,
                check=True,
            )
            output = root / "dist"
            subprocess.run(
                [
                    sys.executable, str(script), "--version", "v1.0.2",
                    "--output", str(output),
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )

            tarball = output / "trae-cn-relay-1.0.2.tar.gz"
            zipball = output / "trae-cn-relay-1.0.2.zip"
            with tarfile.open(tarball, "r:gz") as archive:
                tar_names = archive.getnames()
            with zipfile.ZipFile(zipball) as archive:
                zip_names = archive.namelist()
            for names in (tar_names, zip_names):
                self.assertIn("trae-cn-relay-1.0.2/src/main.py", names)
                self.assertIn("trae-cn-relay-1.0.2/.env.example", names)
                self.assertNotIn("trae-cn-relay-1.0.2/.env", names)
                self.assertFalse(any("/data/" in name for name in names))
                self.assertFalse(any("/.git/" in name for name in names))
            expected_checksums = "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                for path in (tarball, zipball)
            )
            self.assertEqual(
                (output / "trae-cn-relay-1.0.2.sha256").read_text(encoding="utf-8"),
                expected_checksums,
            )
