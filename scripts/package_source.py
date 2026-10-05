#!/usr/bin/env python3
"""Create reproducible source archives for a tagged release.

Only files tracked by git are included.  This keeps local credentials, account
state, Docker backups, probes, and other ignored artifacts out of release
archives without maintaining a second hand-written file allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path


_CORE_VERSION_PATTERN = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_PRERELEASE_IDENTIFIER_PATTERN = re.compile(r"[0-9A-Za-z-]+$")


def normalize_version(value: str) -> str:
    version = value.strip()
    if version.startswith("v"):
        version = version[1:]
    core, separator, prerelease = version.partition("-")
    if not _CORE_VERSION_PATTERN.fullmatch(core) or not separator or not prerelease:
        if not _CORE_VERSION_PATTERN.fullmatch(version):
            raise ValueError("version must use MAJOR.MINOR.PATCH with an optional prerelease suffix")
        return version
    identifiers = prerelease.split(".")
    if any(
        not _PRERELEASE_IDENTIFIER_PATTERN.fullmatch(identifier)
        or (identifier.isdigit() and len(identifier) > 1 and identifier.startswith("0"))
        for identifier in identifiers
    ):
        raise ValueError("version must use MAJOR.MINOR.PATCH with an optional prerelease suffix")
    return version


def _run_git_archive(ref: str, fmt: str, prefix: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "git",
        "archive",
        f"--format={fmt}",
        f"--prefix={prefix}",
        ref,
    ]
    with output.open("wb") as stream:
        completed = subprocess.run(command, stdout=stream, check=False)
    if completed.returncode:
        output.unlink(missing_ok=True)
        raise SystemExit(f"git archive failed for {ref!r} ({fmt})")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release version, with or without a leading v")
    parser.add_argument("--ref", default="HEAD", help="Git ref to archive (default: HEAD)")
    parser.add_argument("--output", default="dist", help="Directory for generated archives")
    args = parser.parse_args()

    try:
        version = normalize_version(args.version)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = Path(args.output)
    prefix = f"trae-cn-relay-{version}/"
    tarball = output_dir / f"trae-cn-relay-{version}.tar.gz"
    zipball = output_dir / f"trae-cn-relay-{version}.zip"

    _run_git_archive(args.ref, "tar.gz", prefix, tarball)
    _run_git_archive(args.ref, "zip", prefix, zipball)

    checksums = output_dir / f"trae-cn-relay-{version}.sha256"
    checksums.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in (tarball, zipball)),
        encoding="utf-8",
    )
    print(tarball)
    print(zipball)
    print(checksums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
