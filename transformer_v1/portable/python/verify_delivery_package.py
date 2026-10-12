"""Verify the integrity and completeness of a delivery ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PACKAGE_ROOT = "coop_tracker_delivery"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()

    with zipfile.ZipFile(package, "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt}")

        manifest_name = f"{PACKAGE_ROOT}/delivery_manifest.json"
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        expected = {
            f"{PACKAGE_ROOT}/{record['archive_path']}": record
            for record in manifest["files"]
        }
        actual = {
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
        }
        allowed_extra = {
            manifest_name,
            f"{PACKAGE_ROOT}/SHA256SUMS.txt",
        }
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected) - allowed_extra)
        if missing or unexpected:
            raise RuntimeError(
                f"Archive membership mismatch: missing={missing}, unexpected={unexpected}"
            )

        for member, record in expected.items():
            payload = archive.read(member)
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != record["size_bytes"] or digest != record["sha256"]:
                raise RuntimeError(f"Integrity mismatch: {member}")

    print(json.dumps({
        "package": str(package),
        "package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        "verified_files": len(expected),
        "acceptance_status": manifest["acceptance"]["overall_status"],
        "integrity": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
