"""Create the complete algorithm software delivery package."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "delivery" / "端到端融合跟踪算法软件_完整交付包.zip"
PACKAGE_ROOT = "end_to_end_tracking_software"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    explicit = [
        "scripts/pipelines/run_register_pipeline.py",
        "portable/python/export_onnx.py",
        "portable/python/verify_onnx.py",
        "portable/python/requirements.txt",
        "portable/verify_delivery.ps1",
        "portable/artifacts/coop_tracker_v7.onnx",
        "portable/c/CMakeLists.txt",
        "portable/c/include/tracker.h",
        "portable/c/src/tracker.c",
        "portable/c/src/verify_golden_case.c",
        "portable/c/build/Release/coop_tracker.lib",
        "portable/c/build/Release/verify_golden_case.exe",
        "portable/c/build/Release/onnxruntime.dll",
        "portable/matlab/run_tracker.m",
        "portable/matlab/coop_tracker_mex.mexw64",
        "portable/matlab/onnxruntime.dll",
        "portable/matlab/tracker_mex.c",
        "portable/matlab/build_mex.m",
        "portable/matlab/load_binary_case.m",
        "portable/matlab/verify_golden_case.m",
    ]
    files = {relative: ROOT / relative for relative in explicit}
    files["train.py"] = ROOT / "train.py"
    for directory in [
        ROOT / "src",
        ROOT / "tracking_system",
        ROOT / "results",
        ROOT / "artifacts/metrics",
        ROOT / "artifacts/reports/batch_tests",
        ROOT / "scripts/evaluation",
        ROOT / "portable/artifacts/golden_case",
        ROOT / "portable/third_party/onnxruntime-win-x64-1.18.1",
    ]:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() not in {".rar", ".md"} and not path.name.startswith("~$"):
                relative = path.relative_to(ROOT).as_posix()
                if not any(part in {"__pycache__", "node_modules", "dist", "build", ".vite"} for part in path.parts) and "资料模板" not in relative and (
                    not relative.startswith("src/") or path.suffix.lower() in {".py", ".m"}
                ):
                    files[relative] = path

    for relative, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{relative}: {path}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    records = [
        {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest(path)}
        for relative, path in sorted(files.items())
    ]
    manifest = json.dumps({"files": records}, ensure_ascii=False, indent=2).encode("utf-8")
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for relative, path in sorted(files.items()):
            package.write(path, f"{PACKAGE_ROOT}/{relative}")
        package.writestr(f"{PACKAGE_ROOT}/SHA256清单.json", manifest)

    with zipfile.ZipFile(OUTPUT) as package:
        bad = package.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
        if len(package.infolist()) != len(files) + 1:
            raise RuntimeError("Unexpected package member count")

    package_hash = digest(OUTPUT)
    OUTPUT.with_suffix(".zip.sha256").write_text(
        f"{package_hash}  {OUTPUT.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "archive": str(OUTPUT),
        "archive_bytes": OUTPUT.stat().st_size,
        "files": len(files),
        "sha256": package_hash,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
