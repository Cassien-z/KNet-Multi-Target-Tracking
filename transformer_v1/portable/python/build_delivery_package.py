"""Build the auditable MATLAB/C algorithm delivery ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTABLE_ROOT = PROJECT_ROOT / "portable"
JOB_ID = "2ea4f9eb0f2845b599fcf4837d71a018"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_tree(files: dict[str, Path], source: Path, archive_root: str, *, exclude=()) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(path.match(pattern) for pattern in exclude):
            continue
        relative = path.relative_to(source).as_posix()
        files[f"{archive_root}/{relative}"] = path


def collect_files() -> dict[str, Path]:
    files: dict[str, Path] = {}

    def add(relative: str, archive_path: str | None = None) -> None:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[archive_path or relative.replace("\\", "/")] = path

    for relative in (
        "portable/README.md",
        "portable/INTERFACE.md",
        "portable/verify_delivery.ps1",
        "portable/artifacts/coop_tracker_v7.onnx",
        "portable/artifacts/export_metadata.json",
        "portable/c/CMakeLists.txt",
        "portable/c/include/tracker.h",
        "portable/c/src/tracker.c",
        "portable/c/src/verify_golden_case.c",
        "portable/c/build/Release/verify_golden_case.exe",
        "portable/c/build/Release/coop_tracker.lib",
        "portable/c/build/Release/onnxruntime.dll",
        "portable/matlab/build_mex.m",
        "portable/matlab/load_binary_case.m",
        "portable/matlab/run_tracker.m",
        "portable/matlab/tracker_mex.c",
        "portable/matlab/verify_golden_case.m",
        "portable/matlab/coop_tracker_mex.mexw64",
        "portable/matlab/onnxruntime.dll",
        "portable/python/build_delivery_package.py",
        "portable/python/verify_delivery_package.py",
        "portable/python/export_onnx.py",
        "portable/python/verify_onnx.py",
        "portable/python/requirements.txt",
    ):
        add(relative)

    add_tree(
        files,
        PORTABLE_ROOT / "artifacts" / "golden_case",
        "portable/artifacts/golden_case",
    )

    ort_root = PORTABLE_ROOT / "third_party" / "onnxruntime-win-x64-1.18.1"
    for name in (
        "LICENSE",
        "ThirdPartyNotices.txt",
        "Privacy.md",
        "README.md",
        "VERSION_NUMBER",
        "GIT_COMMIT_ID",
        "CodeSignSummary-d6ab774a-eb9d-42b1-9076-ca1b60a61a7f.md",
    ):
        path = ort_root / name
        files[f"portable/third_party/onnxruntime-win-x64-1.18.1/{name}"] = path
    add_tree(
        files,
        ort_root / "include",
        "portable/third_party/onnxruntime-win-x64-1.18.1/include",
    )
    add_tree(
        files,
        ort_root / "lib",
        "portable/third_party/onnxruntime-win-x64-1.18.1/lib",
        exclude=("*.pdb",),
    )

    add(
        f"artifacts/reports/batch_tests/{JOB_ID}.json",
        f"evidence/batch_tests/{JOB_ID}.json",
    )
    for name in (
        "association_accuracy.json",
        "rmse_improvement_pct.json",
        "forward_latency.json",
        "ablation_v7.json",
    ):
        add(
            f"artifacts/metrics/acceptance/{JOB_ID}/{name}",
            f"evidence/acceptance/{name}",
        )
    add(
        f"artifacts/reports/v7_final_validation/{JOB_ID}/四项指标评估.json",
        "evidence/四项指标评估.json",
    )
    add("docs/结题技术总结_V7验证证据说明.md", "evidence/验证证据说明.md")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_name = f"coop_tracker_matlab_c_delivery_{datetime.now():%Y%m%d}.zip"
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "delivery" / default_name,
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    batch_path = PROJECT_ROOT / "artifacts" / "reports" / "batch_tests" / f"{JOB_ID}.json"
    assessment_path = (
        PROJECT_ROOT / "artifacts" / "reports" / "v7_final_validation" /
        JOB_ID / "四项指标评估.json"
    )
    export_metadata_path = PORTABLE_ROOT / "artifacts" / "export_metadata.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    export_metadata = json.loads(export_metadata_path.read_text(encoding="utf-8"))

    source_model = Path(batch["result"]["model_path"])
    expected_model_hash = batch["result"]["provenance"]["model_sha256"]
    if sha256(source_model) != expected_model_hash:
        raise RuntimeError("Current V7 checkpoint does not match the acceptance batch")
    if Path(export_metadata["model"]).resolve() != source_model.resolve():
        raise RuntimeError("Exported ONNX metadata refers to a different source checkpoint")
    if assessment.get("overall_status") != "passed" or not assessment.get("formal_evidence_complete"):
        raise RuntimeError("Acceptance evidence is not complete and passed")

    files = collect_files()
    records = []
    for archive_path, source_path in sorted(files.items()):
        records.append({
            "archive_path": archive_path,
            "size_bytes": source_path.stat().st_size,
            "sha256": sha256(source_path),
        })

    criteria = {
        item["id"]: {
            "value": item["value"],
            "operator": item["operator"],
            "threshold": item["threshold"],
            "unit": item["unit"],
            "passed": item["passed"],
        }
        for item in assessment["criteria"]
    }
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_name": output.name,
        "platform": "Windows x64",
        "entry_points": {
            "matlab": "portable/matlab/run_tracker.m",
            "c": "portable/c/include/tracker.h",
            "verification": "portable/verify_delivery.ps1",
        },
        "fixed_interface": {
            "batch_size": 1,
            "sequence_length": 100,
            "uav_count": 5,
            "output_state": ["px", "vx", "py", "vy", "pz", "vz"],
        },
        "source_checkpoint": {
            "path": str(source_model),
            "sha256": expected_model_hash,
        },
        "onnx_model": {
            "path": "portable/artifacts/coop_tracker_v7.onnx",
            "sha256": sha256(PORTABLE_ROOT / "artifacts" / "coop_tracker_v7.onnx"),
        },
        "acceptance": {
            "job_id": JOB_ID,
            "job_signature": next(
                json.loads((PROJECT_ROOT / "artifacts" / "metrics" / "acceptance" / JOB_ID / name)
                           .read_text(encoding="utf-8"))["job_signature"]
                for name in ("association_accuracy.json",)
            ),
            "dataset_sha256": batch["result"]["provenance"]["dataset_sha256"],
            "evaluated_scenarios": batch["result"]["metrics"]["evaluated_scenarios"],
            "evaluated_tracks": batch["result"]["metrics"]["evaluated_tracks"],
            "evaluated_points": batch["result"]["metrics"]["evaluated_points"],
            "overall_status": assessment["overall_status"],
            "criteria": criteria,
        },
        "files": records,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    sums = "".join(f"{record['sha256']}  {record['archive_path']}\n" for record in records)

    package_root = "coop_tracker_delivery"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_path, source_path in sorted(files.items()):
            archive.write(source_path, f"{package_root}/{archive_path}")
        archive.writestr(f"{package_root}/delivery_manifest.json", manifest_bytes)
        archive.writestr(f"{package_root}/SHA256SUMS.txt", sums.encode("utf-8"))

    external_manifest = output.with_suffix(".manifest.json")
    external_manifest.write_bytes(manifest_bytes)
    package_hash = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{package_hash}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "package": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": package_hash,
        "manifest": str(external_manifest),
        "file_count": len(records),
        "acceptance_status": assessment["overall_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
