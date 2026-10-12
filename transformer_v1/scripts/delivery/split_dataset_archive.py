"""Split the compressed V5-V8 dataset ZIP into transferable 512 MiB parts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "artifacts/delivery/机载协同跟踪仿真数据集_V5-V8_高压缩.zip"
PART_DIR = ROOT / "artifacts/delivery/dataset_v5_v8_parts"
PART_PREFIX = "dataset_v5_v8.zip.part"
PART_SIZE = 512 * 1024 * 1024
CHUNK_SIZE = 4 * 1024 * 1024


def main() -> None:
    if PART_DIR.exists():
        raise FileExistsError(PART_DIR)
    PART_DIR.mkdir(parents=True)

    whole_hash = hashlib.sha256()
    records = []
    with ARCHIVE.open("rb") as source:
        index = 1
        while True:
            name = f"{PART_PREFIX}{index:03d}"
            path = PART_DIR / name
            part_hash = hashlib.sha256()
            count = 0
            with path.open("wb") as output:
                while count < PART_SIZE:
                    chunk = source.read(min(CHUNK_SIZE, PART_SIZE - count))
                    if not chunk:
                        break
                    output.write(chunk)
                    whole_hash.update(chunk)
                    part_hash.update(chunk)
                    count += len(chunk)
            if count == 0:
                path.unlink()
                break
            records.append({"file": name, "size_bytes": count, "sha256": part_hash.hexdigest()})
            print(f"Part {index}: {count} bytes", flush=True)
            index += 1

    archive_hash = whole_hash.hexdigest()
    if sum(item["size_bytes"] for item in records) != ARCHIVE.stat().st_size:
        raise RuntimeError("Split byte count does not match archive")
    manifest = {
        "archive": ARCHIVE.name,
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": archive_hash,
        "part_size_bytes": PART_SIZE,
        "parts": records,
    }
    (PART_DIR / "分卷校验清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    merge_script = f'''$ErrorActionPreference = "Stop"
$folder = Split-Path -Parent $MyInvocation.MyCommand.Path
$parts = @(Get-ChildItem -LiteralPath $folder -File | Where-Object {{ $_.Name -match '^dataset_v5_v8\\.zip\\.part[0-9]{{3}}$' }} | Sort-Object Name)
if ($parts.Count -ne {len(records)}) {{ throw "需要 {len(records)} 个分卷，当前找到 $($parts.Count) 个。" }}
$target = Join-Path $folder "{ARCHIVE.name}"
if (Test-Path -LiteralPath $target) {{ throw "目标文件已存在：$target" }}
$destination = [System.IO.File]::Open($target, [System.IO.FileMode]::CreateNew)
try {{
    foreach ($part in $parts) {{
        $inputFile = [System.IO.File]::OpenRead($part.FullName)
        try {{ $inputFile.CopyTo($destination) }} finally {{ $inputFile.Dispose() }}
    }}
}} finally {{ $destination.Dispose() }}
$actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne "{archive_hash}") {{ throw "合并文件 SHA-256 不一致，请检查分卷是否完整。" }}
Write-Host "合并完成并通过 SHA-256 校验：$target"
'''
    (PART_DIR / "合并分卷.ps1").write_text(merge_script, encoding="utf-8-sig")
    (PART_DIR / "使用说明.txt").write_text(
        "请把本文件夹中的全部 .part001～.part006 分卷放在同一文件夹。\n"
        "在 PowerShell 中运行：powershell -ExecutionPolicy Bypass -File .\\合并分卷.ps1\n"
        "脚本会合并成 ZIP，并自动核对整包 SHA-256。\n"
        "邮件建议只发送分享链接，不发送分卷附件。\n",
        encoding="utf-8-sig",
    )
    print(f"Ready: {len(records)} parts, SHA-256 {archive_hash}")


if __name__ == "__main__":
    main()
