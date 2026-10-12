"""Package the original V5-V8 MATLAB datasets for delivery."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "delivery"
ARCHIVE_NAME = "机载协同跟踪仿真数据集_V5-V8_原始数据.zip"
COMPRESSED_ARCHIVE_NAME = "机载协同跟踪仿真数据集_V5-V8_高压缩.zip"
DESCRIPTION = ROOT / "docs" / "dataset_delivery" / "数据集交付清单.md"
DATASETS = {
    "train_data_v5": 8000,
    "val_data_v5": 2000,
    "train_data_v6": 8000,
    "val_data_v6": 2000,
    "train_data_v7": 8000,
    "val_data_v7": 2000,
    "val_data_v8_interference": 1,
}
EXPECTED_BYTES = 28_576_566_736


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compressed", action="store_true", help="Use ZIP Deflate compression")
    args = parser.parse_args()
    archive = OUTPUT_DIR / (COMPRESSED_ARCHIVE_NAME if args.compressed else ARCHIVE_NAME)
    partial = archive.with_suffix(".zip.partial")

    files: list[Path] = []
    for name, count in DATASETS.items():
        directory = ROOT / "data" / name
        items = sorted(directory.iterdir())
        if len(items) != count or any(not p.is_file() or p.suffix != ".mat" for p in items):
            raise RuntimeError(f"Unexpected contents in {directory}")
        files.extend(items)

    total_bytes = sum(p.stat().st_size for p in files)
    if len(files) != 30_001 or total_bytes != EXPECTED_BYTES:
        raise RuntimeError(f"Source count/size changed: {len(files)} files, {total_bytes} bytes")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if partial.exists() or archive.exists():
        raise FileExistsError("Delivery archive or partial archive already exists")

    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest)
    writer.writerow(("path", "size_bytes"))

    method = zipfile.ZIP_DEFLATED if args.compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(partial, "w", compression=method, compresslevel=6, allowZip64=True) as package:
        last_directory = None
        for source in files:
            relative = source.relative_to(ROOT).as_posix()
            package.write(source, arcname=relative)
            writer.writerow((relative, source.stat().st_size))
            if source.parent != last_directory:
                if last_directory is not None:
                    print(f"Finished {last_directory.name}", flush=True)
                last_directory = source.parent
        print(f"Finished {last_directory.name}", flush=True)
        package.writestr("交付说明/数据集交付清单.md", DESCRIPTION.read_bytes())
        package.writestr("交付说明/文件目录.csv", manifest.getvalue().encode("utf-8-sig"))

    with zipfile.ZipFile(partial) as package:
        members = package.infolist()
        data_members = [item for item in members if item.filename.startswith("data/")]
        if len(data_members) != len(files) or sum(item.file_size for item in data_members) != total_bytes:
            raise RuntimeError("Archive content count/size does not match source")
        bad = package.testzip()
        if bad:
            raise RuntimeError(f"Archive integrity check failed: {bad}")

    partial.rename(archive)
    print(f"Ready: {archive} ({archive.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
