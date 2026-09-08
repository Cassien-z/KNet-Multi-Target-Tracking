"""Register raw MAT observations into model-ready PKL tracks.

Observation grouping is performed by measurement-only data association. Ground
truth is attached only after association so it can be used as the learning or
evaluation label.
"""

import argparse
import glob
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_association.pipeline_core import process_file


def _process_task(task):
    mat_path, pkl_path = task
    process_file(mat_path, pkl_path, verbose=False)
    return os.path.basename(mat_path)


def _build_tasks(mat_files, output_dir, overwrite):
    tasks = []
    skipped = 0
    for mat_path in mat_files:
        pkl_name = Path(mat_path).with_suffix(".pkl").name
        pkl_path = os.path.join(output_dir, pkl_name)
        if os.path.exists(pkl_path) and not overwrite:
            skipped += 1
        else:
            tasks.append((mat_path, pkl_path))
    return tasks, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Register MAT files with measurement-only data association."
    )
    parser.add_argument("--in_dir", required=True, help="Directory containing MAT files")
    parser.add_argument("--out_dir", required=True, help="Directory for registered PKL files")
    parser.add_argument("--pattern", default="*.mat", help="Input filename pattern")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PKL files")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.in_dir)
    output_dir = os.path.abspath(args.out_dir)
    os.makedirs(output_dir, exist_ok=True)
    mat_files = sorted(glob.glob(os.path.join(input_dir, args.pattern)))
    if not mat_files:
        parser.error(f"No files matching {args.pattern!r} in {input_dir}")

    tasks, skipped = _build_tasks(mat_files, output_dir, args.overwrite)
    workers = max(1, args.workers)
    print(f"[Info] input={input_dir}")
    print(f"[Info] output={output_dir}")
    print(f"[Info] files={len(mat_files)}, pending={len(tasks)}, skipped={skipped}, workers={workers}")

    completed = 0
    failed = 0
    start = time.perf_counter()

    def report_progress():
        elapsed = max(time.perf_counter() - start, 1e-6)
        speed = (completed + failed) / elapsed
        remaining = (len(tasks) - completed - failed) / max(speed, 1e-6)
        print(
            f"[{completed + failed}/{len(tasks)}] completed={completed} failed={failed} "
            f"speed={speed:.2f} files/s eta={remaining:.0f}s"
        )

    if workers == 1:
        for task in tasks:
            try:
                _process_task(task)
                completed += 1
            except Exception:
                failed += 1
                print(f"[Error] failed: {os.path.basename(task[0])}")
                traceback.print_exc()
            if (completed + failed) % 100 == 0 or completed + failed == len(tasks):
                report_progress()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(_process_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                    completed += 1
                except Exception:
                    failed += 1
                    print(f"[Error] failed: {os.path.basename(task[0])}")
                    traceback.print_exc()
                if (completed + failed) % 100 == 0 or completed + failed == len(tasks):
                    report_progress()

    elapsed = time.perf_counter() - start
    print(f"[Done] completed={completed}, skipped={skipped}, failed={failed}, elapsed={elapsed:.1f}s")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
