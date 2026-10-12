"""Run the complete V7 validation set as one reproducible acceptance batch."""

from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking_system.backend.batch_testing import BatchTestManager
from tracking_system.backend.model_manager import ModelManager


def main() -> None:
    models = ModelManager(device="cuda:0")
    manager = BatchTestManager(models)
    job = manager.create_job(
        name="V7完整验证集结题批测",
        model_id="v7",
        dataset_id="v7",
        device="cuda:0",
        warmup_frames=5,
        batch_size=64,
        workers=0,
        max_scenarios=None,
    )
    job_id = str(job["id"])
    print(f"批测任务：{job_id}", flush=True)
    try:
        while True:
            job = manager.get_job(job_id)
            progress = job["progress"]
            print(
                f"状态：{job['status']}，进度：{progress['percent']}%，"
                f"{progress['message']}",
                flush=True,
            )
            if job["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(15)
    except KeyboardInterrupt:
        manager.cancel_job(job_id)
        raise
    finally:
        manager.shutdown()
    if job["status"] != "completed":
        raise RuntimeError(f"V7批测未完成：{job.get('error')}")
    print(f"报告：{manager.report_dir / (job_id + '.json')}", flush=True)


if __name__ == "__main__":
    main()
