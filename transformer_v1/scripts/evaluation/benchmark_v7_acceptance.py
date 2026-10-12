"""Measure V7 one-frame and 100-frame model-forward latency for one batch job.

Both input shapes are reported because the current temporal model uses a full
sequence in normal batch inference. One-frame timing alone does not establish
the latency of a full 100-frame sequence or online causal operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.acceptance_common import job_signature, load_batch_report, selected_files, write_json_atomic
from scripts.evaluation.review_items_7_9 import runtime_environment, single_frame_batch
from tracking_system.backend.batch_testing import _BatchRegisteredPklDataset, _predict_batch
from tracking_system.backend.runtime import TransformerEngine


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def measure(model: torch.nn.Module, sample: dict, device: torch.device, *, warmup: int, repeats: int) -> dict:
    samples_ms = []
    with torch.inference_mode():
        for _ in range(warmup):
            _predict_batch(model, sample, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(repeats):
            started = time.perf_counter()
            _predict_batch(model, sample, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ms.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "warmup_iterations": warmup,
        "test_iterations": repeats,
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "samples_ms": samples_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, help="Optional evidence path, useful for separate load conditions")
    parser.add_argument("--load-condition", default="unspecified", help="Description of concurrent system load")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 20:
        parser.error("warmup must be non-negative and repeats must be at least 20")

    report = load_batch_report(args.batch_report)
    config = report["config"]
    if config["model_id"] != "v7" or config["dataset_id"] != "v7":
        parser.error("this benchmark requires a V7 model and V7 dataset batch")
    model_path = Path(report["result"]["model_path"])
    expected_sha = report["result"]["provenance"]["model_sha256"]
    if _digest(model_path) != expected_sha:
        raise RuntimeError("V7 model hash differs from the batch report")

    file_path = selected_files(report, ".pkl")[0]
    dataset = _BatchRegisteredPklDataset([file_path], seq_len=100, num_uavs=5)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    device = torch.device(config["device"])
    engine = TransformerEngine(model_path=model_path, data_dir=file_path.parent, device=str(device), seq_len=100)
    model = engine.model.eval()
    one_frame = measure(model, single_frame_batch(batch), device, warmup=args.warmup, repeats=args.repeats)
    full_sequence = measure(model, batch, device, warmup=args.warmup, repeats=args.repeats)
    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": report["id"],
        "job_signature": job_signature(report),
        "model_id": "v7",
        "dataset_id": "v7",
        "model_sha256": expected_sha,
        "sample_file": file_path.name,
        "device": str(device),
        "load_condition": args.load_condition,
        "environment": runtime_environment(device),
        "measurement_scope": "registered tensor input to model state output; synchronized wall-clock forward",
        "one_frame_input": {"sequence_length": 1, **one_frame},
        "full_sequence_input": {"sequence_length": 100, **full_sequence},
        "interpretation": "T=1 and T=100 are separate forward shapes; this non-causal model's T=1 result does not prove online causal operation.",
    }
    destination = args.output or PROJECT_ROOT / "artifacts" / "metrics" / "acceptance" / report["id"] / "forward_latency.json"
    write_json_atomic(destination, output)
    print(json.dumps({
        "output": str(destination),
        "one_frame_p95_ms": one_frame["p95_ms"],
        "full_sequence_p95_ms": full_sequence["p95_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
