"""Audit V7 model components on the exact complete V7 acceptance batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.acceptance_common import job_signature, load_batch_report, selected_files, write_json_atomic
from src.model import CoopTrackerModel
from tracking_system.backend.batch_testing import _BatchRegisteredPklDataset, stable_stage_mask
from train import forward_batch


MODES = ("full", "dual_no_gat", "radar_only", "no_temporal")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class(name: str) -> str:
    match = re.search(r"track[_-](cv|ca|ct|cssw)[_-]", name, re.I)
    if not match:
        raise ValueError(f"Unknown scenario class: {name}")
    return match.group(1).upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", required=True)
    args = parser.parse_args()
    report = load_batch_report(args.batch_report)
    config = report["config"]
    if config["model_id"] != "v7" or config["dataset_id"] != "v7" or config["max_scenarios"] is not None:
        parser.error("requires the complete V7 model and V7 dataset batch")
    paths = selected_files(report, ".pkl")
    model_path = Path(report["result"]["model_path"])
    expected_hash = report["result"]["provenance"]["model_sha256"]
    if _digest(model_path) != expected_hash:
        raise RuntimeError("model hash differs from the acceptance batch")
    dataset = _BatchRegisteredPklDataset(paths, seq_len=100, num_uavs=5)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device(config["device"])
    model = CoopTrackerModel(d_model=96).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = {}
    with torch.inference_mode():
        for mode in MODES:
            errors = []
            class_sums = {name: {"squared_sum": 0.0, "points": 0} for name in ("CV", "CA", "CT", "CSSW")}
            for index, batch in enumerate(loader, start=1):
                prediction, truth, _, _ = forward_batch(model, batch, device, mode=mode)
                distance = torch.linalg.vector_norm(prediction[..., [0, 2, 4]] - truth[..., [0, 2, 4]], dim=-1)
                mask = stable_stage_mask(batch["target_valid_mask"].to(device), 5)
                for row, name in enumerate(batch["file_name"]):
                    values = distance[row][mask[row]].detach().cpu().numpy().astype(np.float64)
                    if not np.isfinite(values).all():
                        raise RuntimeError(f"Non-finite ablation output: {mode}, {name}")
                    errors.append(values)
                    group = class_sums[_class(name)]
                    group["squared_sum"] += float(np.square(values).sum())
                    group["points"] += int(values.size)
                if index % 20 == 0 or index == len(loader):
                    print(f"{mode}: {index}/{len(loader)} batches", flush=True)
            values = np.concatenate(errors)
            rmse = float(np.sqrt(np.mean(np.square(values))))
            results[mode] = {
                "position_rmse_m": rmse,
                "position_p95_m": float(np.percentile(values, 95)),
                "evaluated_points": int(values.size),
                "by_class": {
                    name: {"position_rmse_m": float(np.sqrt(group["squared_sum"] / group["points"])),
                           "evaluated_points": group["points"]}
                    for name, group in class_sums.items()
                },
            }
    expected = float(report["result"]["metrics"]["position_rmse_m"])
    if not np.isclose(results["full"]["position_rmse_m"], expected, atol=1e-4):
        raise RuntimeError("full-mode ablation does not reproduce the acceptance batch RMSE")
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": report["id"],
        "job_signature": job_signature(report),
        "model_sha256": expected_hash,
        "dataset_sha256": report["result"]["provenance"]["dataset_sha256"],
        "warmup_frames_excluded": 5,
        "mode_definition": {
            "full": "完整V7模型",
            "dual_no_gat": "关闭跨平台GAT，保留其他权重",
            "radar_only": "屏蔽光电和电子侦察输入，保留雷达、链路和其他模块",
            "no_temporal": "关闭时序Transformer，保留其他权重",
        },
        "interpretation": "Inference-time component masking; no ablated model was retrained.",
        "results": results,
    }
    destination = PROJECT_ROOT / "artifacts" / "metrics" / "acceptance" / report["id"] / "ablation_v7.json"
    write_json_atomic(destination, payload)
    print(json.dumps({"output": str(destination), "rmse_m": {k: v["position_rmse_m"] for k, v in results.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
