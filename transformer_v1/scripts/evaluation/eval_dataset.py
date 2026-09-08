"""Evaluate one trained tracker on a registered PKL validation set.

Reports position errors after a configurable warm-up and, when present,
aggregates the association audit records written by ``pipeline_core``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_pkl import RegisteredPklDataset
from src.model import CoopTrackerModel
from train_cv import forward_batch


def summarize_association(data_dir: Path, allowed_names: set[str] | None = None) -> dict | None:
    """Aggregate post-association truth audit counters, if available."""
    correct = 0
    total = 0
    per_sensor: dict[str, dict[str, int | float | None]] = {}
    audited_files = 0
    leakage_flags: list[str] = []

    for path in sorted(data_dir.glob("*.pkl")):
        if allowed_names is not None and path.name not in allowed_names:
            continue
        with path.open("rb") as handle:
            stats = pickle.load(handle).get("AssociationStats")
        if not stats:
            continue
        audited_files += 1
        correct += int(stats.get("correct", 0))
        total += int(stats.get("total", 0))
        if stats.get("truth_used_for_observation_grouping", True):
            leakage_flags.append(path.name)
        for sensor, values in stats.get("per_sensor", {}).items():
            dst = per_sensor.setdefault(sensor, {"correct": 0, "total": 0})
            dst["correct"] = int(dst["correct"]) + int(values.get("correct", 0))
            dst["total"] = int(dst["total"]) + int(values.get("total", 0))

    if audited_files == 0:
        return None
    for values in per_sensor.values():
        sensor_total = int(values["total"])
        values["accuracy"] = int(values["correct"]) / sensor_total if sensor_total else None
    return {
        "method": "measurement_only_hungarian_with_temporal_memory",
        "audited_files": audited_files,
        "truth_used_for_observation_grouping": bool(leakage_flags),
        "files_flagged_for_truth_leakage": leakage_flags,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "per_sensor": per_sensor,
    }


def evaluate(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data).resolve()
    model_path = Path(args.model).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RegisteredPklDataset(
        data_dir=str(data_dir), seq_len=args.seq_len, num_uavs=5, return_extra=True
    )
    allowed_names: set[str] | None = None
    if args.association_subset != "all":
        allowed_names = set()
        for path in data_dir.glob("*.pkl"):
            with path.open("rb") as handle:
                stats = pickle.load(handle).get("AssociationStats")
            if not stats or not stats.get("total"):
                continue
            is_perfect = int(stats["correct"]) == int(stats["total"])
            if (args.association_subset == "perfect") == is_perfect:
                allowed_names.add(path.name)
        dataset.index_map = [
            item for item in dataset.index_map
            if Path(dataset.file_list[item[0]]).name in allowed_names
        ]
    if not dataset:
        raise RuntimeError(f"No registered targets found in {data_dir}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = CoopTrackerModel(d_model=96).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()

    point_errors: list[np.ndarray] = []
    track_rmses: list[np.ndarray] = []
    valid_points = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            pred, label_abs, _, _ = forward_batch(model, batch, device, mode="full")
            distance = torch.linalg.vector_norm(
                pred[..., [0, 2, 4]] - label_abs[..., [0, 2, 4]], dim=-1
            )
            mask = batch["target_valid_mask"].to(device).bool()
            if args.warmup:
                mask[:, : args.warmup] = False

            selected = distance[mask]
            if selected.numel():
                point_errors.append(selected.cpu().numpy())
                valid_points += selected.numel()

            per_track = []
            for row in range(distance.shape[0]):
                row_values = distance[row][mask[row]]
                if row_values.numel():
                    per_track.append(torch.sqrt(torch.mean(row_values.square())))
            if per_track:
                track_rmses.append(torch.stack(per_track).cpu().numpy())

            if batch_idx % 25 == 0 or batch_idx == len(loader):
                print(f"evaluated {batch_idx}/{len(loader)} batches", flush=True)

    if not point_errors:
        raise RuntimeError("No valid target points remained after masking")
    errors = np.concatenate(point_errors).astype(np.float64)
    rmses = np.concatenate(track_rmses).astype(np.float64)
    metrics = {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "position_mae_m": float(np.mean(errors)),
        "position_median_m": float(np.median(errors)),
        "position_p95_m": float(np.percentile(errors, 95)),
        "position_max_m": float(np.max(errors)),
        "mean_track_rmse_m": float(np.mean(rmses)),
        "evaluated_tracks": int(rmses.size),
        "evaluated_points": int(valid_points),
    }
    return {
        "model": str(model_path),
        "dataset": str(data_dir),
        "device": str(device),
        "sequence_length": args.seq_len,
        "warmup_frames": args.warmup,
        "association_subset": args.association_subset,
        "metrics": metrics,
        "association_audit": summarize_association(data_dir, allowed_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--data", required=True, help="Registered PKL directory")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--association-subset",
        choices=("all", "perfect", "imperfect"),
        default="all",
        help="Optionally evaluate only files selected by their post-hoc association audit",
    )
    args = parser.parse_args()

    report = evaluate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"saved report: {output}")


if __name__ == "__main__":
    main()
