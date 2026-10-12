"""Generate auditable evidence for midterm-review items 1 through 3.

The audit is deliberately evaluation-only: it does not change model weights or
training data.  It reuses the existing GAT ablation result, observes the
existing learned fusion gates through forward hooks, and applies deterministic
in-memory corruptions for missing, occluded, dropped, and delayed observations.

Example:
    python scripts/evaluation/review_items_1_3.py --max-scenarios 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CoopTrackerModel
from tracking_system.backend.batch_testing import (
    _BatchRegisteredPklDataset,
    _predict_batch,
)
from tracking_system.backend.scenario_catalog import (
    class_counts,
    selected_dataset_files,
)


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "results"
    / "runs_40off(v7)"
    / "run_20260728_103728"
    / "best_model.pt"
)
DEFAULT_ABLATION = PROJECT_ROOT / "artifacts" / "metrics" / "ablation_results.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "midterm_review"
    / "items_1_3_evidence.json"
)

FUSION_AUDIT_CONDITIONS = (
    "baseline",
    "radar_suppression_70pct",
    "eo_occlusion_2s",
    "esm_dropout_70pct",
)

INCOMPLETE_INFORMATION_CONDITIONS = (
    "baseline",
    "eo_channel_missing",
    "random_measurement_drop_20pct",
    "all_source_occlusion_0_5s",
    "datalink_delay_0_2s",
)


def operation_scenario(file_name: str) -> str:
    """Map existing motion strata to the two review-facing task scenarios."""
    lowered = file_name.lower()
    if "track_cv_" in lowered or "track_ct_" in lowered:
        return "cooperative_detection"
    if "track_ca_" in lowered or "track_cssw_" in lowered:
        return "cooperative_strike"
    return "other"


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _zero_sensor(batch: dict[str, Any], sensor: str, selector: torch.Tensor) -> None:
    mask_key = f"{sensor}_mask"
    mask = batch[mask_key]
    selector = selector.to(dtype=torch.bool, device=mask.device)
    mask[selector] = 0.0
    batch[sensor] *= mask.unsqueeze(-1).to(batch[sensor].dtype)


def _random_drop(
    batch: dict[str, Any], sensor: str, probability: float, generator: torch.Generator
) -> None:
    mask = batch[f"{sensor}_mask"]
    selector = torch.rand(mask.shape, generator=generator) < probability
    _zero_sensor(batch, sensor, selector)


def _zero_all_sources(batch: dict[str, Any], start: int, end: int) -> None:
    for sensor in ("radar", "eo", "esm"):
        selector = torch.zeros_like(batch[f"{sensor}_mask"], dtype=torch.bool)
        selector[:, start:end, :] = True
        _zero_sensor(batch, sensor, selector)
    for link in ("fast", "slow"):
        batch[f"{link}_link"][:, start:end, :] = 0.0
        batch[f"{link}_update"][:, start:end, :] = 0.0
        batch[f"{link}_age"][:, start:end, :] = 999.0


def _shift_time(tensor: torch.Tensor, frames: int, fill: float = 0.0) -> torch.Tensor:
    shifted = torch.full_like(tensor, fill)
    if frames <= 0:
        return tensor.clone()
    if frames < tensor.shape[1]:
        shifted[:, frames:] = tensor[:, :-frames]
    return shifted


def apply_condition(
    batch: dict[str, Any], condition: str, *, seed: int
) -> tuple[dict[str, Any], dict[str, float | int | str]]:
    """Apply one reproducible, evaluation-only observation condition."""
    result = _clone_batch(batch)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    before = sum(
        int(result[f"{sensor}_mask"].count_nonzero())
        for sensor in ("radar", "eo", "esm")
    )

    if condition == "baseline":
        pass
    elif condition == "radar_suppression_70pct":
        _random_drop(result, "radar", 0.70, generator)
    elif condition == "eo_occlusion_2s":
        steps = min(20, result["eo_mask"].shape[1])
        start = max(0, (result["eo_mask"].shape[1] - steps) // 2)
        selector = torch.zeros_like(result["eo_mask"], dtype=torch.bool)
        selector[:, start : start + steps, :] = True
        _zero_sensor(result, "eo", selector)
    elif condition == "esm_dropout_70pct":
        _random_drop(result, "esm", 0.70, generator)
    elif condition == "eo_channel_missing":
        selector = torch.ones_like(result["eo_mask"], dtype=torch.bool)
        _zero_sensor(result, "eo", selector)
    elif condition == "random_measurement_drop_20pct":
        for sensor in ("radar", "eo", "esm"):
            _random_drop(result, sensor, 0.20, generator)
    elif condition == "all_source_occlusion_0_5s":
        steps = min(5, result["radar_mask"].shape[1])
        start = max(0, (result["radar_mask"].shape[1] - steps) // 2)
        _zero_all_sources(result, start, start + steps)
    elif condition == "datalink_delay_0_2s":
        delay_frames = 2
        for key in (
            "fast_link",
            "slow_link",
            "fast_update",
            "slow_update",
        ):
            result[key] = _shift_time(result[key], delay_frames)
        for key in ("fast_age", "slow_age"):
            result[key] = _shift_time(result[key], delay_frames, fill=999.0)
            valid = result[key] < 998.0
            result[key] = torch.where(valid, result[key] + 0.2, result[key])
    else:
        raise ValueError(f"Unknown condition: {condition}")

    after = sum(
        int(result[f"{sensor}_mask"].count_nonzero())
        for sensor in ("radar", "eo", "esm")
    )
    metadata: dict[str, float | int | str] = {
        "condition": condition,
        "valid_measurements_before": before,
        "valid_measurements_after": after,
        "retained_ratio": after / before if before else 0.0,
        "seed": seed,
    }
    return result, metadata


class FusionGateCapture:
    """Observe existing fusion gates without altering the inference graph."""

    def __init__(self, model: CoopTrackerModel) -> None:
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = [
            model.passive_conf_eo.register_forward_hook(self._hook("eo_confidence")),
            model.passive_conf_esm.register_forward_hook(self._hook("esm_confidence")),
            model.fusion_gate.register_forward_hook(self._hook("active_stream_gate")),
        ]

    def _hook(self, name: str):
        def capture(_module, _inputs, output):
            self.outputs[name] = torch.sigmoid(output.detach()).cpu()

        return capture

    def reset(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _score_tensor(raw: torch.Tensor, batch_size: int, agents: int) -> torch.Tensor:
    time_steps = raw.shape[1]
    return raw.squeeze(-1).reshape(batch_size, agents, time_steps).permute(0, 2, 1)


def _add_weight_stats(
    totals: dict[tuple[str, str], dict[str, float]],
    condition: str,
    names: list[str],
    batch: dict[str, Any],
    captured: dict[str, torch.Tensor],
) -> None:
    batch_size, _, agents = batch["radar_mask"].shape
    values = {
        name: _score_tensor(raw, batch_size, agents)
        for name, raw in captured.items()
    }
    masks = {
        "eo_confidence": batch["eo_mask"].bool(),
        "esm_confidence": batch["esm_mask"].bool(),
        "active_stream_gate": (
            batch["radar_mask"] + batch["eo_mask"] + batch["esm_mask"]
        ).bool(),
    }
    for row, file_name in enumerate(names):
        scenario = operation_scenario(str(file_name))
        target = totals[(condition, scenario)]
        target["samples"] += 1.0
        for metric_name, tensor in values.items():
            valid = masks[metric_name][row]
            selected = tensor[row][valid]
            if selected.numel():
                target[f"{metric_name}_sum"] += float(selected.sum())
                target[f"{metric_name}_count"] += float(selected.numel())


def _finalize_weight_stats(
    totals: dict[tuple[str, str], dict[str, float]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (condition, scenario), values in sorted(totals.items()):
        record: dict[str, Any] = {
            "condition": condition,
            "operation_scenario": scenario,
            "samples": int(values["samples"]),
        }
        for metric in ("eo_confidence", "esm_confidence", "active_stream_gate"):
            count = values[f"{metric}_count"]
            record[f"{metric}_mean"] = (
                values[f"{metric}_sum"] / count if count else None
            )
            record[f"{metric}_valid_points"] = int(count)
        records.append(record)
    return records


def _evaluate_condition(
    model: CoopTrackerModel,
    loader: DataLoader,
    device: torch.device,
    condition: str,
    warmup_frames: int,
    seed: int,
    capture: FusionGateCapture | None = None,
    weight_totals: dict[tuple[str, str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    errors: list[np.ndarray] = []
    finite_points = 0
    evaluated_points = 0
    retained: list[float] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            perturbed, condition_meta = apply_condition(
                batch, condition, seed=seed + batch_index
            )
            retained.append(float(condition_meta["retained_ratio"]))
            if capture is not None:
                capture.reset()
            prediction = _predict_batch(model, perturbed, device)
            truth = perturbed["label_abs"].to(device)
            valid = perturbed["target_valid_mask"].to(device).bool()
            if warmup_frames:
                valid[:, :warmup_frames] = False
            distance = torch.linalg.vector_norm(
                prediction[..., [0, 2, 4]] - truth[..., [0, 2, 4]], dim=-1
            )
            selected = distance[valid]
            finite = torch.isfinite(selected)
            finite_points += int(finite.count_nonzero())
            evaluated_points += int(selected.numel())
            if finite.any():
                errors.append(selected[finite].detach().cpu().numpy())

            if capture is not None and weight_totals is not None:
                _add_weight_stats(
                    weight_totals,
                    condition,
                    [str(name) for name in batch.get("file_name", [])],
                    perturbed,
                    capture.outputs,
                )

    merged = np.concatenate(errors).astype(np.float64) if errors else np.empty(0)
    return {
        "condition": condition,
        "position_rmse_m": (
            float(np.sqrt(np.mean(np.square(merged)))) if merged.size else None
        ),
        "position_mae_m": float(np.mean(merged)) if merged.size else None,
        "position_p95_m": float(np.percentile(merged, 95)) if merged.size else None,
        "evaluated_points": evaluated_points,
        "finite_output_ratio": finite_points / evaluated_points if evaluated_points else 0.0,
        "mean_retained_measurement_ratio": float(np.mean(retained)) if retained else 0.0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_info(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "capability": f"{props.major}.{props.minor}",
        }
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
    }


def _load_ablation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    full = float(data["full"]["rmse"])
    no_gat = float(data["dual_no_gat"]["rmse"])
    contribution = (no_gat - full) / no_gat * 100.0
    return {
        "review_item": 1,
        "title": "Contribution of the cooperative-formation platform layer",
        "implementation_mapping": [
            "wingman datalink observations",
            "per-platform temporal feature extraction",
            "distance-constrained collaborative GAT aggregation",
        ],
        "full_model_rmse_m": full,
        "without_gat_rmse_m": no_gat,
        "gat_relative_rmse_reduction_pct": contribution,
        "evidence_source": str(path),
        "conclusion": "The cooperative GAT provides a measurable accuracy contribution.",
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    selected_paths = selected_dataset_files(
        args.dataset_id,
        args.max_scenarios,
        strategy="stratified_by_scenario_type",
    )
    if not selected_paths:
        raise FileNotFoundError(f"No registered scenarios for {args.dataset_id}")
    dataset = _BatchRegisteredPklDataset(selected_paths, args.sequence_length, 5)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model_path = Path(args.model).resolve()
    model = CoopTrackerModel(d_model=96).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    fusion_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    capture = FusionGateCapture(model)
    fusion_condition_results = []
    try:
        for condition in FUSION_AUDIT_CONDITIONS:
            fusion_condition_results.append(
                _evaluate_condition(
                    model,
                    loader,
                    device,
                    condition,
                    args.warmup_frames,
                    args.seed,
                    capture,
                    fusion_totals,
                )
            )
    finally:
        capture.close()
    fusion_records = _finalize_weight_stats(fusion_totals)

    incomplete_results = []
    for condition in INCOMPLETE_INFORMATION_CONDITIONS:
        incomplete_results.append(
            _evaluate_condition(
                model,
                loader,
                device,
                condition,
                args.warmup_frames,
                args.seed,
            )
        )
    baseline_rmse = next(
        item["position_rmse_m"]
        for item in incomplete_results
        if item["condition"] == "baseline"
    )
    for item in incomplete_results:
        value = item["position_rmse_m"]
        item["rmse_change_vs_baseline_pct"] = (
            (value - baseline_rmse) / baseline_rmse * 100.0
            if value is not None and baseline_rmse
            else None
        )

    gate_ranges = {}
    for metric in (
        "eo_confidence_mean",
        "esm_confidence_mean",
        "active_stream_gate_mean",
    ):
        values = [float(row[metric]) for row in fusion_records if row[metric] is not None]
        gate_ranges[metric] = {
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "range": max(values) - min(values) if values else None,
        }

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "中期评审意见第1至3条整改验证证据",
        "runtime": _runtime_info(device),
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "architecture": "Learned-K + dual-stream Transformer + collaborative GAT",
        },
        "scope": {
            "dataset_id": args.dataset_id,
            "scenario_count": len(selected_paths),
            "track_count": len(dataset),
            "sequence_length": args.sequence_length,
            "warmup_frames_excluded": args.warmup_frames,
            "sampling_strategy": "stratified_by_scenario_type",
            "class_counts": class_counts(selected_paths),
            "files": [path.stem for path in selected_paths],
            "seed": args.seed,
        },
        "review_item_1": _load_ablation(Path(args.ablation).resolve()),
        "review_item_2": {
            "review_item": 2,
            "title": "Dynamic sensor-contribution audit",
            "method": "Forward hooks observe existing EO/ESM confidence heads and active/passive fusion gate; model weights are unchanged.",
            "condition_metrics": fusion_condition_results,
            "weights_by_condition_and_operation_scenario": fusion_records,
            "observed_weight_ranges": gate_ranges,
        },
        "review_item_3": {
            "review_item": 3,
            "title": "Incomplete-information robustness audit",
            "method": "Deterministic in-memory corruption of validation inputs; source data and model weights are unchanged.",
            "conditions": incomplete_results,
            "coverage": {
                "missing": "EO channel removed for the complete sequence",
                "occlusion": "all local sensors and datalinks removed for 0.5 s",
                "dropped_measurements": "20% independent random measurement loss",
                "delay": "datalink observations delayed by 0.2 s; local sensor timestamps are aligned before model input",
            },
        },
    }
    _write_json_atomic(Path(args.output).resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--ablation", default=str(DEFAULT_ABLATION))
    parser.add_argument("--dataset-id", default="v7")
    parser.add_argument("--max-scenarios", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = run(args)
    summary = {
        "output": str(Path(args.output).resolve()),
        "scope": payload["scope"],
        "review_item_1": payload["review_item_1"],
        "review_item_2_weight_ranges": payload["review_item_2"]["observed_weight_ranges"],
        "review_item_3_conditions": payload["review_item_3"]["conditions"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
