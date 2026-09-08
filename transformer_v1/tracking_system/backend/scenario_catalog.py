"""Validation dataset and acceptance-scenario catalog.

The acceptance scenarios are logical views over validation-only data.  They do
not participate in training and therefore do not require new model weights.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tracking_system.backend.runtime import PROJECT_ROOT


DATASET_CATALOG: dict[str, dict[str, Any]] = {
    "scenario_cooperative": {
        "name": "想定一：协同编队目标跟踪",
        "description": "多平台雷达、光电、电子侦察与数据链协同，覆盖 CV/CA 常规机动",
        "kind": "acceptance_scenario",
        "path": PROJECT_ROOT / "data" / "val_data_v7_registered",
        "raw_path": PROJECT_ROOT / "data" / "val_data_v7",
        "file_prefixes": ("track_cv_", "track_ca_"),
        "scenario_classes": ("CV", "CA"),
        "sensor_modalities": ("雷达", "光电", "电子侦察", "协同数据链"),
        "interference_modes": ("无专项干扰",),
        "scenario_note": "复用 V7 验证数据的 CV/CA 子集，仅用于测试，不进入训练。",
    },
    "scenario_maneuver_jamming": {
        "name": "想定二：高机动强干扰跟踪",
        "description": "多目标高机动条件下叠加调幅闪烁、软压制与距离抖动干扰",
        "kind": "acceptance_scenario",
        "path": PROJECT_ROOT / "data" / "val_data_v8_interference_registered",
        "raw_path": PROJECT_ROOT / "data" / "val_data_v8_interference",
        "scenario_classes": ("COMPOSITE-JAMMING",),
        "sensor_modalities": ("雷达", "光电", "电子侦察", "协同数据链"),
        "interference_modes": ("调幅闪烁", "软压制", "距离抖动"),
        "scenario_note": "当前 V8 样本在同一想定中复合施加三类干扰；后续独立模式样本可直接追加。",
    },
    "v5": {
        "name": "V5 标准验证集",
        "description": "标准噪声条件注册验证集",
        "kind": "technical_dataset",
        "path": PROJECT_ROOT / "data" / "val_data_v5_registered",
        "raw_path": PROJECT_ROOT / "data" / "val_data_v5",
        "scenario_classes": ("CV", "CA", "CT", "CSSW"),
    },
    "v6": {
        "name": "V6 降噪验证集",
        "description": "相对标准条件降低 30% 噪声的注册验证集",
        "kind": "technical_dataset",
        "path": PROJECT_ROOT / "data" / "val_data_v6_registered",
        "raw_path": PROJECT_ROOT / "data" / "val_data_v6",
        "scenario_classes": ("CV", "CA", "CT", "CSSW"),
    },
    "v7": {
        "name": "V7 低噪验证集",
        "description": "相对标准条件降低 40% 噪声的注册验证集",
        "kind": "technical_dataset",
        "path": PROJECT_ROOT / "data" / "val_data_v7_registered",
        "raw_path": PROJECT_ROOT / "data" / "val_data_v7",
        "scenario_classes": ("CV", "CA", "CT", "CSSW"),
    },
    "v8_interference": {
        "name": "V8 强干扰验证集",
        "description": "三类干扰复合施加的强干扰注册验证集",
        "kind": "technical_dataset",
        "path": PROJECT_ROOT / "data" / "val_data_v8_interference_registered",
        "raw_path": PROJECT_ROOT / "data" / "val_data_v8_interference",
        "scenario_classes": ("COMPOSITE-JAMMING",),
        "interference_modes": ("调幅闪烁", "软压制", "距离抖动"),
    },
}

_CLASS_PATTERN = re.compile(
    r"^track[_-](cv|ca|ct|cssw|blinking|suppression|smearing)(?:[_-]|$)", re.I
)


def scenario_class(path: Path) -> str:
    """Return a stable stratum label derived only from the scenario filename."""
    match = _CLASS_PATTERN.search(path.stem)
    if match:
        return match.group(1).upper()
    if "highfi" in path.stem.lower() or "interference" in path.stem.lower():
        return "COMPOSITE-JAMMING"
    return "OTHER"


def dataset_files(dataset_id: str, suffix: str = ".pkl") -> list[Path]:
    metadata = DATASET_CATALOG[dataset_id]
    root_key = "raw_path" if suffix == ".mat" else "path"
    root = Path(metadata[root_key]).resolve()
    prefixes = tuple(str(item).lower() for item in metadata.get("file_prefixes", ()))
    paths = sorted(root.glob(f"*{suffix}")) if root.is_dir() else []
    if prefixes:
        paths = [path for path in paths if path.stem.lower().startswith(prefixes)]
    return paths


def select_stratified(paths: Iterable[Path], maximum: int | None) -> list[Path]:
    """Select a deterministic, balanced sample across filename-derived strata."""
    ordered = sorted(paths)
    if maximum is None or maximum >= len(ordered):
        return ordered
    if maximum <= 0:
        return []
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in ordered:
        groups[scenario_class(path)].append(path)
    selected: list[Path] = []
    group_names = sorted(groups)
    offset = 0
    while len(selected) < maximum:
        added = False
        for name in group_names:
            group = groups[name]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        offset += 1
    return selected


def selected_dataset_files(
    dataset_id: str,
    maximum: int | None,
    *,
    suffix: str = ".pkl",
    strategy: str = "stratified_by_scenario_type",
) -> list[Path]:
    paths = dataset_files(dataset_id, suffix)
    if strategy == "leading_sorted":
        return paths[:maximum] if maximum is not None else paths
    return select_stratified(paths, maximum)


def class_counts(paths: Iterable[Path]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for path in paths:
        counts[scenario_class(path)] += 1
    return dict(sorted(counts.items()))
