"""Verify the packaged ONNX model without importing the PyTorch project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("portable/artifacts/coop_tracker_v7.onnx"))
    parser.add_argument("--case", type=Path, default=Path("portable/artifacts/golden_case/golden_case.npz"))
    args = parser.parse_args()

    case = np.load(args.case)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inputs = {item.name: case[item.name].astype(np.float32) for item in session.get_inputs()}
    actual = session.run(["prediction"], inputs)[0]
    expected = case["prediction"].astype(np.float32)
    delta = np.abs(actual - expected)
    position_delta = np.linalg.norm(actual[..., [0, 2, 4]] - expected[..., [0, 2, 4]], axis=-1)
    velocity_delta = delta[..., [1, 3, 5]]
    result = {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "max_position_delta_m": float(position_delta.max()),
        "mean_position_delta_m": float(position_delta.mean()),
        "max_velocity_abs_error": float(velocity_delta.max()),
    }
    for name, value in result.items():
        print(f"{name}={value:.9g}")
    if result["max_position_delta_m"] > 1.0 or result["max_velocity_abs_error"] > 0.2:
        raise SystemExit("ONNX output is outside the packaged golden-case tolerance")


if __name__ == "__main__":
    main()
