import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.model import CoopTrackerModel
from src.dataset_pkl import RegisteredPklDataset
from train import forward_batch

def evaluate_combination(model_path, data_dir, device):
    print(f"Loading dataset from {data_dir}...")
    sys.stdout.flush()
    dataset = RegisteredPklDataset(data_dir=data_dir, seq_len=100, num_uavs=5, return_extra=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = CoopTrackerModel(d_model=96).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_sq_errs = []
    with torch.no_grad():
        for batch in loader:
            pred, label_abs, _, _ = forward_batch(model, batch, device, mode="full")
            # pred: [B, T, 6], label_abs: [B, T, 6]
            # Ignore the first 5 frames as warmup (same as training script)
            pred = pred[:, 5:, :]
            label_abs = label_abs[:, 5:, :]
            dist = torch.norm(pred[:, :, [0, 2, 4]] - label_abs[:, :, [0, 2, 4]], dim=-1)
            dist_sq = (dist ** 2).cpu().numpy()
            all_sq_errs.extend(dist_sq.flatten().tolist())

    if len(all_sq_errs) == 0:
        return 0.0
    return float(np.sqrt(np.mean(all_sq_errs)))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "Model-v5 (Standard)": PROJECT_ROOT / "results" / "runs_standard(v5)" / "run_20260715_194636" / "best_model.pt",
        "Model-v6 (-30% Noise)": PROJECT_ROOT / "results" / "runs_30off(v6)" / "run_20260717_152359" / "best_model.pt",
        "Model-v7 (-40% Noise)": PROJECT_ROOT / "results" / "runs_40off(v7)" / "run_20260728_103728" / "best_model.pt",
    }
    
    datasets = {
        "Val-v5 (Standard)": PROJECT_ROOT / "data" / "val_data_v5_registered",
        "Val-v6 (-30% Noise)": PROJECT_ROOT / "data" / "val_data_v6_registered",
        "Val-v7 (-40% Noise)": PROJECT_ROOT / "data" / "val_data_v7_registered",
        "Val-v8 (Interference)": PROJECT_ROOT / "data" / "val_data_v8_interference_registered_10hz",
    }

    results_matrix = {}
    
    for m_name, m_path in models.items():
        results_matrix[m_name] = {}
        for d_name, d_path in datasets.items():
            print(f"--- Evaluating {m_name} on {d_name} ---")
            sys.stdout.flush()
            rmse = evaluate_combination(m_path, d_path, device)
            print(f"RMSE: {rmse:.2f} m")
            sys.stdout.flush()
            results_matrix[m_name][d_name] = rmse

    output_file = PROJECT_ROOT / "artifacts" / "metrics" / "cross_dataset_rmse.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_matrix, f, indent=4)
    print(f"\nSaved results to {output_file}")

if __name__ == "__main__":
    main()
