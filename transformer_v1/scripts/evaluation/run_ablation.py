import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
import json

from src.model import CoopTrackerModel
from src.dataset_pkl import RegisteredPklDataset
from train import forward_batch

def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = PROJECT_ROOT / "results" / "runs_standard(v5)" / "run_20260715_194636"
    model_path = os.path.join(run_dir, "best_model.pt")
    data_dir = PROJECT_ROOT / "data" / "val_data_v5_registered"

    print("Loading dataset...")
    dataset = RegisteredPklDataset(data_dir=data_dir, seq_len=100, num_uavs=5, return_extra=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print("Loading model...")
    model = CoopTrackerModel(d_model=96).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    modes = ["full", "dual_no_gat", "radar_only", "no_temporal"]
    results = {}

    print("Evaluating Ablation Modes...")
    with torch.no_grad():
        for mode in modes:
            print(f"Running mode: {mode}")
            sq_errors = []
            
            # evaluate
            for batch in loader:
                # Need to ignore warmup frames (5 frames)
                pred, label_abs, label_pos_rel, pred_pos_rel = forward_batch(model, batch, device, mode=mode)
                
                # Exclude first 5 frames
                dist = torch.norm(pred[:, 5:, [0, 2, 4]] - label_abs[:, 5:, [0, 2, 4]], dim=-1)
                dist_sq = (dist ** 2).cpu().numpy()
                sq_errors.extend(dist_sq.flatten().tolist())
                
            rmse = np.sqrt(np.mean(sq_errors))
            p95 = np.percentile(np.sqrt(sq_errors), 95)
            results[mode] = {"rmse": float(rmse), "p95": float(p95)}
            print(f"  {mode}: RMSE = {rmse:.2f}m, P95 = {p95:.2f}m")

    output_path = PROJECT_ROOT / "artifacts" / "metrics" / "ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    run_ablation()
