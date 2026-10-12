import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

from src.model import CoopTrackerModel
from src.dataset_pkl import RegisteredPklDataset
from train import forward_batch

def evaluate_by_mode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = PROJECT_ROOT / "results" / "runs_standard(v5)" / "run_20260715_194636"
    model_path = os.path.join(run_dir, "best_model.pt")
    data_dir = PROJECT_ROOT / "data" / "val_data_v6_registered"

    print("Loading dataset...")
    dataset = RegisteredPklDataset(data_dir=data_dir, seq_len=100, num_uavs=5, return_extra=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print("Loading model...")
    model = CoopTrackerModel(d_model=96).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Dictionary to store squared errors and counts for each mode
    mode_sq_errors = defaultdict(list)

    print("Evaluating...")
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            pred, label_abs, _, _ = forward_batch(model, batch, device, mode="full")
            
            # calculate squared errors for each sample in the batch
            # pred: [B, T, 6], label_abs: [B, T, 6]
            # dist is [B, T]
            dist = torch.norm(pred[:, :, [0, 2, 4]] - label_abs[:, :, [0, 2, 4]], dim=-1)
            dist_sq = (dist ** 2).cpu().numpy() # [B, T]
            
            # map back to filenames
            batch_size = dist_sq.shape[0]
            for b in range(batch_size):
                file_idx, target_idx = dataset.index_map[sample_idx]
                file_path = dataset.file_list[file_idx]
                basename = os.path.basename(file_path)
                
                # Determine mode from filename (e.g., track_cv_123.pkl -> cv)
                if "cv" in basename:
                    mode = "CV (匀速)"
                elif "ca" in basename:
                    mode = "CA (匀加速)"
                elif "ct" in basename:
                    mode = "CT (协同转弯)"
                elif "cssw" in basename:
                    mode = "CSSW (复合机动)"
                else:
                    mode = "Unknown"
                    
                mode_sq_errors[mode].extend(dist_sq[b].flatten().tolist())
                sample_idx += 1

    print("\n========== Validation RMSE by Motion Mode ==========")
    for mode, sq_errs in mode_sq_errors.items():
        if len(sq_errs) == 0:
            continue
        rmse = np.sqrt(np.mean(sq_errs))
        p95 = np.percentile(np.sqrt(sq_errs), 95)
        print(f"{mode: <18}: RMSE = {rmse:5.2f}m, P95 = {p95:5.2f}m (Count: {len(sq_errs) // 100} targets)")

if __name__ == "__main__":
    evaluate_by_mode()
