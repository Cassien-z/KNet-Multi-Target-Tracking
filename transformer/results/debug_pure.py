import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from dataset import SingerDataset
from model import CoopTrackerModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoopTrackerModel(d_model=96).to(device)
    weights_path = os.path.join(script_dir, "run_20260516_204818", "best_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    folder_path = os.path.join(workspace_dir, "train_data", "train_data_ca")
    
    dataset = SingerDataset(
        data_dir=folder_path,
        seq_len=100,
        num_uavs=5,
        return_extra=True
    )
    
    # find track_data_ca_0342.mat
    idx = -1
    for i, f in enumerate(dataset.file_list):
        if "0342.mat" in f:
            idx = i
            break
            
    batch = dataset[idx]
    
    inputs = {
        "radar": batch["radar"].unsqueeze(0).to(device),
        "eo": batch["eo"].unsqueeze(0).to(device),
        "esm": batch["esm"].unsqueeze(0).to(device),
        "radar_mask": batch["radar_mask"].unsqueeze(0).to(device),
        "eo_mask": batch["eo_mask"].unsqueeze(0).to(device),
        "esm_mask": batch["esm_mask"].unsqueeze(0).to(device),
        "ref_pos": batch["ref_pos"].unsqueeze(0).to(device),
        "rotmat_b2e": batch["rotmat_b2e"].unsqueeze(0).to(device),
        "uav_pos": batch["uav_pos"].unsqueeze(0).to(device),
        "fast_link": batch["fast_link"].unsqueeze(0).to(device),
        "slow_link": batch["slow_link"].unsqueeze(0).to(device),
        "fast_update": batch["fast_update"].unsqueeze(0).to(device),
        "slow_update": batch["slow_update"].unsqueeze(0).to(device),
        "imu": batch["imu"].unsqueeze(0).to(device),
        "fast_age": batch["fast_age"].unsqueeze(0).to(device),
        "slow_age": batch["slow_age"].unsqueeze(0).to(device),
        "esm_mode": batch["esm_mode"].unsqueeze(0).to(device),
        "dt": batch["dt"].unsqueeze(0).to(device),
    }
    labels = batch["label_abs"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(**inputs)
        p = preds[0].cpu().numpy()
        l = labels[0].cpu().numpy()
        
        errors = np.linalg.norm(p[:, [0, 2, 4]] - l[:, [0, 2, 4]], axis=1)
        print(f"Overall RMSE: {errors.mean():.2f}")
        
        mean_diff = (p[:, [0, 2, 4]] - l[:, [0, 2, 4]]).mean(axis=0)
        l_shifted = l.copy()
        l_shifted[:, [0, 2, 4]] += mean_diff
        errors_shifted = np.linalg.norm(p[:, [0, 2, 4]] - l_shifted[:, [0, 2, 4]], axis=1)
        print(f"RMSE after shifting truth: {errors_shifted.mean():.2f}")

if __name__ == "__main__":
    main()
