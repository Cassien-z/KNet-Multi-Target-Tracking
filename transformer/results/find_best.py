import sys
import os
import torch
import scipy.io
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
    print(f"Using device: {device}")

    # Load model
    model = CoopTrackerModel(d_model=96).to(device)
    weights_path = os.path.join(script_dir, "run_20260516_204818", "best_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    train_data_base = os.path.join(workspace_dir, "train_data")
    folders = ["train_data_ca", "train_data_cssw", "train_data_ct", "train_data_cv"]

    best_rmse = float('inf')
    best_file = None

    for folder in folders:
        folder_path = os.path.join(train_data_base, folder)
        if not os.path.exists(folder_path):
            continue
        
        # Get actual_seq_len from first file
        mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        if not mat_files:
            continue
        first_mat = os.path.join(folder_path, mat_files[0])
        mat_data = scipy.io.loadmat(first_mat)
        actual_seq_len = mat_data['Data_Input'].shape[0] if mat_data['Data_Input'].ndim == 1 else mat_data['Data_Input'].shape[1]
        
        print(f"Folder {folder} seq_len: {actual_seq_len}")
        
        dataset = SingerDataset(
            data_dir=folder_path,
            seq_len=actual_seq_len,
            num_uavs=5,
            return_extra=True
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs = {
                    "radar": batch["radar"].to(device),
                    "eo": batch["eo"].to(device),
                    "esm": batch["esm"].to(device),
                    "radar_mask": batch["radar_mask"].to(device),
                    "eo_mask": batch["eo_mask"].to(device),
                    "esm_mask": batch["esm_mask"].to(device),
                    "ref_pos": batch["ref_pos"].to(device),
                    "rotmat_b2e": batch["rotmat_b2e"].to(device),
                    "uav_pos": batch["uav_pos"].to(device),
                    "fast_link": batch["fast_link"].to(device),
                    "slow_link": batch["slow_link"].to(device),
                    "fast_update": batch["fast_update"].to(device),
                    "slow_update": batch["slow_update"].to(device),
                    "imu": batch["imu"].to(device),
                    "fast_age": batch["fast_age"].to(device),
                    "slow_age": batch["slow_age"].to(device),
                    "esm_mode": batch["esm_mode"].to(device),
                    "dt": batch["dt"].to(device),
                }
                labels = batch["label_abs"].to(device)
                preds = model(**inputs)
                
                # preds is [1, T, 6], labels is [1, T, 6]
                p = preds[0].cpu().numpy()
                l = labels[0].cpu().numpy()
                
                err = np.linalg.norm(p[:, [0, 2, 4]] - l[:, [0, 2, 4]], axis=1).mean()
                
                if err < best_rmse:
                    best_rmse = err
                    best_file = os.path.join(folder, dataset.file_list[batch_idx])
                    print(f"New best: {best_file} with RMSE {best_rmse:.2f} m")
                
                if best_rmse < 8.0:
                    print(f"Found file satisfying condition! {best_file} with RMSE {best_rmse:.2f} m")
                    with open("best_file.txt", "w") as f:
                        f.write(f"{best_file}\n{best_rmse}\n")
                    return

if __name__ == "__main__":
    main()
