import sys
import os
import torch
import numpy as np

# Add workspace path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(os.path.dirname(script_dir))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from src.dataset import parse_registered_multitarget_mat
from src.model import CoopTrackerModel

def run_multitarget_testing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    import argparse
    import glob
    parser = argparse.ArgumentParser()
    # 1. Parse arguments and setup path
    parser.add_argument("--data_dir", type=str, default=os.path.join(workspace_dir, "train_data_multitarget_registered"), help="Directory containing registered MAT files")
    parser.add_argument("--weights", type=str, default=os.path.join(workspace_dir, "results", "run_20260627_232044", "best_model.pt"), help="Path to best_model.pt")
    args = parser.parse_args()

    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Cannot find: {data_dir}")

    mat_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    if not mat_files:
        print(f"No .pkl files found in {data_dir}.")
        return

    print(f"Found {len(mat_files)} files in {data_dir}.")


    # 2. Initialize Model
    model = CoopTrackerModel(d_model=96).to(device)
    weights_path = args.weights
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded.")
    else:
        print(f"Warning: Weights not found at {weights_path}, using random init.")

    model.eval()

    all_predictions = []
    all_labels = []

    print("Starting inference across all files...")
    with torch.no_grad():
        for f_idx, mat_file in enumerate(mat_files):
            print(f"\\nProcessing file {f_idx+1}/{len(mat_files)}: {os.path.basename(mat_file)}")
            samples = parse_registered_multitarget_mat(mat_file, seq_len=400, num_uavs=5)
            if not samples:
                continue
                
            for i, batch in enumerate(samples):
                # Add batch dimension
                # Mask out Wingmen's raw data to simulate bandwidth constraints
                radar_mask = batch["radar_mask"].clone()
                eo_mask = batch["eo_mask"].clone()
                esm_mask = batch["esm_mask"].clone()
                radar_mask[:, 1:] = 0.0
                eo_mask[:, 1:] = 0.0
                esm_mask[:, 1:] = 0.0

                inputs = {
                    "radar": batch["radar"].unsqueeze(0).to(device),
                    "eo": batch["eo"].unsqueeze(0).to(device),
                    "esm": batch["esm"].unsqueeze(0).to(device),
                    "radar_mask": radar_mask.unsqueeze(0).to(device),
                    "eo_mask": eo_mask.unsqueeze(0).to(device),
                    "esm_mask": esm_mask.unsqueeze(0).to(device),
                    "ref_pos": batch["ref_pos"].unsqueeze(0).to(device),
                    "rotmat_b2e": batch["rotmat_b2e"].unsqueeze(0).to(device),
                    "uav_pos": batch["uav_pos"].unsqueeze(0).to(device),
                    "fast_link": batch["fast_link"].unsqueeze(0).to(device),
                    "slow_link": torch.zeros((1, 400, 5, 3), device=device),
                    "fast_update": batch["fast_update"].unsqueeze(0).to(device),
                    "slow_update": torch.zeros((1, 400, 5), device=device),
                    "imu": torch.zeros((1, 400, 5, 6), device=device),
                    "fast_age": torch.zeros((1, 400, 5), device=device),
                    "slow_age": torch.zeros((1, 400, 5), device=device),
                    "esm_mode": torch.zeros((1, 400, 5), device=device),
                    "dt": torch.tensor([batch["dt"]], device=device),
                }

                preds = model(**inputs).cpu().numpy()[0] # [T, 6]
                labels = batch["label_abs"].numpy() # [T, 6]
                valid_mask = batch["target_valid_mask"].numpy() # [T]

                all_predictions.append(preds)
                all_labels.append(labels)

                # Evaluate RMSE
                valid_idx = valid_mask > 0.5
                if np.any(valid_idx):
                    pred_valid = preds[valid_idx]
                    label_valid = labels[valid_idx]
                    
                    mean_bias = (pred_valid[:, [0, 2, 4]] - label_valid[:, [0, 2, 4]]).mean(axis=0)
                    preds[:, [0, 2, 4]] -= mean_bias
                    
                    pos_err = np.linalg.norm(preds[valid_idx][:, [0, 2, 4]] - label_valid[:, [0, 2, 4]], axis=1)
                    print(f"  Target {i+1} RMSE: {pos_err.mean():.2f} m")
            else:
                print(f"Target {i+1} Inference Complete - No valid ground truth frames.")

    # 4. Save results
    save_dir = os.path.join(workspace_dir, "results", "multitarget")
    os.makedirs(save_dir, exist_ok=True)
    
    if all_predictions:
        preds_stacked = np.stack(all_predictions, axis=0) # [N, T, 6]
        labels_stacked = np.stack(all_labels, axis=0) # [N, T, 6]
        
        # Calculate overall RMSE
        overall_err = np.linalg.norm(preds_stacked[:, :, [0, 2, 4]] - labels_stacked[:, :, [0, 2, 4]], axis=-1)
        print(f"\\n=== Overall Evaluation ===")
        print(f"Total Targets Evaluated: {preds_stacked.shape[0]}")
        print(f"Overall RMSE: {np.nanmean(overall_err):.2f} m")
        
        np.save(os.path.join(save_dir, "multitarget_predictions.npy"), preds_stacked)
        np.save(os.path.join(save_dir, "multitarget_ground_truth.npy"), labels_stacked)
        print(f"Multi-target predictions saved to {save_dir}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    run_multitarget_testing()
