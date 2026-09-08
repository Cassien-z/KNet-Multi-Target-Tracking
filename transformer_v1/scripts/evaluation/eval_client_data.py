import os
import sys
import numpy as np
import scipy.io
import pickle
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.ecef_utils import geodetic_to_ecef
# In pipeline_core, there's run_registration_pipeline
import subprocess
from scripts.evaluation.baseline import build_imm_tracker

def evaluate_client_data():
    client_mat_dir = os.path.join('data', 'client_sim_mat')
    client_pkl_dir = os.path.join('data', 'client_sim_pkl')
    os.makedirs(client_pkl_dir, exist_ok=True)
    
    mat_files = [f for f in os.listdir(client_mat_dir) if f.endswith('.mat')]
    if not mat_files:
        print("No .mat files found in data/client_sim_mat. Run script_import_client_data.py first.")
        return
        
    print(f"Found {len(mat_files)} client simulation scenarios. Running Registration Pipeline...")
    
    # 1. Register and Upsample using our existing pipeline
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pipeline_script = os.path.join(workspace_dir, "scripts", "pipelines", "run_register_pipeline.py")
    print("Calling scripts/pipelines/run_register_pipeline.py...")
    try:
        subprocess.run(
            [sys.executable, pipeline_script, "--in_dir", "data/client_sim_mat", "--out_dir", "data/client_sim_pkl", "--overwrite"],
            check=True,
            cwd=workspace_dir,
        )
        print("Registration complete.")
    except Exception as e:
        print("Registration failed:", e)
        return
                
    # 2. Run Transformer vs IMM-JPDA
    import torch
    from src.dataset import SingerDataset
    from scripts.evaluation.eval_compare import load_model
    from scripts.evaluation.baseline import load_dataset_mat
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # You might need to change the path to your actual best model
    model_path = os.path.join('results', 'runs_40off(v7)', 'run_20260728_103728', 'best_model.pt')
    
    try:
        model = load_model(model_path, device)
        model.eval()
        print("Loaded trained Transformer.")
    except Exception as e:
        print(f"Warning: Failed to load Transformer model: {e}")
        return
        
    all_transformer_rmses = []
    
    def compute_rmse(preds, labels, mask):
        # preds: [seq_len, 6] (X,Vx,Y,Vy,Z,Vz)
        # mask: [seq_len]
        label_pos = labels[:, [0, 2, 4]]
        valid_idx = np.where((mask > 0) & (np.linalg.norm(label_pos, axis=1) > 1e-3))[0]
        if len(valid_idx) == 0:
            return None
        pos_err = preds[valid_idx][:, [0, 2, 4]] - label_pos[valid_idx]
        return np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
    
    from src.dataset import parse_registered_multitarget_mat
    
    registered_files = [f for f in os.listdir(client_pkl_dir) if f.endswith('.pkl')]
    if not registered_files:
        print("No registered pickle files found!")
        return
        
    for pkl_file in registered_files:
        pkl_path = os.path.join(client_pkl_dir, pkl_file)
        print(f"Evaluating {pkl_file}...")
        try:
            samples = parse_registered_multitarget_mat(pkl_path, seq_len=500, num_uavs=5)
            for i, sample in enumerate(samples):
                seq_len = sample["radar"].shape[0]
                inputs = {
                    "radar": sample["radar"].unsqueeze(0).to(device),
                    "eo": sample["eo"].unsqueeze(0).to(device),
                    "esm": sample["esm"].unsqueeze(0).to(device),
                    "radar_mask": sample["radar_mask"].unsqueeze(0).to(device),
                    "eo_mask": sample["eo_mask"].unsqueeze(0).to(device),
                    "esm_mask": sample["esm_mask"].unsqueeze(0).to(device),
                    "ref_pos": sample["ref_pos"].unsqueeze(0).to(device),
                    "rotmat_b2e": sample["rotmat_b2e"].unsqueeze(0).to(device),
                    "uav_pos": sample["uav_pos"].unsqueeze(0).to(device),
                    "fast_link": sample["fast_link"].unsqueeze(0).to(device),
                    "slow_link": torch.zeros((1, seq_len, 5, 3), device=device),
                    "fast_update": sample["fast_update"].unsqueeze(0).to(device),
                    "slow_update": torch.zeros((1, seq_len, 5), device=device),
                    "imu": torch.zeros((1, seq_len, 5, 6), device=device),
                    "fast_age": torch.zeros((1, seq_len, 5), device=device),
                    "slow_age": torch.zeros((1, seq_len, 5), device=device),
                    "esm_mode": torch.zeros((1, seq_len, 5), device=device),
                    "dt": torch.tensor([sample["dt"]], device=device) if isinstance(sample["dt"], float) else torch.tensor(sample["dt"], device=device).view(-1),
                }
                
                with torch.no_grad():
                    preds = model(**inputs).cpu().numpy()[0]
                    
                labels = sample["label_abs"].numpy()
                
                print(f"Target {i} - preds[0]: {preds[0]}")
                print(f"Target {i} - labels[0]: {labels[0]}")
                
                mask = samples[i]['target_valid_mask']
                rmse = compute_rmse(preds, labels, mask)
                if rmse is not None:
                    print(f"Target {i} - RMSE: {rmse:.2f}m")
                    all_transformer_rmses.append(rmse)
                else:
                    print(f"Target {i} - RMSE: None (no valid frames)")
            print(f"  -> Processed {len(samples)} targets, current avg RMSE: {np.mean(all_transformer_rmses):.2f}m")
        except Exception as e:
            print(f"Transformer skipped {pkl_file}:", e)
            
    if all_transformer_rmses:
        print(f"Final Average Transformer RMSE: {np.mean(all_transformer_rmses):.2f} m")
        
if __name__ == '__main__':
    evaluate_client_data()
