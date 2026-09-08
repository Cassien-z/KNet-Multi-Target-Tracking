import os
import sys
import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    data_dir = os.path.join('data', 'reference_data')
    out_dir = os.path.join('data', 'client_sim_mat')
    os.makedirs(out_dir, exist_ok=True)
    
    num_scenes = 2  # Only test on 2 scenarios for fast evaluation
    targets_per_scene = 8
    total_targets = num_scenes * targets_per_scene
    
    # 3 cols per target for tgt, 6 cols per target for full/miss
    tgt_cols = list(range(total_targets * 3))
    full_cols = list(range(total_targets * 6))
    
    print("Loading Excel files using python-calamine directly...")
    from python_calamine import CalamineWorkbook
    
    def read_xlsx(path, cols=None):
        wb = CalamineWorkbook.from_path(path)
        data = wb.get_sheet_by_name(wb.sheet_names[0]).to_python()
        if cols:
            return np.array([[row[i] for i in cols] for row in data])
        else:
            return np.array(data)

    df_own = read_xlsx(os.path.join(data_dir, 'scenario1_ownship.xlsx'))
    df_tgt = read_xlsx(os.path.join(data_dir, 'scenario1_target.xlsx'), tgt_cols)
    df_full1 = read_xlsx(os.path.join(data_dir, 'scenario1_full1.xlsx'), full_cols)
    df_miss1 = read_xlsx(os.path.join(data_dir, 'scenario1_missing1.xlsx'), full_cols)
    df_full2 = read_xlsx(os.path.join(data_dir, 'scenario1_full2.xlsx'), full_cols)
    
    num_frames = df_own.shape[0]
    
    print(f"Frames: {num_frames}, Total Targets: {total_targets}, Generating {num_scenes} scenes...")
    
    # Ownship is shared across all targets
    own_pos = df_own  # [num_frames, 3]
    
    for scene_id in range(num_scenes):
        start_tgt = scene_id * targets_per_scene
        end_tgt = start_tgt + targets_per_scene
        
        # Prepare structures matching script_generate_dataset_unified.py output
        UAV = []
        # UAV 0 (Host)
        UAV0 = {
            'Pose': {'PositionWorld': np.zeros((num_frames, 3)), 'VelocityWorld': np.zeros((num_frames, 3)), 'Euler': np.zeros((num_frames, 3))},
            'Radar': {'Meas': np.zeros((num_frames, targets_per_scene, 3)), 'Mask': np.zeros((num_frames, targets_per_scene), dtype=bool)},
            'EO': {'Meas': np.zeros((num_frames, targets_per_scene, 2)), 'Mask': np.zeros((num_frames, targets_per_scene), dtype=bool)},
            'ESM': {'Meas': np.zeros((num_frames, targets_per_scene, 2)), 'Mask': np.zeros((num_frames, targets_per_scene), dtype=bool)}
        }
        UAV0['Pose']['PositionWorld'] = own_pos
        
        # UAV 1 (Wingman, simulated by Full2)
        UAV1 = {
            'Pose': {'PositionWorld': np.zeros((num_frames, 3)), 'VelocityWorld': np.zeros((num_frames, 3)), 'Euler': np.zeros((num_frames, 3))},
            'FastLink': {'Meas': np.zeros((num_frames, targets_per_scene, 3)), 'Mask': np.zeros((num_frames, targets_per_scene), dtype=bool)}
        }
        
        UAV.append(UAV0)
        UAV.append(UAV1)
        
        Target_GroundTruth = np.zeros((num_frames, targets_per_scene, 6)) # X, Vx, Y, Vy, Z, Vz
        
        for local_i, tgt_idx in enumerate(range(start_tgt, end_tgt)):
            # Target GT
            tgt_xyz = df_tgt[:, tgt_idx*3 : tgt_idx*3+3]
            Target_GroundTruth[:, local_i, 0] = tgt_xyz[:, 0]
            Target_GroundTruth[:, local_i, 2] = tgt_xyz[:, 1]
            Target_GroundTruth[:, local_i, 4] = tgt_xyz[:, 2]
            # (Velocity is zero since it's 1Hz, we'll let pipeline PCHIP interpolation figure out velocity later)
            
            # Full1 -> Radar (Range, Az, El)
            f1_cols = df_full1[:, tgt_idx*6 : tgt_idx*6+6]
            for t in range(num_frames):
                r = f1_cols[t, 0]
                if r != 0 and not np.isnan(r):
                    # Compute true spherical from absolute XYZ
                    dp = f1_cols[t, 3:6] - own_pos[t, :3]
                    r_true = np.linalg.norm(dp)
                    if r_true > 1e-6:
                        az = np.arctan2(dp[1], dp[0])
                        el = np.arcsin(dp[2] / r_true)
                        UAV0['Radar']['Meas'][t, local_i, :] = [az, el, r_true]
                        UAV0['Radar']['Mask'][t, local_i] = True
            
            # Miss1 -> EO (Az, El)
            m1_cols = df_miss1[:, tgt_idx*6 : tgt_idx*6+6]
            for t in range(num_frames):
                # Miss1 has computed XYZ at cols 3:6
                if np.abs(m1_cols[t, 1]) > 1e-6 or np.abs(m1_cols[t, 2]) > 1e-6:
                    dp = m1_cols[t, 3:6] - own_pos[t, :3]
                    r_true = np.linalg.norm(dp)
                    if r_true > 1e-6:
                        az = np.arctan2(dp[1], dp[0])
                        el = np.arcsin(dp[2] / r_true)
                        UAV0['EO']['Meas'][t, local_i, :] = [az, el]
                        UAV0['EO']['Mask'][t, local_i] = True
            
            # Full2 -> FastLink XYZ
            f2_cols = df_full2[:, tgt_idx*6 : tgt_idx*6+6]
            for t in range(num_frames):
                r = f2_cols[t, 0]
                if r != 0 and not np.isnan(r):
                    UAV1['FastLink']['Meas'][t, local_i, :] = f2_cols[t, 3:6]
                    UAV1['FastLink']['Mask'][t, local_i] = True
                    
        Data_Input = np.empty(num_frames, dtype=object)
        
        for t in range(num_frames):
            uav_list = np.empty(2, dtype=object)
            
            # UAV 0
            uav_list[0] = {
                'Pose': {
                    'PositionWorld': own_pos[t],
                    'VelocityWorld': np.zeros(3),
                    'EulerRPY': np.array([1e-5, 1e-5, 1e-5])
                },
                'Radar': {
                    'Meas': UAV0['Radar']['Meas'][t],
                    'Mask': UAV0['Radar']['Mask'][t]
                },
                'EO': {
                    'Meas': UAV0['EO']['Meas'][t],
                    'Mask': UAV0['EO']['Mask'][t]
                },
                'ESM': {
                    'Meas': UAV0['ESM']['Meas'][t],
                    'Mask': UAV0['ESM']['Mask'][t]
                }
            }
            
            # UAV 1
            uav_list[1] = {
                'Pose': {
                    'PositionWorld': np.zeros(3),
                    'VelocityWorld': np.zeros(3),
                    'EulerRPY': np.array([1e-5, 1e-5, 1e-5])
                },
                'FastLink': {
                    'Meas': UAV1['FastLink']['Meas'][t],
                    'Mask': UAV1['FastLink']['Mask'][t]
                }
            }
            
            Data_Input[t] = {'GlobalTime': float(t), 'UAV': uav_list}
            
        out_path = os.path.join(out_dir, f'client_data_{scene_id:04d}.mat')
        scipy.io.savemat(out_path, {
            'Data_Input': Data_Input.tolist(), # scipy.io.savemat prefers list of dicts for cell arrays
            'Data_Label': Target_GroundTruth,
            'T': 1.0 # explicitly mark as 1Hz
        })
        
        if (scene_id + 1) % 10 == 0:
            print(f"Saved {scene_id + 1} / {num_scenes}")

if __name__ == '__main__':
    main()
