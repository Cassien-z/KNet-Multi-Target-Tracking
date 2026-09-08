import os
import sys
import glob
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_association.track_association import TrackAssociation

def evaluate_association(data_dir="data/val_data_v8_interference"):
    mat_files = glob.glob(os.path.join(data_dir, "track_*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {data_dir}. Waiting for dataset generation to complete.")
        return

    print(f"Found {len(mat_files)} scenarios. Evaluating Data Association Success Rate...")
    
    total_possible_associations = 0
    total_formed_associations = 0
    correct_associations = 0
    
    for mat_file in mat_files:
        mat = sio.loadmat(mat_file, simplify_cells=True)
        Data_Input = mat['Data_Input']
        
        # Initialize one association module per scenario
        associator = TrackAssociation()
        
        for frame in Data_Input:
            uavs = frame['UAV']
            if not isinstance(uavs, (list, np.ndarray)):
                continue
                
            # UAV 0 has Radar and EO
            if len(uavs) > 0 and 'Radar' in uavs[0]:
                r_m = np.array(uavs[0]['Radar']['Meas'])
                r_mask = np.array(uavs[0]['Radar']['Mask']) > 0.5
                e_m = np.array(uavs[0]['EO']['Meas'])
                e_mask = np.array(uavs[0]['EO']['Mask']) > 0.5
            else:
                continue
                
            # UAV 2 has ESM in the original setup 
            # Check Wingman 2 (index 2)
            if len(uavs) > 2 and 'ESM' in uavs[2]:
                s_m = np.array(uavs[2]['ESM']['Meas'])
                s_mask = np.array(uavs[2]['ESM']['Mask']) > 0.5
            else:
                s_m = np.zeros_like(e_m)
                s_mask = np.zeros_like(e_mask, dtype=bool)

            uav_pos_dict = {}
            uav_euler_dict = {}
            for u_idx, uav in enumerate(uavs):
                if 'Pose' in uav:
                    uav_pos_dict[u_idx] = np.array(uav['Pose']['PositionWorld'])
                    uav_euler_dict[u_idx] = np.array(uav['Pose']['EulerRPY'])

            # To prevent cheating, we shuffle the valid measurements and keep the true ID mapping
            def shuffle_measurements(meas, mask):
                meas = np.atleast_2d(meas)
                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    return np.zeros((0, meas.shape[1])), np.zeros(0, dtype=bool), []
                shuffled_indices = np.random.permutation(valid_indices)
                shuffled_meas = meas[shuffled_indices]
                shuffled_mask = np.ones(len(shuffled_indices), dtype=bool)
                return shuffled_meas, shuffled_mask, shuffled_indices.tolist()

            r_meas_shuffled, r_mask_shuffled, r_true_ids = shuffle_measurements(r_m, r_mask)
            e_meas_shuffled, e_mask_shuffled, e_true_ids = shuffle_measurements(e_m, e_mask)
            s_meas_shuffled, s_mask_shuffled, s_true_ids = shuffle_measurements(s_m, s_mask)
            
            groups = associator.associate_frame(
                r_meas_shuffled, e_meas_shuffled, s_meas_shuffled,
                r_mask_shuffled, e_mask_shuffled, s_mask_shuffled,
                uav_pos_dict, uav_euler_dict
            )
            
            for group in groups:
                true_ids_in_group = []
                if group['Radar_local_id'] is not None:
                    true_ids_in_group.append(r_true_ids[group['Radar_local_id']])
                if group['EO_local_id'] is not None:
                    true_ids_in_group.append(e_true_ids[group['EO_local_id']])
                if group['ESM_local_id'] is not None:
                    true_ids_in_group.append(s_true_ids[group['ESM_local_id']])
                
                # We consider an association formed if it has at least 2 sensors
                if len(true_ids_in_group) > 1:
                    total_formed_associations += 1
                    if all(tid == true_ids_in_group[0] for tid in true_ids_in_group):
                        correct_associations += 1
                
                # Each valid Radar measurement is a possible association base
                if group['Radar_local_id'] is not None:
                    total_possible_associations += 1

    if total_formed_associations == 0:
        print("No valid groups formed. Something might be wrong with the data or association module.")
        return

    success_rate = (correct_associations / total_formed_associations) * 100.0 if total_formed_associations > 0 else 0.0
    recall_rate = (correct_associations / total_possible_associations) * 100.0 if total_possible_associations > 0 else 0.0
    print(f"Total Possible Associations (Radar targets): {total_possible_associations}")
    print(f"Total Formed Multi-Sensor Associations: {total_formed_associations}")
    print(f"Correct Multi-Sensor Associations: {correct_associations}")
    print(f"Data Association Success Rate (Precision): {success_rate:.2f}%")
    print(f"Data Association Recall Rate: {recall_rate:.2f}%")
    
    if success_rate >= 85.0:
        print("[SUCCESS] The metric requirement (>= 85%) is met!")
    else:
        print("[FAILED] The metric requirement (>= 85%) is not met.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_association(sys.argv[1])
    else:
        evaluate_association()
