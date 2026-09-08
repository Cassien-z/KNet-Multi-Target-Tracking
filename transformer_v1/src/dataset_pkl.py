import os
import glob
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from src.dataset import _parse_registered_pkl

class RegisteredPklDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 100,
        num_uavs: int = 5,
        return_extra: bool = True,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.num_uavs = num_uavs
        self.return_extra = return_extra

        abs_path = os.path.abspath(data_dir)
        print(f"[RegisteredPklDataset] loading from: {abs_path}")

        if not os.path.isdir(data_dir):
            self.file_list = []
        else:
            self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))

        print(f"[RegisteredPklDataset] found {len(self.file_list)} trace files")
        
        # Build index mapping: (file_idx, target_idx)
        self.index_map = []
        for file_idx, file_path in enumerate(self.file_list):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                num_targets = len(data.get("RegisteredTrack", []))
                for target_idx in range(num_targets):
                    self.index_map.append((file_idx, target_idx))
            except Exception as e:
                print(f"[Warning] Failed to read {file_path}: {e}")
                
        print(f"[RegisteredPklDataset] Total valid targets: {len(self.index_map)}")

        # Cache to avoid reloading the same file if consecutive idxs are from the same file
        self._cache_file_idx = -1
        self._cache_samples = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        file_idx, target_idx = self.index_map[idx]
        
        if self._cache_file_idx != file_idx:
            file_path = self.file_list[file_idx]
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            basename = os.path.basename(file_path)
            self._cache_samples = _parse_registered_pkl(data, self.seq_len, self.num_uavs, basename)
            self._cache_file_idx = file_idx
            
        sample = self._cache_samples[target_idx]
        
        # Convert numpy arrays to torch tensors
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = torch.from_numpy(v)
                
        # Ensure dt is a tensor
        if not isinstance(sample.get("dt"), torch.Tensor):
            sample["dt"] = torch.tensor(sample.get("dt", 0.1), dtype=torch.float32)

        if self.return_extra:
            zeros = torch.zeros
            extras = {
                "imu": zeros((self.seq_len, self.num_uavs, 6), dtype=torch.float32),
                "euler": zeros((self.seq_len, self.num_uavs, 3), dtype=torch.float32),
                "quat": zeros((self.seq_len, self.num_uavs, 4), dtype=torch.float32),
                "fast_link": zeros((self.seq_len, self.num_uavs, 3), dtype=torch.float32),
                "slow_link": zeros((self.seq_len, self.num_uavs, 3), dtype=torch.float32),
                "fast_age": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                "slow_age": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                "fast_update": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                "slow_update": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                "esm_mode": zeros((self.seq_len, self.num_uavs), dtype=torch.float32),
                "global_time": zeros((self.seq_len,), dtype=torch.float32),
            }
            for k, v in extras.items():
                if k not in sample:
                    sample[k] = v
            
        return sample
