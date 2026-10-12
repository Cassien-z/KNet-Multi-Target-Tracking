"""Reference-pilot adapter for the existing registered-PKL/model interface."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dataset_pkl import RegisteredPklDataset


class ReferencePilotDataset(Dataset):
    """Cache small samples and anchor inputs to actual observations, never target truth.

    Fixed-length windows contain no truth padding. The runner separately
    excludes any prefix before measurement-based initialization.
    """
    def __init__(self, data_dir, seq_len=100):
        base = RegisteredPklDataset(str(data_dir), seq_len=seq_len, num_uavs=5, return_extra=True)
        self.samples = []
        self.metadata = []
        for index in range(len(base)):
            file_idx, target_idx = base.index_map[index]
            if target_idx != 0:
                raise ValueError('Pilot files must contain exactly one target')
            file_path = Path(base.file_list[file_idx])
            with file_path.open('rb') as handle:
                payload = pickle.load(handle)
            sample = base[index]
            track = payload['RegisteredTrack'][0]
            positions = np.asarray(track['target_pos'], np.float64)
            velocities = np.asarray(track['target_vel'], np.float64)
            if len(positions) != seq_len or not np.all(np.linalg.norm(positions, axis=1)>1e6):
                raise ValueError(f'Invalid/padded pilot truth: {file_path}')
            anchor = np.asarray(payload['ReferenceOrigin'], np.float64)
            sample['ref_pos'] = torch.from_numpy(anchor.astype(np.float32))
            rel = (positions - anchor).astype(np.float32)
            sample['label_pos_rel'] = torch.from_numpy(rel)
            sample['label_pos_abs'] = torch.from_numpy(positions.astype(np.float32))
            sample['label'][:, [0,2,4]] = torch.from_numpy(rel)
            sample['label'][:, [1,3,5]] = torch.from_numpy(velocities.astype(np.float32))
            sample['label_abs'][:, [0,2,4]] = torch.from_numpy(positions.astype(np.float32))
            sample['label_abs'][:, [1,3,5]] = torch.from_numpy(velocities.astype(np.float32))
            sample['uav_pos'] = torch.from_numpy((np.asarray(payload['UAV_Pos'], np.float64)-anchor).astype(np.float32))
            sample['target_valid_mask'] = torch.ones(seq_len, dtype=torch.float32)
            # No datalink exists in the reference files: zero values and stale ages.
            sample['fast_link'].zero_(); sample['slow_link'].zero_()
            sample['fast_update'].zero_(); sample['slow_update'].zero_()
            sample['fast_age'].fill_(999.); sample['slow_age'].fill_(999.)
            sample['source_target_id'] = payload['SourceMetadata']['source_target_id']
            sample['source_start_frame'] = payload['SourceMetadata']['source_start_frame']
            sample['initialization_frame'] = payload.get('Initialization', {}).get('frame', 0)
            for key, value in sample.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite {key}: {file_path}')
            self.samples.append(sample)
            self.metadata.append({'file': str(file_path), **payload['SourceMetadata']})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]
