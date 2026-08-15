import json
import torch
from torch.utils.data import Dataset
import numpy as np

ACTION_SPACE_SIZE = 32768

class AugmentDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        spatial = np.array(sample['spatial'], dtype=np.float32).reshape(64, 8, 8)
        global_feat = np.array(sample['global'], dtype=np.float32)
        
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for m_idx in sample['mask']:
            if m_idx < ACTION_SPACE_SIZE:
                mask[m_idx] = True
                
        policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for act_idx, prob in sample['policy']:
            if act_idx < ACTION_SPACE_SIZE:
                policy_target[act_idx] = prob
                
        value_target = np.array([sample['value']], dtype=np.float32)
        
        return {
            'spatial': torch.from_numpy(spatial),
            'global': torch.from_numpy(global_feat),
            'mask': torch.from_numpy(mask),
            'policy': torch.from_numpy(policy_target),
            'value': torch.from_numpy(value_target),
        }

