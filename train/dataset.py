import json
import torch
from torch.utils.data import Dataset
import numpy as np

ACTION_SPACE_SIZE = 32768
MOVE_COUNT = 64 * 64 * 6

def flip_square(sq_idx: int) -> int:
    return (sq_idx // 8) * 8 + (7 - (sq_idx % 8))

FLIP_ACTION_LUT = np.arange(ACTION_SPACE_SIZE, dtype=np.int32)
for action_idx in range(MOVE_COUNT):
    from_sq = action_idx // (64 * 6)
    remainder = action_idx % (64 * 6)
    to_sq = remainder // 6
    promo = remainder % 6

    flipped_from = flip_square(from_sq)
    flipped_to = flip_square(to_sq)
    FLIP_ACTION_LUT[action_idx] = flipped_from * 64 * 6 + flipped_to * 6 + promo

class AugmentDataset(Dataset):
    def __init__(self, primary_jsonl, replay_jsonl=None, replay_ratio=0.0, augment_flip=True):
        import os
        self.samples = []

        def load_raw(path, max_samples=None):
            items = []
            if not path or not os.path.isfile(path):
                return items
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            items.append(json.loads(line))
                            if max_samples and len(items) >= max_samples:
                                break
            except Exception as e:
                print(f"[Dataset] Warning loading {path}: {e}")
            return items

        primary_items = load_raw(primary_jsonl)
        print(f"[Dataset] Loaded {len(primary_items)} primary samples from {primary_jsonl}")

        all_items = list(primary_items)
        if replay_jsonl and replay_ratio > 0.0:
            target_replay_count = int(len(primary_items) * replay_ratio / (1.0 - replay_ratio)) if primary_items else 1000
            replay_items = load_raw(replay_jsonl, max_samples=target_replay_count)
            all_items.extend(replay_items)
            print(f"[Dataset] Added {len(replay_items)} replay samples from {replay_jsonl}")

        # Store compact lightweight memory (spatial uint8/float32 + sparse policy)
        for sample in all_items:
            sp = np.array(sample['spatial'], dtype=np.float32).reshape(64, 8, 8)
            gl = np.array(sample['global'], dtype=np.float32)
            val = float(sample['value'])
            mask_list = np.array([m for m in sample['mask'] if m < ACTION_SPACE_SIZE], dtype=np.int32)
            policy_list = [(int(act), float(p)) for act, p in sample['policy'] if act < ACTION_SPACE_SIZE]

            self.samples.append((sp, gl, mask_list, policy_list, val))

            if augment_flip:
                flipped_sp = np.flip(sp, axis=2).copy()
                flipped_gl = gl.copy()
                # Swap castling rights: [7, 8] My K/Q, [9, 10] Opp K/Q
                flipped_gl[7], flipped_gl[8] = flipped_gl[8], flipped_gl[7]
                flipped_gl[9], flipped_gl[10] = flipped_gl[10], flipped_gl[9]

                flipped_mask = FLIP_ACTION_LUT[mask_list]
                flipped_policy = [(int(FLIP_ACTION_LUT[act]), p) for act, p in policy_list]

                self.samples.append((flipped_sp, flipped_gl, flipped_mask, flipped_policy, val))

        print(f"[Dataset] Prepared {len(self.samples)} samples in compact memory (Flip factor: 2x).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sp, gl, mask_list, policy_list, val = self.samples[idx]

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        if len(mask_list) > 0:
            mask[mask_list] = True

        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for act, prob in policy_list:
            policy[act] = prob

        return {
            'spatial': torch.from_numpy(sp),
            'global': torch.from_numpy(gl),
            'mask': torch.from_numpy(mask),
            'policy': torch.from_numpy(policy),
            'value': torch.tensor([val], dtype=torch.float32),
        }
