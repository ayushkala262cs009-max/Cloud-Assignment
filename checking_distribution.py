import numpy as np

def max_consecutive_missing(mask_col):
    """mask_col shape (30,) — returns longest run of 0s"""
    max_run = 0
    current = 0
    for v in mask_col:
        if v == 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run

# Load one test file, compute distribution
import os
data = np.load('data_corrupted/penetration_20/run_seed_17.npz', allow_pickle=True)
mask = data['X_mask'][:, :, 0]  # shape (N, 30)
data.close()

burst_lengths = np.array([max_consecutive_missing(mask[i]) for i in range(mask.shape[0])])

print("Burst length distribution:")
for b in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    count = (burst_lengths == b).sum()
    print(f"  length {b}: {count} windows")
print(f"  length 6+: {(burst_lengths >= 6).sum()} windows")