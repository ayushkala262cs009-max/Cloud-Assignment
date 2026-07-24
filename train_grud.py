from channel2_grud import TCMImputer, TrajectoryTCMDataset, train_tcm_grud
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np

# Load full dataset
print("Loading dataset...")
full_dataset = TrajectoryTCMDataset(data_dir="data_corrupted")
print(f"Total samples loaded: {len(full_dataset)}")

# 80/20 split by sample index (No leakage!)
n_train = int(0.8 * len(full_dataset))
n_val = len(full_dataset) - n_train

# Fix the seed for the split to ensure reproducibility across runs
train_set, val_set = random_split(
    full_dataset, 
    [n_train, n_val], 
    generator=torch.Generator().manual_seed(42)
)

# ==============================================================================
# --- NEW: COMPUTE SCALERS FROM TRAINING SET ONLY ---
# ==============================================================================
print("Computing training normalization scalers...")
# Extract all ground truth arrays from the training subset
train_gt_arrays = np.stack([full_dataset.samples[i][0] for i in train_set.indices])

scaler_mean = torch.tensor(train_gt_arrays.mean(axis=(0, 1)), dtype=torch.float32)
scaler_std = torch.tensor(train_gt_arrays.std(axis=(0, 1)), dtype=torch.float32)
# Prevent division by zero if standard deviation is absolutely flat
scaler_std = torch.clamp(scaler_std, min=1e-6)

print(f"Scaler Mean: {scaler_mean.tolist()}")
print(f"Scaler Std:  {scaler_std.tolist()}")

# ==============================================================================
# --- NEW: NORMALISE ENTIRE DATASET IN MEMORY ---
# ==============================================================================
mean_np = scaler_mean.numpy()
std_np = scaler_std.numpy()

for i in range(len(full_dataset.samples)):
    x_gt, x_corr, mask = full_dataset.samples[i]
    
    # Normalise ground truth (so loss is computed fairly across all features)
    norm_gt = (x_gt - mean_np) / std_np
    
    # Normalise corrupted data, forcing masked entries to remain cleanly 0.0
    # This ensures the GRU-D gracefully decays from its baseline during drops.
    norm_corr = np.where(mask == 1.0, (x_corr - mean_np) / std_np, 0.0)
    
    # Update the dataset item in-place
    full_dataset.samples[i] = (norm_gt.astype(np.float32), norm_corr.astype(np.float32), mask)

# Build data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)

# Initialise and train model, securely passing the computed scalers to the architecture
model = TCMImputer(feature_dim=4, hidden_dim=64, scaler_mean=scaler_mean, scaler_std=scaler_std)
train_tcm_grud(model, train_loader, val_loader, epochs=50, lr=1e-3, device="cpu")