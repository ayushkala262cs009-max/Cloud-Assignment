import torch
import numpy as np
from channel2_grud import TCMImputer

# Load model
model = TCMImputer(feature_dim=4, hidden_dim=64)
model.load_state_dict(torch.load('tcm_grud_best.pt', map_location='cpu'))
model.eval()

print(f"Scaler mean: {model.scaler_mean}")
print(f"Scaler std:  {model.scaler_std}")

# Load one real sample
data = np.load('data_corrupted/penetration_20/run_seed_1.npz', allow_pickle=True)
X_gt   = data['X_gt'][0, :, 0:4].astype(np.float32)
X_corr = data['X_corrupted'][0, :, 0:4].astype(np.float32)
X_mask = data['X_mask'][0, :, 0:4].astype(np.float32)
data.close()

mean = model.scaler_mean.numpy()
std  = model.scaler_std.numpy()

X_corr_norm = np.where(X_mask == 1.0, (X_corr - mean) / std, 0.0)

x_t = torch.tensor(X_corr_norm).unsqueeze(0)
m_t = torch.tensor(X_mask).unsqueeze(0)

with torch.no_grad():
    out_norm = model(x_t, m_t).squeeze(0).numpy()

out_phys = out_norm * std + mean

print("\nSample verification (first 5 steps):")
print(f"{'Step':>4} {'GT_speed':>10} {'Imp_speed':>10} {'mask':>6}")
for t in range(5):
    print(f"{t:>4} {X_gt[t,0]:>10.3f} {out_phys[t,0]:>10.3f} {X_mask[t,0]:>6.0f}")