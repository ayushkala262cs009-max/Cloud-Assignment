import torch
import numpy as np
from channel2_grud import TCMImputer

model = TCMImputer(feature_dim=4, hidden_dim=64)
model.load_state_dict(torch.load('tcm_grud_best.pt', map_location='cpu'))
model.eval()

mean = model.scaler_mean.numpy()
std  = model.scaler_std.numpy()

data = np.load('data_corrupted/penetration_20/run_seed_1.npz', allow_pickle=True)
X_gt   = data['X_gt'][:, :, 0:4].astype(np.float32)
X_corr = data['X_corrupted'][:, :, 0:4].astype(np.float32)
X_mask = data['X_mask'][:, :, 0:4].astype(np.float32)
data.close()

# Find a window with actual missing steps in ego columns
missing_counts = (X_mask[:, :, 0] == 0).sum(axis=1)
idx = np.where(missing_counts > 0)[0]

if len(idx) == 0:
    print("No missing steps found in this file.")
else:
    i = idx[0]
    print(f"Using window {i} with {int(missing_counts[i])} missing ego steps\n")

    X_corr_norm = np.where(X_mask[i] == 1.0,
                           (X_corr[i] - mean) / std, 0.0)

    x_t = torch.tensor(X_corr_norm).unsqueeze(0)
    m_t = torch.tensor(X_mask[i]).unsqueeze(0)

    with torch.no_grad():
        out_norm = model(x_t, m_t).squeeze(0).numpy()

    out_phys = out_norm * std + mean

    print(f"{'Step':>4} {'GT_spd':>8} {'Imp_spd':>8} {'mask':>6} {'error':>8}")
    for t in range(30):
        err = abs(X_gt[i, t, 0] - out_phys[t, 0]) if X_mask[i, t, 0] == 0 else 0.0
        flag = " <- missing" if X_mask[i, t, 0] == 0 else ""
        print(f"{t:>4} {X_gt[i,t,0]:>8.3f} {out_phys[t,0]:>8.3f} "
              f"{X_mask[i,t,0]:>6.0f} {err:>8.3f}{flag}")