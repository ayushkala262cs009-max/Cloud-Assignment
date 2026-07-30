import numpy as np
data = np.load("data_highd_corrupted/penetration_20/highd_25_base.npz", allow_pickle=True)
X_gt = data['X_gt']
print(f"HighD ego_x: mean={X_gt[:,:,2].mean():.2f}, std={X_gt[:,:,2].std():.2f}")
print(f"SUMO scaler: mean=406.8, std=120.0")
z_score_x = (X_gt[:,:,2].mean() - 406.8) / 120.0
print(f"Resulting z-score offset: {z_score_x:.2f}")