import numpy as np

# HighD
data_h = np.load("data_highd_corrupted/penetration_20/highd_25_base.npz", allow_pickle=True)
mask_h = data_h['X_mask']

print("=== HighD penetration_20 ===")
print(f"TCM cols (0-3) missingness: {(1 - mask_h[:,:,0:4].mean())*100:.2f}%")
print(f"SFM cols (4-6) missingness: {(1 - mask_h[:,:,4:7].mean())*100:.2f}%")

# SUMO
data_s = np.load("../data_corrupted/penetration_20/run_seed_1.npz", allow_pickle=True)
mask_s = data_s['X_mask']

print("\n=== SUMO penetration_20 ===")
print(f"TCM cols (0-3) missingness: {(1 - mask_s[:,:,0:4].mean())*100:.2f}%")
print(f"SFM cols (4-6) missingness: {(1 - mask_s[:,:,4:7].mean())*100:.2f}%")