import numpy as np
data = np.load("data_highd_corrupted/penetration_20/highd_25_base.npz", allow_pickle=True)
mask = data['X_mask']
missing_rate = (1.0 - mask.mean()) * 100
print(f"Overall missingness: {missing_rate:.2f}%")


data = np.load("../data_corrupted/penetration_20/run_seed_1.npz", allow_pickle=True)
mask = data['X_mask']
missing_rate = (1.0 - mask.mean()) * 100
print(f"SUMO penetration_20 missingness: {missing_rate:.2f}%")