import numpy as np

data = np.load('data_corrupted/penetration_20/run_seed_1.npz', allow_pickle=True)
X_gt = data['X_gt']
X_corrupted = data['X_corrupted']
X_ch1 = data['X_ch1_imputed']
X_mask = data['X_mask']

print("=== NaN CHECK ===")
print(f"X_corrupted col4 NaN: {np.isnan(X_corrupted[:,:,4]).mean()*100:.1f}%")
print(f"X_corrupted col5 NaN: {np.isnan(X_corrupted[:,:,5]).mean()*100:.1f}%")
print(f"X_ch1 col4 NaN:       {np.isnan(X_ch1[:,:,4]).mean()*100:.1f}%")
print(f"X_ch1 col5 NaN:       {np.isnan(X_ch1[:,:,5]).mean()*100:.1f}%")
print(f"X_ch1 col6 NaN:       {np.isnan(X_ch1[:,:,6]).mean()*100:.1f}%")

print("\n=== IMPUTATION ACCURACY (where SFM was masked) ===")
sfm_mask_col5 = X_mask[:,:,5] == 0
gt_speed = X_gt[:,:,5][sfm_mask_col5]
ch1_speed = X_ch1[:,:,5][sfm_mask_col5]
valid = ~np.isnan(ch1_speed)
if valid.sum() > 0:
    mae = np.abs(gt_speed[valid] - ch1_speed[valid]).mean()
    print(f"Leader speed MAE (KF vs ground truth): {mae:.3f} m/s")
    print(f"Coverage (imputed / total SFM cells):  {valid.sum()}/{sfm_mask_col5.sum()}")

print("\n=== COLUMN 6 UNCHANGED CHECK ===")
col6_same = np.array_equal(
    np.nan_to_num(X_corrupted[:,:,6]),
    np.nan_to_num(X_ch1[:,:,6])
)
print(f"Column 6 identical to X_corrupted: {col6_same}")

data.close()