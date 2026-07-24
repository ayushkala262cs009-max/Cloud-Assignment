import numpy as np

data = np.load('data_corrupted/penetration_20/run_seed_1.npz', allow_pickle=True)
rsu = data['rsu_obs']      # shape (N, 30, 2)
X_gt = data['X_gt']        # shape (N, 30, 7)

print(f"rsu_obs shape: {rsu.shape}")

# NaN rate - should be ~99% for penetration_20 (most leaders are HDV but
# camera still observes them, so NaN only from missed detections + no leader)
nan_gap   = np.isnan(rsu[:,:,0]).mean() * 100
nan_speed = np.isnan(rsu[:,:,1]).mean() * 100
print(f"NaN in rsu gap:   {nan_gap:.1f}%")
print(f"NaN in rsu speed: {nan_speed:.1f}%")

# Noise check - difference between rsu and ground truth should be small
# Only compare where both are observed
mask = ~np.isnan(rsu[:,:,0]) & ~np.isnan(X_gt[:,:,4])
if mask.sum() > 0:
    gap_error = rsu[:,:,0][mask] - X_gt[:,:,4][mask]
    speed_error = rsu[:,:,1][mask] - X_gt[:,:,5][mask]
    print(f"Gap error   mean={gap_error.mean():.3f}m  std={gap_error.std():.3f}m  (expected ~0, ~0.5)")
    print(f"Speed error mean={speed_error.mean():.3f}m/s std={speed_error.std():.3f}m/s (expected ~0, ~0.3)")

data.close()