import numpy as np
data = np.load("data_highd_corrupted/penetration_20/highd_25_base.npz", allow_pickle=True)
X_gt = data['X_gt']

print(f"Speed percentiles: 5th={np.percentile(X_gt[:,:,0], 5):.2f}, "
      f"50th={np.percentile(X_gt[:,:,0], 50):.2f}, "
      f"95th={np.percentile(X_gt[:,:,0], 95):.2f}")
print(f"Speed histogram (rough):")
for lo, hi in [(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35)]:
    frac = ((X_gt[:,:,0] >= lo) & (X_gt[:,:,0] < hi)).mean() * 100
    print(f"  {lo}-{hi} m/s: {frac:.1f}%")