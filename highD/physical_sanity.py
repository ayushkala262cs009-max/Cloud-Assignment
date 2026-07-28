import numpy as np

def run_sanity_check(npz_path: str):
    data = np.load(npz_path)
    X = data['X_gt']
    
    print("\n" + "="*50)
    print("1. RANGES & NA RATES (The Sanity Check)")
    print("="*50)
    print(f"Shape: {X.shape}")
    print(f"Speed range:  {X[:,:,0].min():.2f} to {X[:,:,0].max():.2f} m/s")
    print(f"Accel range:  {X[:,:,1].min():.2f} to {X[:,:,1].max():.2f} m/s²")
    print(f"X pos range:  {X[:,:,2].min():.2f} to {X[:,:,2].max():.2f} m")
    print(f"Y pos range:  {X[:,:,3].min():.2f} to {X[:,:,3].max():.2f} m")
    print(f"Gap NaN rate: {np.isnan(X[:,:,4]).mean()*100:.1f}%")
    print(f"Leader spd NaN rate: {np.isnan(X[:,:,5]).mean()*100:.1f}%")
    print(f"Follower spd NaN rate: {np.isnan(X[:,:,6]).mean()*100:.1f}%")

    print("\n" + "="*50)
    print("2. POSITION-VELOCITY DRIFT (The Consistency Check)")
    print("="*50)
    # Estimate speed purely from position displacement (Euclidean distance over dt=0.1s)
    dx = np.diff(X[:,:,2], axis=1)
    dy = np.diff(X[:,:,3], axis=1)
    v_derived = np.hypot(dx, dy) / 0.1
    
    # Compare against our independently splined speed column
    v_splined = X[:, :-1, 0]
    
    abs_err = np.abs(v_derived - v_splined)
    
    print(f"Mean Absolute Error:   {np.mean(abs_err):.4f} m/s")
    print(f"Median Absolute Error: {np.median(abs_err):.4f} m/s")
    print(f"99th Percentile Error: {np.percentile(abs_err, 99):.4f} m/s")
    print(f"Max Absolute Error:    {np.max(abs_err):.4f} m/s")
    
    # Let's see what that drift is as a relative percentage of actual speed
    # (Excluding near-zero speeds to avoid divide-by-zero explosions)
    valid_speeds = v_splined > 1.0 
    if np.any(valid_speeds):
        rel_err = (abs_err[valid_speeds] / v_splined[valid_speeds]) * 100
        print(f"\nMean Relative Error:   {np.mean(rel_err):.2f}%")
        print(f"99th Pct Relative Err: {np.percentile(rel_err, 99):.2f}%")

if __name__ == "__main__":
    run_sanity_check("data_highd_extracted/highd_25_extracted.npz")