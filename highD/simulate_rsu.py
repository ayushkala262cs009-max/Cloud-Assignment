import os
import numpy as np

# Ensure deterministic measurement noise profiles across runs
np.random.seed(42)

def simulate_rsu_camera(
    X_gt: np.ndarray, 
    leader_meta: list, 
    position_noise_std: float = 0.5, 
    speed_noise_std: float = 0.3, 
    detection_rate: float = 0.95
) -> np.ndarray:
    """
    Simulates RSU camera observations of leading vehicles (both HDVs and CAVs).
    Treats both vehicle types identically at the sensor level, applying physically 
    calibrated Gaussian noise and random dropouts.
    
    Args:
        X_gt (np.ndarray): Shape (30, 7), uncorrupted ground-truth features.
                           Column 4: gap_to_leader, Column 5: leader_speed.
        leader_meta (list): 30 strings representing the leader type ('none', 'HDV', 'CAV') 
                            at each timestep.
        position_noise_std (float): Gaussian noise standard deviation for gap measurement (meters).
        speed_noise_std (float): Gaussian noise standard deviation for speed measurement (m/s).
        detection_rate (float): Probability of successful camera detection.
        
    Returns:
        np.ndarray: Shape (30, 2), containing [noisy_gap, noisy_leader_speed].
    """
    # Initialize the output array with NaNs (defaults to missing)
    rsu_obs = np.full((30, 2), np.nan, dtype=np.float32)
    
    for t in range(30):
        l_type = leader_meta[t]
        
        # Decode byte strings if passed from serialized npz structures
        if isinstance(l_type, bytes):
            l_type = l_type.decode('utf-8')
            
        # Case 1: No leader exists. Keep as NaN (no vehicle present to observe).
        if l_type == 'none':
            continue
            
        # Case 2: A leader is physically present (either HDV or CAV)
        if l_type in ('HDV', 'CAV'):
            # Check if the RSU camera successfully detects the vehicle (95% chance)
            if np.random.random() < detection_rate:
                # Retrieve ground-truth relative gap and absolute speed
                gt_gap = X_gt[t, 4]
                gt_speed = X_gt[t, 5]
                
                # Double-check that ground-truth values are valid before applying noise
                if not np.isnan(gt_gap) and not np.isnan(gt_speed):
                    # Apply additive zero-mean Gaussian measurement noise
                    rsu_obs[t, 0] = gt_gap + np.random.normal(0, position_noise_std)
                    rsu_obs[t, 1] = gt_speed + np.random.normal(0, speed_noise_std)
                else:
                    # Defensive trigger for Step 2 extraction audits
                    print(f"\nWARNING: gt NaN at t={t} despite leader_type={l_type}")
                    
            # (If random check fails, it remains NaN representing a "missed detection")
                    
    return rsu_obs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_corrupted")
    args = parser.parse_args()
    
    CORRUPTED_DIR = args.data_dir
    
    assert "highd" in CORRUPTED_DIR.lower() or CORRUPTED_DIR == "data_corrupted", \
        f"Unexpected directory '{CORRUPTED_DIR}' — verify this is intentional before proceeding."

    SCENARIOS = ["penetration_10", "penetration_20", "penetration_30"]
    
    print("=" * 60)
    print("RUNNING RSU CAMERA SIMULATION AND INTEGRATION")
    print("=" * 60)
    
    processed_count = 0
    skipped_count = 0
    
    for scenario in SCENARIOS:
        scenario_path = os.path.join(CORRUPTED_DIR, scenario)
        
        if not os.path.exists(scenario_path):
            print(f"Skipping empty path: {scenario_path}")
            continue
            
        print(f"\nProcessing Scenario Folder: {scenario}")
        npz_files = [f for f in os.listdir(scenario_path) if f.endswith(".npz")]
        
        for file_name in sorted(npz_files):
            file_path = os.path.join(scenario_path, file_name)
            
            # Load the archive into memory to safely close the context before overwriting
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    X_gt = data['X_gt']
                    Y = data['Y']
                    X_corrupted = data['X_corrupted']
                    X_mask = data['X_mask']
                    leader_meta = data['leader_meta']
                    follower_meta = data['follower_meta']
                    
                    # Restartability: If rsu_obs already exists, skip processing
                    if 'rsu_obs' in data:
                        print(f" -> Skipping: {file_name} already contains integrated RSU camera observations.")
                        skipped_count += 1
                        continue
            except Exception as e:
                print(f" -> ERROR loading {file_name}: {e}")
                continue
                
            n_samples = X_gt.shape[0]
            batch_rsu_obs = []
            
            # Process each individual window slice sequentially
            for i in range(n_samples):
                window_rsu_obs = simulate_rsu_camera(X_gt[i], leader_meta[i])
                batch_rsu_obs.append(window_rsu_obs)
                
            rsu_obs_array = np.array(batch_rsu_obs, dtype=np.float32)
            
            # Write to a temporary file first, then perform an atomic replacement
            temp_path = file_path.replace(".npz", ".tmp.npz")
            try:
                np.savez_compressed(
                    temp_path,
                    X_gt=X_gt,
                    Y=Y,
                    X_corrupted=X_corrupted,
                    X_mask=X_mask,
                    leader_meta=leader_meta,
                    follower_meta=follower_meta,
                    rsu_obs=rsu_obs_array
                )
                # Atomically swap the temporary file over the target path
                os.replace(temp_path, file_path)
                print(f" -> Updated {file_name} with integrated RSU observations shape: {rsu_obs_array.shape}")
                processed_count += 1
            except OSError as e:
                print(f" -> ERROR saving modifications back to {file_name}: {e}")
                # Clean up the dangling temporary file if it failed to replace the original
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                
    print("\n" + "=" * 60)
    print("INTEGRATION PIPELINE RUN COMPLETED")
    print(f"Successfully processed (newly updated): {processed_count} files.")
    print(f"Successfully skipped (already present): {skipped_count} files.")
    print(f"Total verified files in corrupted folder: {processed_count + skipped_count}")
    print("=" * 60)