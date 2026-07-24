import os
import numpy as np

# Ensure exact reproducibility of random packet loss profiles across experiments
np.random.seed(42)

# Gilbert-Elliott channel state parameters
# Mean burst duration of failure = 1 / P_EXIT_BAD = 2.5 steps (250 milliseconds)
P_ENTER_BAD = 0.05  # Probability of transitioning from GOOD to BAD state
P_EXIT_BAD = 0.40   # Probability of transitioning from BAD to GOOD state

# Directory structures matching our Step 3 penetration sweeps
INPUT_DIR = "data"
OUTPUT_DIR = "data_corrupted"

SCENARIOS = [
    "penetration_10",
    "penetration_20",
    "penetration_30"
]

def generate_tcm_mask(length: int = 30, p_enter: float = 0.05, p_exit: float = 0.40) -> np.ndarray:
    """
    Generates a single 1D temporal mask where 1.0 indicates an observed V2X frame
    and 0.0 indicates a dropped packet, utilizing a two-state Markov Chain.
    """
    mask = np.ones(length, dtype=np.float32)
    state = 0  # 0 = GOOD state (Packet Received), 1 = BAD state (Packet Dropped)
    
    for t in range(length):
        if state == 0:
            if np.random.random() < p_enter:
                state = 1
        else:  # state == 1
            if np.random.random() < p_exit:
                state = 0
                
        if state == 1:
            mask[t] = 0.0
            
    return mask

def inject_sfm(X_gt: np.ndarray, leader_meta: np.ndarray, follower_meta: np.ndarray) -> tuple:
    """
    [Intern's Parallel Module]
    Applies Structural Feature Missingness (SFM) based on neighbor V2X connectivity.
    
    Rules:
      - Column 4 (gap_to_leader): Retained when leader is HDV (tracked via radar). 
                                  Masked ONLY if no leader exists ('none').
      - Column 5 (leader_speed): Masked if leader is 'none' or 'HDV' (lacks V2X).
      - Column 6 (follower_speed): Masked if follower is 'none' or 'HDV' (lacks V2X).
    """
    n_samples, n_steps, n_features = X_gt.shape
    X_sfm = X_gt.copy().astype(np.float32)
    sfm_mask = np.ones((n_samples, n_steps, n_features), dtype=np.float32)
    
    for idx in range(n_samples):
        for t in range(n_steps):
            l_type = leader_meta[idx, t]
            f_type = follower_meta[idx, t]
            
            # Decode byte strings if loaded from npz
            if isinstance(l_type, bytes):
                l_type = l_type.decode('utf-8')
            if isinstance(f_type, bytes):
                f_type = f_type.decode('utf-8')
                
            # Column 4: gap_to_leader (Only missing if no leader exists)
            if l_type == 'none':
                X_sfm[idx, t, 4] = np.nan
                sfm_mask[idx, t, 4] = 0.0
                
            # Column 5: leader_speed (Requires V2X; missing if leader is none or HDV)
            if l_type == 'none' or l_type == 'HDV':
                X_sfm[idx, t, 5] = np.nan
                sfm_mask[idx, t, 5] = 0.0
                
            # Column 6: follower_speed (Requires V2X; missing if follower is none or HDV)
            if f_type == 'none' or f_type == 'HDV':
                X_sfm[idx, t, 6] = np.nan
                sfm_mask[idx, t, 6] = 0.0
                
    return X_sfm, sfm_mask

def inject_tcm_to_dataset(X_input: np.ndarray, p_enter: float = 0.05, p_exit: float = 0.40) -> tuple:
    """
    [Our Parallel Module]
    Applies unified Temporal Cluster Missingness (TCM) to the ego-state columns.
    If a V2X packet is lost at timestep t, columns 0, 1, 2, and 3 are lost simultaneously.
    """
    n_samples, n_steps, n_features = X_input.shape
    X_tcm = X_input.copy().astype(np.float32)
    tcm_mask = np.ones((n_samples, n_steps, n_features), dtype=np.float32)
    
    for idx in range(n_samples):
        # Generate exactly ONE packet loss profile per window
        mask_1d = generate_tcm_mask(length=n_steps, p_enter=p_enter, p_exit=p_exit)
        
        for t in range(n_steps):
            if mask_1d[t] == 0.0:
                # Force columns 0 to 3 to be missing simultaneously
                X_tcm[idx, t, 0:4] = np.nan
                tcm_mask[idx, t, 0:4] = 0.0
                
    return X_tcm, tcm_mask

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SFM & TCM CORRUPTION PIPELINE")
    print("=" * 60)
    
    processed_count = 0
    total_corrupted_cells = 0
    total_total_cells = 0
    
    for scenario in SCENARIOS:
        src_scenario_dir = os.path.join(INPUT_DIR, scenario)
        dest_scenario_dir = os.path.join(OUTPUT_DIR, scenario)
        
        # Safe directory initialization
        os.makedirs(dest_scenario_dir, exist_ok=True)
        
        if not os.path.exists(src_scenario_dir):
            print(f"Skipping empty scenario path: {src_scenario_dir}")
            continue
            
        print(f"\nProcessing scenario: {scenario}")
        npz_files = [f for f in os.listdir(src_scenario_dir) if f.endswith(".npz")]
        
        for file_name in sorted(npz_files):
            src_file_path = os.path.join(src_scenario_dir, file_name)
            dest_file_path = os.path.join(dest_scenario_dir, file_name)
            
            # Restartable check: skip if corrupted file is already successfully processed
            if os.path.exists(dest_file_path):
                try:
                    with np.load(dest_file_path, allow_pickle=True) as data:
                        _ = data['X_corrupted']
                    print(f" -> Skipping: {dest_file_path} already exists and is valid.")
                    continue
                except Exception:
                    print(f" -> Corrupt target detected. Deleting and re-running: {dest_file_path}")
                    try:
                        os.remove(dest_file_path)
                    except OSError:
                        pass

            # Load original ground truth files and cleanly extract the metadata structure
            try:
                with np.load(src_file_path, allow_pickle=True) as data:
                    X = data['X']
                    Y = data['Y']
                    meta = data['metadata']
                    leader_meta = np.array([m['leader'] for m in meta])
                    follower_meta = np.array([m['follower'] for m in meta])
            except Exception as e:
                print(f" -> Error reading {file_name}: {e}")
                continue
                
            # 1. Apply Structural Feature Missingness first
            X_sfm, sfm_mask = inject_sfm(X, leader_meta, follower_meta)
            
            # 2. Apply Temporal Cluster Missingness on top of the SFM-corrupted state
            X_corrupted, tcm_mask = inject_tcm_to_dataset(X_sfm, P_ENTER_BAD, P_EXIT_BAD)
            
            # 3. Calculate combined mask (1.0 = Observed, 0.0 = Missing)
            X_mask = sfm_mask * tcm_mask
            
            # Telemetry calculations
            missing_rate = (1.0 - (np.sum(X_mask) / X_mask.size)) * 100.0
            total_corrupted_cells += np.sum(X_mask == 0.0)
            total_total_cells += X_mask.size
            
            try:
                np.savez_compressed(
                    dest_file_path,
                    X_gt=X,                     # Keep ground truth features fully safe
                    Y=Y,                        # Keep original labels intact
                    X_corrupted=X_corrupted,    # The model input features
                    X_mask=X_mask,              # The combined 0/1 observation mask
                    leader_meta=leader_meta,    # Ground truth leader tracking
                    follower_meta=follower_meta # Ground truth follower tracking
                )
                print(f" -> Corrupted {file_name} | Combined Missingness: {missing_rate:.2f}%")
                processed_count += 1
            except OSError as e:
                print(f" -> ERROR saving modifications to {file_name}: {e}")
                
    print("\n" + "=" * 60)
    print("CORRUPTION SEQUENCE COMPLETE")
    if total_total_cells > 0:
        overall_missing_rate = (total_corrupted_cells / total_total_cells) * 100.0
        print(f"Total Files Modified:                   {processed_count}")
        print(f"Global Dataset Missingness Rate:        {overall_missing_rate:.2f}%")
    print("=" * 60)