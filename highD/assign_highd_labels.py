import os
import shutil
import numpy as np
import pandas as pd

# Paths to your local setup
EXTRACTED_DIR = "data_highd_extracted"
HIGHD_META_DIR = "D:/highD-dataset-v1.0/data"
OUTPUT_DIR = "data_highd"  # Will contain penetration_10, 20, 30

PENETRATION_RATES = [10, 20, 30]

def process_recording_labels(npz_file: str, meta_file: str, pen_rate: int, output_path: str):
    """
    Applies per-recording, deterministic CAV/HDV labels to the integer-ID dataset.
    Filters the dataset so the Ego vehicle is strictly a CAV.
    """
    # 1. Load the pristine extracted arrays
    data = np.load(npz_file)
    X_gt = data['X_gt']
    Y = data['Y']
    ego_ids = data['ego_id']
    leader_ids = data['leader_id']
    follower_ids = data['follower_id']
    
    # 2. Load the metadata to determine classes and total count
    meta = pd.read_csv(meta_file)
    total_vehicles = len(meta)
    
    # Strictly isolate eligible CAVs to Cars only
    cars = meta[meta['class'] == 'Car']['id'].values
    
    # Target number of CAVs is a percentage of ALL traffic (including trucks)
    target_cav_count = int((pen_rate / 100.0) * total_vehicles)
    
    # EMPIRICAL CAP CHECK
    if target_cav_count > len(cars):
        print(f" [!] WARNING: CAPPING TRIGGERED on {os.path.basename(meta_file)} at {pen_rate}%! "
              f"(Target: {target_cav_count}, Available Cars: {len(cars)})")
    
    target_cav_count = min(target_cav_count, len(cars)) 
    
    # 3. Random draw for THIS recording and THIS penetration rate
    recording_prefix = os.path.basename(meta_file).split('_')[0]
    
    # Uniquely and deterministically seed this specific combination
    seed = int(recording_prefix) * 100 + pen_rate
    
    # BUG 2 FIX: Use isolated Generator instance instead of global np.random state
    rng = np.random.default_rng(seed)
    cav_pool = set(rng.choice(cars, target_cav_count, replace=False))
    
    # 4. Filter the dataset: Ego MUST be a CAV for our V2X trajectory prediction problem
    ego_is_cav = np.isin(ego_ids, list(cav_pool))
    
    X_gt_filtered = X_gt[ego_is_cav]
    Y_filtered = Y[ego_is_cav]
    leader_ids_filtered = leader_ids[ego_is_cav]
    follower_ids_filtered = follower_ids[ego_is_cav]
    
    surviving_windows = np.sum(ego_is_cav)
    
    if surviving_windows == 0:
        print(f" -> {recording_prefix} at {pen_rate}%: 0 windows survived (no Ego CAVs). Skipping.")
        return 0

    # 5. Map the neighbour IDs to String Categorical Arrays (The Fallback Rule)
    def map_to_string_meta(id_matrix):
        # Default everything to 'HDV'
        meta_matrix = np.full(id_matrix.shape, 'HDV', dtype='<U4')
        
        # Overwrite 0 strictly to 'none'
        meta_matrix[id_matrix == 0] = 'none'
        
        # Overwrite selected IDs to 'CAV'
        is_cav_mask = np.isin(id_matrix, list(cav_pool))
        meta_matrix[is_cav_mask] = 'CAV'
        
        return meta_matrix

    leader_meta = map_to_string_meta(leader_ids_filtered)
    follower_meta = map_to_string_meta(follower_ids_filtered)
    
    # 6. Save the perfectly formatted output ready for SFM corruption
    temp_path = output_path + ".tmp"
    actual_temp_path = temp_path + ".npz"
    
    try:
        np.savez_compressed(
            temp_path,
            X_gt=X_gt_filtered,
            Y=Y_filtered,
            leader_meta=leader_meta,
            follower_meta=follower_meta
            # We explicitly drop the integer IDs here, mimicking SUMO Step 3 exactly
        )
        shutil.move(actual_temp_path, output_path)
        return surviving_windows
    except Exception as e:
        print(f" -> ERROR saving modifications to {output_path}: {e}")
        if os.path.exists(actual_temp_path):
            try:
                os.remove(actual_temp_path)
            except OSError:
                pass
        return 0

if __name__ == "__main__":
    print("=" * 60)
    print("ASSIGNING HIGH-D CAV/HDV METADATA (BY PENETRATION RATE)")
    print("=" * 60)
    
    total_windows_all_rates = {}
    
    for pen_rate in PENETRATION_RATES:
        rate_dir = os.path.join(OUTPUT_DIR, f"penetration_{pen_rate}")
        os.makedirs(rate_dir, exist_ok=True)
        
        print(f"\n--- Processing {pen_rate}% Penetration Rate ---")
        
        npz_files = [f for f in os.listdir(EXTRACTED_DIR) if f.endswith(".npz")]
        rate_windows = 0
        
        for npz_filename in sorted(npz_files):
            prefix = npz_filename.split('_')[1] # highd_01_extracted.npz -> '01'
            meta_filename = f"{prefix}_tracksMeta.csv"
            
            npz_path = os.path.join(EXTRACTED_DIR, npz_filename)
            meta_path = os.path.join(HIGHD_META_DIR, meta_filename)
            output_path = os.path.join(rate_dir, f"highd_{prefix}_base.npz")
            
            # Restartability
            if os.path.exists(output_path):
                print(f" -> Skipping {prefix}: Already processed for {pen_rate}%.")
                # Need to add up already processed windows for final counts
                with np.load(output_path, allow_pickle=True) as d:
                    rate_windows += d['X_gt'].shape[0]
                continue
                
            if os.path.exists(meta_path):
                yielded = process_recording_labels(npz_path, meta_path, pen_rate, output_path)
                rate_windows += yielded
            else:
                print(f" -> WARNING: Meta file {meta_filename} not found.")
                
        total_windows_all_rates[pen_rate] = rate_windows
        
    print("\n" + "=" * 60)
    print("CAV/HDV LABELING COMPLETE")
    for rate, count in total_windows_all_rates.items():
        print(f"Penetration {rate}%: Generated {count} labeled windows.")
    print("=" * 60)