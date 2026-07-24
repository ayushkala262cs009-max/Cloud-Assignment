import os
import shutil
import numpy as np
from channel1_kf import sfm_impute_channel1

CORRUPTED_DIR = "data_corrupted"
SCENARIOS = ["penetration_10", "penetration_20", "penetration_30"]

processed_count = 0
skipped_count = 0
total_cells_patched = 0

print("=" * 60)
print("RUNNING CHANNEL 1 DATASET INTEGRATION PIPELINE")
print("=" * 60)

for scenario in SCENARIOS:
    scenario_path = os.path.join(CORRUPTED_DIR, scenario)
    if not os.path.exists(scenario_path):
        print(f"Skipping empty scenario path: {scenario_path}")
        continue
        
    print(f"\nProcessing Scenario Folder: {scenario}")
    # Guard against processing active/orphaned temp files by ignoring anything with '.tmp'
    npz_files = [f for f in os.listdir(scenario_path) if f.endswith(".npz") and ".tmp" not in f]
    
    for file_name in sorted(npz_files):
        file_path = os.path.join(scenario_path, file_name)
        
        # Check if the file has already been integrated
        try:
            with np.load(file_path, allow_pickle=True) as data:
                if 'X_ch1_imputed' in data:
                    print(f" -> Skipping: {file_name} already contains Channel 1 imputed states.")
                    skipped_count += 1
                    continue
        except Exception:
            # File might be locked or empty, attempt to process anyway
            pass

        try:
            with np.load(file_path, allow_pickle=True) as data:
                X_gt = data['X_gt']
                Y = data['Y']
                X_corrupted = data['X_corrupted']
                X_mask = data['X_mask']
                rsu_obs = data['rsu_obs']
                leader_meta = data['leader_meta']
                follower_meta = data['follower_meta']
        except Exception as e:
            print(f" -> ERROR loading {file_name}: {e}")
            continue
            
        n_samples = X_gt.shape[0]
        
        # Initialize the new array as a copy of the corrupted input features
        X_ch1_imputed = X_corrupted.copy()
        
        for i in range(n_samples):
            # Pass pristine local ego coordinates for precise relative-to-absolute reconstruction
            imputed_gaps, imputed_speeds = sfm_impute_channel1(rsu_obs[i], X_gt[i, :, 0:4])
            
            for t in range(30):
                l_type = leader_meta[i, t]
                if isinstance(l_type, bytes):
                    l_type = l_type.decode('utf-8')
                    
                # Case: A physical leader exists (either an HDV or a CAV)
                if l_type in ('HDV', 'CAV'):
                    # Overwrite ONLY if V2X speed telemetry is missing (value is 0.0 in mask)
                    # This preserves perfect, lossless V2X broadcasts when links are active
                    if X_mask[i, t, 5] == 0.0:
                        X_ch1_imputed[i, t, 4] = imputed_gaps[t]
                        X_ch1_imputed[i, t, 5] = imputed_speeds[t]
                        total_cells_patched += 2
                        
        temp_path = file_path + ".tmp"
        actual_temp_path = temp_path + ".npz"  # NumPy silently appends .npz to the temp file
        
        try:
            np.savez_compressed(
                temp_path,
                X_gt=X_gt,
                Y=Y,
                X_corrupted=X_corrupted,
                X_mask=X_mask,
                rsu_obs=rsu_obs,
                leader_meta=leader_meta,
                follower_meta=follower_meta,
                X_ch1_imputed=X_ch1_imputed
            )
            # Perform atomic replacement using shutil.move to handle Windows permissions cleanly
            shutil.move(actual_temp_path, file_path)
            print(f" -> Updated {file_name} with Channel 1 imputed shape: {X_ch1_imputed.shape}")
            processed_count += 1
        except Exception as e:
            print(f" -> ERROR saving modifications to {file_name}: {e}")
            if os.path.exists(actual_temp_path):
                try:
                    os.remove(actual_temp_path)
                except OSError:
                    pass

print("\n" + "=" * 60)
print("CHANNEL 1 INTEGRATION SEQUENCE COMPLETED")
print(f"Successfully processed (newly updated): {processed_count} files.")
print(f"Successfully skipped (already present): {skipped_count} files.")
print(f"Total physical files verified on disk: {processed_count + skipped_count}")
print(f"Total dataset metric fields imputed:   {total_cells_patched} cells.")
print("=" * 60)