import os
import shutil
import numpy as np
import torch
from channel2_grud import TCMImputer

CORRUPTED_DIR = "data_corrupted"
SCENARIOS = ["penetration_10", "penetration_20", "penetration_30"]
MODEL_PATH = "tcm_grud_best.pt"

def run_integration_pipeline():
    print("=" * 60)
    print("RUNNING STEP 9: CHANNEL 2 INFERENCE & DUAL INTEGRATION")
    print("=" * 60)

    # 1. Load the trained GRU-D model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading TCMImputer from {MODEL_PATH} on {device}...")
    
    model = TCMImputer(feature_dim=4, hidden_dim=64)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # CRITICAL: Put model in inference mode
    
    processed_count = 0
    skipped_count = 0

    for scenario in SCENARIOS:
        scenario_path = os.path.join(CORRUPTED_DIR, scenario)
        if not os.path.exists(scenario_path):
            continue
            
        print(f"\nProcessing Scenario Folder: {scenario}")
        npz_files = [f for f in os.listdir(scenario_path) if f.endswith(".npz") and ".tmp" not in f]
        
        for file_name in sorted(npz_files):
            file_path = os.path.join(scenario_path, file_name)
            
            # Load all existing arrays cleanly into memory
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    # Restartable check
                    if 'X_fully_imputed' in data:
                        print(f" -> Skipping: {file_name} already contains X_fully_imputed.")
                        skipped_count += 1
                        continue
                        
                    # Extract all data into a dictionary for safe saving later
                    data_dict = {key: data[key] for key in data.files}
            except Exception as e:
                print(f" -> ERROR loading {file_name}: {e}")
                continue

            # Ensure Channel 1 data exists before running Channel 2
            if 'X_ch1_imputed' not in data_dict:
                print(f" -> WARNING: X_ch1_imputed missing in {file_name}. Run Step 7 first.")
                continue

            X_ch1 = data_dict['X_ch1_imputed'].astype(np.float32)
            X_mask = data_dict['X_mask'].astype(np.float32)
            
            # ---------------------------------------------------------
            # PHASE 1: Input Extraction & Normalization
            # ---------------------------------------------------------
            # Extract ego columns (0:4)
            raw_ego = X_ch1[:, :, 0:4].copy()
            mask_ego = X_mask[:, :, 0:4].copy()
            
            # Sanitize NaNs to 0.0 before tensor conversion
            raw_ego = np.nan_to_num(raw_ego, nan=0.0)
            
            raw_ego_t = torch.from_numpy(raw_ego).to(device)
            mask_ego_t = torch.from_numpy(mask_ego).to(device)
            
            with torch.no_grad():
                # Apply training scalers to shift into z-score latent space
                norm_ego_t = (raw_ego_t - model.scaler_mean) / model.scaler_std
                
                # CRITICAL: Re-apply the mask to force missing entries strictly to 0.0
                norm_ego_t = torch.where(mask_ego_t == 1.0, norm_ego_t, 
                                         torch.zeros_like(norm_ego_t))
                
                # ---------------------------------------------------------
                # PHASE 2: Model Inference
                # ---------------------------------------------------------
                pred_norm_t = model(norm_ego_t, mask_ego_t)
                
                # Denormalize predictions back to physical units (m/s, m/s^2, m)
                physical_preds_t = model.denormalise(pred_norm_t)
                physical_preds = physical_preds_t.cpu().numpy()

            # ---------------------------------------------------------
            # PHASE 3: Dataset Integration & Verification
            # ---------------------------------------------------------
            X_fully_imputed = X_ch1.copy()
            
            # Overwrite the ego state columns with the GRU-D's smoothed, continuous predictions
            X_fully_imputed[:, :, 0:4] = physical_preds
            
            # VERIFICATION: Channel 2 must have filled columns 0-3 completely
            nan_count_ego = np.isnan(X_fully_imputed[:, :, 0:4]).sum()
            if nan_count_ego > 0:
                raise ValueError(f"Channel 2 failed for {file_name}: {nan_count_ego} NaNs in ego columns 0-3!")

            # Report residual NaNs in neighbour columns honestly
            nan_col4 = np.isnan(X_fully_imputed[:, :, 4]).sum()
            nan_col5 = np.isnan(X_fully_imputed[:, :, 5]).sum()
            nan_col6 = np.isnan(X_fully_imputed[:, :, 6]).sum()
            
            # Update the dataset dictionary
            data_dict['X_fully_imputed'] = X_fully_imputed
            
            # Atomic Save Sequence
            temp_path = file_path + ".tmp"
            actual_temp_path = temp_path + ".npz"
            try:
                np.savez_compressed(temp_path, **data_dict)
                shutil.move(actual_temp_path, file_path)
                print(f" -> Processed {file_name} | Ego NaNs: {nan_count_ego} | "
                      f"Col4 NaN: {nan_col4} | Col5 NaN: {nan_col5} | Col6 NaN: {nan_col6}")
                processed_count += 1
            except Exception as e:
                print(f" -> ERROR saving modifications to {file_name}: {e}")
                if os.path.exists(actual_temp_path):
                    os.remove(actual_temp_path)

    print("\n" + "=" * 60)
    print("STEP 9 INTEGRATION COMPLETE")
    print(f"Newly processed files: {processed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print("=" * 60)

if __name__ == "__main__":
    run_integration_pipeline()