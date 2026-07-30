import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_predictor import load_lstm

import baseline as baselines

def compute_distances(preds_physical: np.ndarray, y_physical: np.ndarray):
    """Returns the raw distance array (N, 50) for stratified metric calculation."""
    return np.sqrt(np.sum((preds_physical - y_physical) ** 2, axis=2))

def run_highd_e1_evaluation():
    print("=" * 60)
    print("HIGHD CROSS-DATASET E1 EVALUATION (WITH STRATIFICATION)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the frozen LSTM instrument
    print(f"Loading frozen LSTM baseline on {device}...")
    lstm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lstm_baseline.pt")
    lstm_path = os.path.normpath(lstm_path)
    model = load_lstm(lstm_path, device=device)
    
    sx_m = model.scaler_X_mean.cpu().numpy()
    sx_s = model.scaler_X_std.cpu().numpy()
    sy_m = model.scaler_Y_mean.cpu().numpy()
    sy_s = model.scaler_Y_std.cpu().numpy()
    
    # 2. Load Test Recordings, Re-center Y, and Stratify by Regime
    path_highd_20 = os.path.join("data_highd_corrupted", "penetration_20")
    HIGHD_META_DIR = "D:/highD-dataset-v1.0/data"
    
    raw_Y_list, raw_X_corr_list, raw_X_ch1_list, raw_X_fully_list = [], [], [], []
    regime_labels = []
    
    print("Loading HighD test data, re-centering Y, and analyzing regimes...")
    npz_files = [f for f in os.listdir(path_highd_20) if f.endswith(".npz") and ".tmp" not in f]
    
    for fname in sorted(npz_files):
        prefix = fname.split('_')[1]
        
        # Load Meta for Y-centering
        meta_path = os.path.join(HIGHD_META_DIR, f"{prefix}_recordingMeta.csv")
        meta = pd.read_csv(meta_path)
        
        # Parse lane markings
        upper_marks = [float(x) for x in meta.iloc[0]['upperLaneMarkings'].split(';')]
        lower_marks = [float(x) for x in meta.iloc[0]['lowerLaneMarkings'].split(';')]
        upper_mid = (upper_marks[0] + upper_marks[-1]) / 2.0
        lower_mid = (lower_marks[0] + lower_marks[-1]) / 2.0
        
        # Load Data
        data = np.load(os.path.join(path_highd_20, fname), allow_pickle=True)
        X_gt = data['X_gt'].copy()
        Y = data['Y'].copy()
        X_corr = data['X_corrupted'].copy()
        X_ch1 = data['X_ch1_imputed'].copy()
        X_fully = data['X_fully_imputed'].copy()
        
        # Apply Y-Centering per window based on initial Y position
        N_windows = X_corr.shape[0]
        for i in range(N_windows):
            y_val = X_gt[i, 0, 3]
            
            # Decide direction by proximity to lane midpoints
            offset = upper_mid if abs(y_val - upper_mid) < abs(y_val - lower_mid) else lower_mid
            
            Y[i, :, 1] -= offset
            X_corr[i, :, 3] -= offset
            X_ch1[i, :, 3] -= offset
            X_fully[i, :, 3] -= offset
            
        # Determine Traffic Regime based on this recording's mean speed
        mean_speed = np.nanmean(X_corr[:, :, 0])
        regime = "HighD (Free-Flow)" if mean_speed >= 20.0 else "HighD (Congested)"
        regime_labels.extend([regime] * N_windows)
        
        raw_Y_list.append(Y)
        raw_X_corr_list.append(X_corr)
        raw_X_ch1_list.append(X_ch1)
        raw_X_fully_list.append(X_fully)
            
    test_Y = np.concatenate(raw_Y_list, axis=0).astype(np.float32)
    test_X_corr = np.concatenate(raw_X_corr_list, axis=0).astype(np.float32)
    regime_array = np.array(regime_labels)
    
    print(f"Loaded {test_Y.shape[0]} windows.")
    print(f" -> Free-Flow (>20 m/s): {np.sum(regime_array == 'HighD (Free-Flow)')}")
    print(f" -> Congested (<=20 m/s): {np.sum(regime_array == 'HighD (Congested)')}")
    
    # 3. Generate the baselines
    print("\nGenerating baseline imputations in memory (Spline may take a few minutes)...")
    datasets = {
        'Baseline 1: Zero-fill': baselines.zero_fill(test_X_corr),
        'Baseline 2: Forward-fill': baselines.forward_fill(test_X_corr),
        'Baseline 3: Mean-fill (Local)': baselines.mean_fill(test_X_corr),
        'Baseline 4: Linear Interpolation': baselines.linear_interp(test_X_corr),
        'Baseline 5: Spline Interpolation': baselines.spline_interp(test_X_corr),
        'Condition B: Channel 1 Only': np.concatenate(raw_X_ch1_list, axis=0).astype(np.float32),
        'Condition C: Dual-Channel (Proposed)': np.concatenate(raw_X_fully_list, axis=0).astype(np.float32)
    }
    
    highd_results = []
    
    print("\nRunning Zero-Shot Model Evaluation...")
    with torch.no_grad():
        for name, X_physical in datasets.items():
            
            X_physical_clean = np.nan_to_num(X_physical, nan=0.0)
            X_norm = (X_physical_clean - sx_m) / sx_s
            
            N = X_norm.shape[0]
            batch_size = 1024
            preds_physical_list = []
            
            for start_idx in range(0, N, batch_size):
                batch_X = torch.from_numpy(X_norm[start_idx : start_idx+batch_size]).to(device)
                preds_norm = model(batch_X)
                batch_preds_phys = (preds_norm.cpu().numpy() * sy_s) + sy_m
                preds_physical_list.append(batch_preds_phys)
                
            preds_physical = np.concatenate(preds_physical_list, axis=0)
            
            # Get raw distances to compute stratified metrics
            distances = compute_distances(preds_physical, test_Y)
            sample_ade = np.mean(distances, axis=1)
            sample_fde = distances[:, -1]
            
            # Stratify by Regime
            for regime in ["HighD (Free-Flow)", "HighD (Congested)"]:
                mask = (regime_array == regime)
                if np.sum(mask) > 0:
                    highd_results.append({
                        'Condition': name,
                        'Regime': regime,
                        'ADE (m)': round(np.mean(sample_ade[mask]), 3),
                        'ADE Std': round(np.std(sample_ade[mask]), 3),
                        'FDE (m)': round(np.mean(sample_fde[mask]), 3),
                        'FDE Std': round(np.std(sample_fde[mask]), 3)
                    })
            print(f"[{name}] Evaluated.")
            
    # 4. Save HighD Table
    df_highd = pd.DataFrame(highd_results)
    df_highd.to_csv("Table6_HighD_E1_Stratified.csv", index=False)
    print("\nSaved Table6_HighD_E1_Stratified.csv")
    
    # 5. Merge with SUMO data for 3-Bar Chart
    name_map = {
        'Baseline 1: Zero-fill (No Imputation)': 'Baseline 1: Zero-fill',
        'Baseline 1: Zero-fill': 'Baseline 1: Zero-fill',
        'Baseline 2: Forward-fill': 'Baseline 2: Forward-fill',
        'Baseline 3: Mean-fill (Local)': 'Baseline 3: Mean-fill (Local)',
        'Baseline 4: Linear Interpolation': 'Baseline 4: Linear Interpolation',
        'Baseline 5: Spline Interpolation': 'Baseline 5: Spline Interpolation',
        'Condition B: Channel 1 Only': 'Condition B: Channel 1 Only',
        'Condition C: Dual-Channel (Proposed)': 'Condition C: Dual-Channel (Proposed)'
    }
    
    try:
        sumo_table_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Table1_E1_Metrics.csv")
        sumo_table_path = os.path.normpath(sumo_table_path)
        
        df_sumo = pd.read_csv(sumo_table_path)
        df_sumo['Condition'] = df_sumo['Condition'].map(name_map)
        df_sumo['Regime'] = 'SUMO (Free-Flow)'
        
        # Standardize columns and combine
        df_sumo = df_sumo[['Condition', 'Regime', 'ADE (m)', 'FDE (m)']]
        df_highd_subset = df_highd[['Condition', 'Regime', 'ADE (m)', 'FDE (m)']]
        df_all = pd.concat([df_sumo, df_highd_subset], ignore_index=True)
        
        conditions = [
            'Baseline 1: Zero-fill', 'Baseline 2: Forward-fill', 'Baseline 3: Mean-fill (Local)',
            'Baseline 4: Linear Interpolation', 'Baseline 5: Spline Interpolation',
            'Condition B: Channel 1 Only', 'Condition C: Dual-Channel (Proposed)'
        ]
        
        x = np.arange(len(conditions))
        width = 0.25
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        
        colors = {
            'SUMO (Free-Flow)': '#4c72b0',
            'HighD (Free-Flow)': '#55a868',
            'HighD (Congested)': '#c44e52'
        }
        
        offsets = [-width, 0, width]
        labels = [c.split(':')[0] if ':' in c else c for c in conditions]
        
        for i, regime in enumerate(['SUMO (Free-Flow)', 'HighD (Free-Flow)', 'HighD (Congested)']):
            subset = df_all[df_all['Regime'] == regime].set_index('Condition')
            # Ensure order matches
            ades = [subset.loc[c, 'ADE (m)'] if c in subset.index else 0 for c in conditions]
            fdes = [subset.loc[c, 'FDE (m)'] if c in subset.index else 0 for c in conditions]
            
            ax1.bar(x + offsets[i], ades, width, label=regime, color=colors[regime], edgecolor='black')
            ax2.bar(x + offsets[i], fdes, width, label=regime, color=colors[regime], edgecolor='black')

        # Formatting AX1 (ADE)
        ax1.set_ylabel('Average Displacement Error (Meters)', fontsize=12)
        ax1.set_title('Cross-Dataset ADE by Traffic Regime', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax1.set_axisbelow(True)
        ax1.legend(fontsize=11)
        
        # Formatting AX2 (FDE)
        ax2.set_ylabel('Final Displacement Error (Meters)', fontsize=12)
        ax2.set_title('Cross-Dataset FDE by Traffic Regime', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig("Fig6_HighD_Stratified_BarChart.png", dpi=300)
        print("Saved Fig6_HighD_Stratified_BarChart.png (3-Bar Comparison)")
        
    except FileNotFoundError:
        print("\nWARNING: 'Table1_E1_Metrics.csv' not found. Ensure it is in the parent SUMO directory.")

if __name__ == "__main__":
    run_highd_e1_evaluation()