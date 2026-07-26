import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_predictor import load_lstm
import baseline as baselines

def get_max_burst_lengths(mask_3d: np.ndarray) -> np.ndarray:
    """
    Computes the maximum length of contiguous missing steps (0.0) 
    in the ego coordinates for each window.
    mask_3d shape: (N, 30, F). We only evaluate column 0 (speed).
    """
    N, T = mask_3d.shape[0], mask_3d.shape[1]
    max_bursts = np.zeros(N, dtype=int)
    
    # Extract just the 1D temporal mask for ego state
    ego_masks = mask_3d[:, :, 0]
    
    for i in range(N):
        # 1.0 = observed, 0.0 = missing. We invert to count contiguous 1s representing dropouts.
        missing_mask = (ego_masks[i] == 0.0).astype(int)
        
        count = 0
        max_count = 0
        for val in missing_mask:
            if val == 1:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        max_bursts[i] = max_count
        
    return max_bursts

def assign_burst_bin(burst_len: int) -> str:
    """Categorizes the contiguous burst length into the 4 analytical bins."""
    if burst_len == 0:
        return "0 steps"
    elif 1 <= burst_len <= 2:
        return "1-2 steps"
    elif 3 <= burst_len <= 5:
        return "3-5 steps"
    else:
        return "6+ steps"

def run_e4_evaluation():
    print("=" * 60)
    print("STEP 11 - EXPERIMENT 4: ROBUSTNESS TO TCM SEVERITY")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the frozen LSTM instrument
    print(f"Loading frozen LSTM baseline on {device}...")
    model = load_lstm("lstm_baseline.pt", device=device)
    
    sx_m = model.scaler_X_mean.cpu().numpy()
    sx_s = model.scaler_X_std.cpu().numpy()
    sy_m = model.scaler_Y_mean.cpu().numpy()
    sy_s = model.scaler_Y_std.cpu().numpy()
    
    # 2. Load Test Data for 20% Penetration (Seeds 17-20)
    test_seeds = {f"run_seed_{i}.npz" for i in range(17, 21)}
    path_20 = os.path.join("data_corrupted", "penetration_20")
    
    raw_Y_list, raw_X_corr_list, raw_X_mask_list = [], [], []
    raw_X_ch1_list, raw_X_fully_list = [], []
    
    print("Loading test data (penetration_20, seeds 17-20)...")
    for fname in sorted(os.listdir(path_20)):
        if fname in test_seeds:
            data = np.load(os.path.join(path_20, fname), allow_pickle=True)
            raw_Y_list.append(data['Y'])
            raw_X_corr_list.append(data['X_corrupted'])
            raw_X_mask_list.append(data['X_mask'])
            raw_X_ch1_list.append(data['X_ch1_imputed'])
            raw_X_fully_list.append(data['X_fully_imputed'])
            
    test_Y = np.concatenate(raw_Y_list, axis=0).astype(np.float32)
    test_X_corr = np.concatenate(raw_X_corr_list, axis=0).astype(np.float32)
    test_X_mask = np.concatenate(raw_X_mask_list, axis=0).astype(np.float32)
    
    # 3. Compute Burst Bins for all N windows
    burst_lengths = get_max_burst_lengths(test_X_mask)
    burst_bins_array = np.array([assign_burst_bin(bl) for bl in burst_lengths])
    
    bins_order = ["0 steps", "1-2 steps", "3-5 steps", "6+ steps"]
    
    # Console audit of bin counts
    print("\nBurst Bin Distribution:")
    for b_name in bins_order:
        count = np.sum(burst_bins_array == b_name)
        print(f"  {b_name}: {count} windows")
    
    # 4. Prepare only the 4 required conditions
    print("\nGenerating baselines in memory (Zero-fill and Spline)...")
    datasets = {
        'Baseline 1: Zero-fill': baselines.zero_fill(test_X_corr),
        'Baseline 5: Spline Interpolation': baselines.spline_interp(test_X_corr),
        'Condition B: Channel 1 Only': np.concatenate(raw_X_ch1_list, axis=0).astype(np.float32),
        'Condition C: Dual-Channel (Proposed)': np.concatenate(raw_X_fully_list, axis=0).astype(np.float32)
    }
    
    results = []
    
    # 5. Evaluation Loop
    print("\nRunning Model Evaluation...")
    with torch.no_grad():
        for condition_name, X_physical in datasets.items():
            
            # Universal physical-space zero-fill for unresolved NaNs
            X_physical_clean = np.nan_to_num(X_physical, nan=0.0)
            X_norm = (X_physical_clean - sx_m) / sx_s
            
            N = X_norm.shape[0]
            batch_size = 1024
            preds_physical_list = []
            
            # Batch Prediction
            for start_idx in range(0, N, batch_size):
                batch_X = torch.from_numpy(X_norm[start_idx : start_idx+batch_size]).to(device)
                preds_norm = model(batch_X)
                batch_preds_phys = (preds_norm.cpu().numpy() * sy_s) + sy_m
                preds_physical_list.append(batch_preds_phys)
                
            preds_physical = np.concatenate(preds_physical_list, axis=0)
            
            # Compute distances: Euclidean distance at every timestep (N, 50)
            distances = np.sqrt(np.sum((preds_physical - test_Y) ** 2, axis=2))
            
            # Collapse to mean ADE per sample (N,)
            sample_ade = np.mean(distances, axis=1)
            
            # 6. Aggregate by Burst Bin
            for b_name in bins_order:
                bin_mask = (burst_bins_array == b_name)
                n_windows = np.sum(bin_mask)
                
                if n_windows > 0:
                    bin_ades = sample_ade[bin_mask]
                    mean_ade = np.mean(bin_ades)
                    std_ade = np.std(bin_ades)
                else:
                    mean_ade, std_ade = 0.0, 0.0
                    
                results.append({
                    'Condition': condition_name,
                    'Burst_Bin': b_name,
                    'N_Windows': int(n_windows),
                    'ADE (m)': round(mean_ade, 3),
                    'ADE Std': round(std_ade, 3)
                })
                
            print(f"[{condition_name}] Evaluated.")

    # 7. Save Long-Format Table
    df = pd.DataFrame(results)
    df.to_csv("Table4_E4_TCM_Severity.csv", index=False)
    print("\nSaved Table4_E4_TCM_Severity.csv")
    
    # 8. Generate Figure 4 (Grouped Bar Chart)
    plt.figure(figsize=(14, 8))
    
    conditions = list(datasets.keys())
    x = np.arange(len(bins_order))
    width = 0.2  # 4 bars -> total width 0.8
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    colors = ['gray', 'red', 'purple', 'blue']
    
    for i, condition in enumerate(conditions):
        subset = df[df['Condition'] == condition].copy()
        # Ensure ordering matches bins_order
        subset['Burst_Bin'] = pd.Categorical(subset['Burst_Bin'], categories=bins_order, ordered=True)
        subset = subset.sort_values('Burst_Bin')
        
        ades = subset['ADE (m)'].values
        # To avoid massive overlap clutter, we let the Table carry std values and keep the chart clean, 
        # or optionally plot thin yerr bars here. We will omit yerr for cleanliness, matching E1.
        
        plt.bar(x + offsets[i], ades, width, label=condition, color=colors[i], edgecolor='black')

    plt.title('Experiment 4: Trajectory Error by TCM Burst Severity (20% CAV MPR)', fontsize=15, fontweight='bold')
    plt.xlabel('Maximum Contiguous V2X Packet Loss (Steps)', fontsize=13)
    plt.ylabel('Average Displacement Error (Meters)', fontsize=13)
    plt.xticks(x, bins_order, fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Legend formatting
    plt.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("Fig4_E4_TCM_BarChart.png", dpi=300)
    print("Saved Fig4_E4_TCM_BarChart.png")

if __name__ == "__main__":
    run_e4_evaluation()