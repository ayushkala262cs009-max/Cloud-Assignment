import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from lstm_predictor import load_lstm
import baseline as baselines

def get_modal_leaders(leader_meta_2d: np.ndarray) -> np.ndarray:
    """
    Computes the modal (most frequent) leader type for each 30-step window.
    leader_meta_2d shape: (N, 30)
    Returns: 1D array of shape (N,) containing 'CAV', 'HDV', or 'none'.
    """
    N, T = leader_meta_2d.shape
    modal_leaders = []
    
    for i in range(N):
        window = leader_meta_2d[i]
        # Decode bytes if loaded from serialized npz
        if isinstance(window[0], bytes):
            window = [w.decode('utf-8') for w in window]
            
        counts = Counter(window)
        # most_common(1) returns a list like [('HDV', 25)]
        most_common_type = counts.most_common(1)[0][0]
        modal_leaders.append(most_common_type)
        
    return np.array(modal_leaders)

def run_e5_evaluation():
    print("=" * 60)
    print("STEP 11 - EXPERIMENT 5: ROBUSTNESS TO SFM (LEADER CONNECTIVITY)")
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
    
    raw_Y_list, raw_X_corr_list = [], []
    raw_X_ch1_list, raw_X_fully_list = [], []
    leader_meta_list = []
    
    print("Loading test data (penetration_20, seeds 17-20)...")
    for fname in sorted(os.listdir(path_20)):
        if fname in test_seeds:
            data = np.load(os.path.join(path_20, fname), allow_pickle=True)
            raw_Y_list.append(data['Y'])
            raw_X_corr_list.append(data['X_corrupted'])
            raw_X_ch1_list.append(data['X_ch1_imputed'])
            raw_X_fully_list.append(data['X_fully_imputed'])
            leader_meta_list.append(data['leader_meta'])
            
    test_Y = np.concatenate(raw_Y_list, axis=0).astype(np.float32)
    test_X_corr = np.concatenate(raw_X_corr_list, axis=0).astype(np.float32)
    test_leader_meta = np.concatenate(leader_meta_list, axis=0)
    
    # 3. Compute Modal Leader Bins for all N windows
    modal_leaders_array = get_modal_leaders(test_leader_meta)
    bins_order = ["CAV", "HDV", "none"]
    
    # Console audit of bin counts
    print("\nModal Leader Bin Distribution:")
    for b_name in bins_order:
        count = np.sum(modal_leaders_array == b_name)
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
            
            # Universal physical-space zero-fill for unresolved NaNs BEFORE normalisation
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
            
            # 6. Aggregate by Leader Bin
            for b_name in bins_order:
                bin_mask = (modal_leaders_array == b_name)
                n_windows = np.sum(bin_mask)
                
                if n_windows > 0:
                    bin_ades = sample_ade[bin_mask]
                    mean_ade = np.mean(bin_ades)
                    std_ade = np.std(bin_ades)
                else:
                    mean_ade, std_ade = 0.0, 0.0
                    
                results.append({
                    'Condition': condition_name,
                    'Leader_Bin': b_name,
                    'N_Windows': int(n_windows),
                    'ADE (m)': round(mean_ade, 3),
                    'ADE Std': round(std_ade, 3)
                })
                
            print(f"[{condition_name}] Evaluated.")

    # 7. Save Long-Format Table
    df = pd.DataFrame(results)
    df.to_csv("Table5_E5_SFM_Leader.csv", index=False)
    print("\nSaved Table5_E5_SFM_Leader.csv")
    
    # 8. Generate Figure 5 (Grouped Bar Chart)
    plt.figure(figsize=(12, 7))
    
    conditions = list(datasets.keys())
    x = np.arange(len(bins_order))
    width = 0.2  # 4 bars -> total width 0.8
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    colors = ['gray', 'red', 'purple', 'blue']
    
    for i, condition in enumerate(conditions):
        subset = df[df['Condition'] == condition].copy()
        
        # Ensure ordering matches bins_order
        subset['Leader_Bin'] = pd.Categorical(subset['Leader_Bin'], categories=bins_order, ordered=True)
        subset = subset.sort_values('Leader_Bin')
        
        ades = subset['ADE (m)'].values
        plt.bar(x + offsets[i], ades, width, label=condition, color=colors[i], edgecolor='black')

    plt.title('Experiment 5: Trajectory Error by SFM Leader Connectivity (20% CAV MPR)', fontsize=14, fontweight='bold')
    plt.xlabel('Modal Leader Type (Window Connectivity Status)', fontsize=12)
    plt.ylabel('Average Displacement Error (Meters)', fontsize=12)
    plt.xticks(x, bins_order, fontsize=11)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Legend formatting
    plt.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("Fig5_E5_SFM_BarChart.png", dpi=300)
    print("Saved Fig5_E5_SFM_BarChart.png")

if __name__ == "__main__":
    run_e5_evaluation()