import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_predictor import load_lstm
import baseline as baselines

def compute_temporal_error(preds_physical: np.ndarray, y_physical: np.ndarray) -> np.ndarray:
    """
    Computes mean Euclidean distance per timestep across all N samples.
    Returns a 1D array of shape (50,).
    """
    # distances shape: (N, 50)
    distances = np.sqrt(np.sum((preds_physical - y_physical) ** 2, axis=2))
    
    # Average across the N samples (axis 0), preserving the 50 timesteps
    error_at_t = np.mean(distances, axis=0)
    return error_at_t

def run_e3_evaluation():
    print("=" * 60)
    print("STEP 11 - EXPERIMENT 3: TEMPORAL PREDICTION HORIZON DEGRADATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the frozen LSTM instrument and extract scalers directly from buffers
    print(f"Loading frozen LSTM baseline on {device}...")
    model = load_lstm("lstm_baseline.pt", device=device)
    
    sx_m = model.scaler_X_mean.cpu().numpy()
    sx_s = model.scaler_X_std.cpu().numpy()
    sy_m = model.scaler_Y_mean.cpu().numpy()
    sy_s = model.scaler_Y_std.cpu().numpy()
    
    # 2. Load Test Data strictly for 20% Penetration (Base Case)
    test_seeds = {f"run_seed_{i}.npz" for i in range(17, 21)}
    path_20 = os.path.join("data_corrupted", "penetration_20")
    
    raw_Y_list, raw_X_corr_list, raw_X_ch1_list, raw_X_fully_list = [], [], [], []
    
    print("Loading test data (penetration_20, seeds 17-20)...")
    for fname in sorted(os.listdir(path_20)):
        if fname in test_seeds:
            data = np.load(os.path.join(path_20, fname), allow_pickle=True)
            raw_Y_list.append(data['Y'])
            raw_X_corr_list.append(data['X_corrupted'])
            raw_X_ch1_list.append(data['X_ch1_imputed'])
            raw_X_fully_list.append(data['X_fully_imputed'])
            
    test_Y = np.concatenate(raw_Y_list, axis=0).astype(np.float32)
    test_X_corr = np.concatenate(raw_X_corr_list, axis=0).astype(np.float32)
    
    # 3. Generate ONLY the necessary baselines to save compute time
    print("Generating necessary baselines in memory (Zero-fill and Spline)...")
    datasets = {
        'Baseline 1: Zero-fill': baselines.zero_fill(test_X_corr),
        'Baseline 5: Spline Interpolation': baselines.spline_interp(test_X_corr),
        'Condition B: Channel 1 Only': np.concatenate(raw_X_ch1_list, axis=0).astype(np.float32),
        'Condition C: Dual-Channel (Proposed)': np.concatenate(raw_X_fully_list, axis=0).astype(np.float32)
    }
    
    results = []
    
    # 4. Evaluation Loop
    print("\nRunning Model Evaluation...")
    with torch.no_grad():
        for name, X_physical in datasets.items():
            
            # Universal physical-space zero-fill for unresolved NaNs
            X_physical_clean = np.nan_to_num(X_physical, nan=0.0)
            
            # Normalize
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
            
            # Compute temporal error array (shape 50,)
            errors_at_t = compute_temporal_error(preds_physical, test_Y)
            
            # Record per-timestep data
            for t_idx, error in enumerate(errors_at_t):
                results.append({
                    'Condition': name,
                    'Timestep': t_idx + 1,
                    'Time (s)': round((t_idx + 1) * 0.1, 1),
                    'ADE_at_t (m)': round(error, 3)
                })
            
            # Quick summary metric for the console
            print(f"[{name}] Completed. Error at 5.0s: {errors_at_t[-1]:.2f}m")
            
    # 5. Save Table in Long Format
    df = pd.DataFrame(results)
    df.to_csv("Table3_E3_Horizon_Degradation.csv", index=False)
    print("\nSaved Table3_E3_Horizon_Degradation.csv")
    
    # 6. Generate Figure 3 (Horizon Degradation Line Chart)
    plt.figure(figsize=(12, 7))
    
    # Plotting styles
    styles = {
        'Baseline 1: Zero-fill': {'color': 'gray', 'linestyle': '--'},
        'Baseline 5: Spline Interpolation': {'color': 'red', 'linestyle': '-.'},
        'Condition B: Channel 1 Only': {'color': 'purple', 'linestyle': '-', 'linewidth': 2},
        'Condition C: Dual-Channel (Proposed)': {'color': 'blue', 'linestyle': '-', 'linewidth': 3}
    }
    
    for condition, style in styles.items():
        subset = df[df['Condition'] == condition]
        plt.plot(subset['Time (s)'], subset['ADE_at_t (m)'], 
                 label=condition, 
                 color=style['color'],
                 linestyle=style['linestyle'],
                 linewidth=style.get('linewidth', 1.5))
                 
    # Add horizontal reference line for safe planning horizon (Lane Width)
    plt.axhline(y=3.5, color='darkorange', linestyle=':', linewidth=2, 
                label='Safety Threshold (3.5m Lane Width)')
                 
    plt.title('Experiment 3: Temporal Prediction Horizon Degradation (20% CAV MPR)', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Horizon (Seconds)', fontsize=12)
    plt.ylabel('Average Displacement Error (Meters)', fontsize=12)
    plt.xticks(np.arange(0, 5.5, 0.5))
    plt.xlim(0.1, 5.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Legend placement
    plt.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("Fig3_E3_Horizon_Degradation.png", dpi=300)
    print("Saved Fig3_E3_Horizon_Degradation.png")

if __name__ == "__main__":
    run_e3_evaluation()