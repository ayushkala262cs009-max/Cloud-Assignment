import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_predictor import load_lstm
import baseline as baselines

def compute_metrics(preds_physical: np.ndarray, y_physical: np.ndarray):
    """Computes overall ADE and FDE in physical meters."""
    distances = np.sqrt(np.sum((preds_physical - y_physical) ** 2, axis=2))
    ade = np.mean(distances)
    ade_std = np.std(np.mean(distances, axis=1))
    fde = np.mean(distances[:, -1])
    fde_std = np.std(distances[:, -1])
    return ade, ade_std, fde, fde_std

def evaluate_penetration_slice(pen_rate: str, model: torch.nn.Module, test_seeds: set, device: torch.device) -> list:
    """
    Reusable evaluation core. Loads test seeds for a specific penetration rate, 
    generates all 5 baselines dynamically, runs them through the frozen LSTM, 
    and returns a list of metric dictionaries.
    """
    path = os.path.join("data_corrupted", f"penetration_{pen_rate}")
    raw_Y_list, raw_X_corr_list, raw_X_ch1_list, raw_X_fully_list = [], [], [], []
    
    for fname in sorted(os.listdir(path)):
        if fname in test_seeds:
            data = np.load(os.path.join(path, fname), allow_pickle=True)
            raw_Y_list.append(data['Y'])
            raw_X_corr_list.append(data['X_corrupted'])
            raw_X_ch1_list.append(data['X_ch1_imputed'])
            raw_X_fully_list.append(data['X_fully_imputed'])
            
    test_Y = np.concatenate(raw_Y_list, axis=0).astype(np.float32)
    test_X_corr = np.concatenate(raw_X_corr_list, axis=0).astype(np.float32)
    
    print(f"Generating mathematical baselines for {pen_rate}% CAV...")
    datasets = {
        'Baseline 1: Zero-fill': baselines.zero_fill(test_X_corr),
        'Baseline 2: Forward-fill': baselines.forward_fill(test_X_corr),
        'Baseline 3: Mean-fill (Local)': baselines.mean_fill(test_X_corr),
        'Baseline 4: Linear Interpolation': baselines.linear_interp(test_X_corr),
        'Baseline 5: Spline Interpolation': baselines.spline_interp(test_X_corr),
        'Condition B: Channel 1 Only': np.concatenate(raw_X_ch1_list, axis=0).astype(np.float32),
        'Condition C: Dual-Channel (Proposed)': np.concatenate(raw_X_fully_list, axis=0).astype(np.float32)
    }
    
    # Extract scalers directly from model buffers
    sx_m = model.scaler_X_mean.cpu().numpy()
    sx_s = model.scaler_X_std.cpu().numpy()
    sy_m = model.scaler_Y_mean.cpu().numpy()
    sy_s = model.scaler_Y_std.cpu().numpy()
    
    results = []
    
    with torch.no_grad():
        for name, X_physical in datasets.items():
            
            # UNIVERSAL MISSING DATA RULE:
            # If any method fails to impute a value, it defaults to physical 0.0 BEFORE normalization.
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
            ade, ade_std, fde, fde_std = compute_metrics(preds_physical, test_Y)
            
            results.append({
                'Condition': name,
                'Penetration (%)': int(pen_rate),
                'ADE (m)': round(ade, 3),
                'ADE Std': round(ade_std, 3),
                'FDE (m)': round(fde, 3),
                'FDE Std': round(fde_std, 3)
            })
            print(f"  [{name}] ADE: {ade:.2f}m")
            
    return results

def run_e2_evaluation():
    print("=" * 60)
    print("STEP 11 - EXPERIMENT 2: SENSITIVITY TO CAV PENETRATION RATE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the frozen LSTM instrument
    print(f"Loading frozen LSTM baseline on {device}...")
    model = load_lstm("lstm_baseline.pt", device=device)
    
    test_seeds = {f"run_seed_{i}.npz" for i in range(17, 21)}
    penetrations = ["10", "20", "30"]
    
    all_results = []
    
    # 2. Execute Outer Loop over Penetration Rates
    for pen in penetrations:
        print(f"\n--- Evaluating Penetration Rate: {pen}% CAV ---")
        slice_results = evaluate_penetration_slice(pen, model, test_seeds, device)
        all_results.extend(slice_results)
                
    # 3. Save Table in Long Format
    df = pd.DataFrame(all_results)
    df.to_csv("Table2_E2_MPR_Metrics.csv", index=False)
    print("\nSaved Table2_E2_MPR_Metrics.csv")
    
    # 4. Generate Figure 2 (Line Chart)
    plt.figure(figsize=(14, 8))
    
    # Define plotting styles for the 7 conditions
    styles = {
        'Baseline 1: Zero-fill': {'color': 'gray', 'linestyle': '--', 'marker': 'x'},
        'Baseline 2: Forward-fill': {'color': 'silver', 'linestyle': ':', 'marker': '.'},
        'Baseline 3: Mean-fill (Local)': {'color': 'brown', 'linestyle': ':', 'marker': 'v'},
        'Baseline 4: Linear Interpolation': {'color': 'orange', 'linestyle': '-.', 'marker': 's'},
        'Baseline 5: Spline Interpolation': {'color': 'red', 'linestyle': '-.', 'marker': '^'},
        'Condition B: Channel 1 Only': {'color': 'purple', 'linestyle': '-', 'marker': 'd', 'linewidth': 2},
        'Condition C: Dual-Channel (Proposed)': {'color': 'blue', 'linestyle': '-', 'marker': 'o', 'linewidth': 3}
    }
    
    for condition, style in styles.items():
        subset = df[df['Condition'] == condition]
        plt.plot(subset['Penetration (%)'], subset['ADE (m)'], 
                 label=condition, 
                 color=style['color'],
                 linestyle=style['linestyle'],
                 marker=style['marker'],
                 markersize=8,
                 linewidth=style.get('linewidth', 1.5))
                 
    plt.title('Experiment 2: Trajectory Error by CAV Penetration Rate', fontsize=16, fontweight='bold')
    plt.xlabel('CAV Market Penetration Rate (%)', fontsize=14)
    plt.ylabel('Average Displacement Error (ADE) in Meters', fontsize=14)
    plt.xticks([10, 20, 30], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Place legend outside to prevent cluttering the phase-transition curve
    plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("Fig2_E2_MPR_LineChart.png", dpi=300, bbox_inches='tight')
    print("Saved Fig2_E2_MPR_LineChart.png")

if __name__ == "__main__":
    run_e2_evaluation()