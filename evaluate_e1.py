import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_predictor import load_lstm

# Import the baseline imputation functions we just wrote
import baseline as baselines

def compute_metrics(preds_physical: np.ndarray, y_physical: np.ndarray):
    """
    Computes overall ADE and FDE in physical meters.
    preds_physical, y_physical shape: (N, 50, 2)
    """
    # Euclidean distance at every timestep for every sample: shape (N, 50)
    distances = np.sqrt(np.sum((preds_physical - y_physical) ** 2, axis=2))
    
    # ADE: Mean error across all 50 future timesteps, then across all N samples
    ade = np.mean(distances)
    ade_std = np.std(np.mean(distances, axis=1))
    
    # FDE: Mean error at exactly the final timestep (t=49), across all N samples
    fde = np.mean(distances[:, -1])
    fde_std = np.std(distances[:, -1])
    
    return ade, ade_std, fde, fde_std

def run_e1_evaluation():
    print("=" * 60)
    print("STEP 11 - EXPERIMENT 1: OVERALL IMPUTATION ABLATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1 & 2. Load the frozen LSTM instrument and extract scalers directly from its buffers
    print(f"Loading frozen LSTM baseline on {device}...")
    model = load_lstm("lstm_baseline.pt", device=device)
    
    sx_m = model.scaler_X_mean.cpu().numpy()
    sx_s = model.scaler_X_std.cpu().numpy()
    sy_m = model.scaler_Y_mean.cpu().numpy()
    sy_s = model.scaler_Y_std.cpu().numpy()
    
    # 3. Load Test Data for 20% Penetration (Seeds 17-20)
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
    # 4. Generate the baselines on the fly
    print("Generating baseline imputations in memory (this may take a moment)...")
    datasets = {
        'Baseline 1: Zero-fill (No Imputation)': baselines.zero_fill(test_X_corr),
        'Baseline 2: Forward-fill': baselines.forward_fill(test_X_corr),
        'Baseline 3: Mean-fill (Local)': baselines.mean_fill(test_X_corr),
        'Baseline 4: Linear Interpolation': baselines.linear_interp(test_X_corr),
        'Baseline 5: Spline Interpolation': baselines.spline_interp(test_X_corr),
        'Condition B: Channel 1 Only': np.concatenate(raw_X_ch1_list, axis=0).astype(np.float32),
        'Condition C: Dual-Channel (Proposed)': np.concatenate(raw_X_fully_list, axis=0).astype(np.float32)
    }
    
    results = []
    
    print("\nRunning Model Evaluation...")
    with torch.no_grad():
        for name, X_physical in datasets.items():
            
            # UNIVERSAL MISSING DATA RULE:
            # Any NaN surviving to this point (whether an un-imputed ego state in Cond B, 
            # a complete RSU blackout, or a structurally missing Col 6) is treated as a 
            # physical sensor dropout reading 0.0 before entering the normalizer.
            X_physical_clean = np.nan_to_num(X_physical, nan=0.0)
            
            # Apply training scalers to the fully realized physical array
            X_norm = (X_physical_clean - sx_m) / sx_s
            
            # Batch the inference to prevent memory spikes
            N = X_norm.shape[0]
            batch_size = 1024
            preds_physical_list = []
            
            for start_idx in range(0, N, batch_size):
                batch_X = torch.from_numpy(X_norm[start_idx : start_idx+batch_size]).to(device)
                
                # Predict in normalized space
                preds_norm = model(batch_X)
                
                # Denormalize predictions back to physical coordinates
                batch_preds_phys = (preds_norm.cpu().numpy() * sy_s) + sy_m
                preds_physical_list.append(batch_preds_phys)
                
            preds_physical = np.concatenate(preds_physical_list, axis=0)
            
            # Compute Metrics against Ground Truth Y
            ade, ade_std, fde, fde_std = compute_metrics(preds_physical, test_Y)
            
            results.append({
                'Condition': name,
                'ADE (m)': round(ade, 3),
                'ADE Std': round(ade_std, 3),
                'FDE (m)': round(fde, 3),
                'FDE Std': round(fde_std, 3)
            })
            
            print(f"[{name}] ADE: {ade:.2f}m \u00B1 {ade_std:.2f}m | FDE: {fde:.2f}m \u00B1 {fde_std:.2f}m")
            
    # 5. Save Table
    df = pd.DataFrame(results)
    df.to_csv("Table1_E1_Metrics.csv", index=False)
    print("\nSaved Table1_E1_Metrics.csv")
    
    # 6. Generate Figure 1 (Bar Chart)
    conditions = df['Condition']
    ades = df['ADE (m)']
    fdes = df['FDE (m)']
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, ades, width, label='ADE (m)', color='#4c72b0', edgecolor='black')
    rects2 = ax.bar(x + width/2, fdes, width, label='FDE (m)', color='#dd8452', edgecolor='black')
    
    ax.set_ylabel('Displacement Error (Meters)', fontsize=12)
    ax.set_title('Experiment 1: Overall Imputation Ablation (20% CAV MPR)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=11)
    
    # Add gridlines for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("Fig1_E1_BarChart.png", dpi=300)
    print("Saved Fig1_E1_BarChart.png")
    
if __name__ == "__main__":
    run_e1_evaluation()