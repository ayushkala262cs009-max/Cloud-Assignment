import torch
from torch.utils.data import DataLoader
from lstm_predictor import compute_scalers, TrajectoryDataset, LSTMPredictor, train_lstm

def run_training_pipeline():
    data_dir = "data_corrupted"
    
    # 70% Train, 10% Validation, 20% Test (strictly by Run ID)
    train_seeds = {f"run_seed_{i}.npz" for i in range(1, 15)}
    val_seeds = {f"run_seed_{i}.npz" for i in range(15, 17)}
    
    print("Computing unified normalization scalers from training Ground Truth...")
    sx_m, sx_s, sy_m, sy_s = compute_scalers(data_dir, train_seeds)
    
    print("Scaler X Mean (7D):", sx_m)
    print("Scaler X Std  (7D):", sx_s)
    print("Scaler Y Mean (2D):", sy_m)
    print("Scaler Y Std  (2D):", sy_s)

    print("\nInitializing Training Dataset (Seeds 1-14)...")
    train_dataset = TrajectoryDataset(
        data_dir=data_dir, 
        valid_seeds=train_seeds, 
        input_key="X_gt",  # Train STRICTLY on ideal ground truth
        scaler_X_mean=sx_m, scaler_X_std=sx_s, 
        scaler_Y_mean=sy_m, scaler_Y_std=sy_s
    )
    
    print("Initializing Validation Dataset (Seeds 15-16)...")
    # NOTE: Val loss here measures prediction quality from perfect X_gt only.
    # This is used for early stopping / checkpoint selection only.
    # Step 11 evaluation uses test seeds 17-20 with corrupted/imputed inputs.
    val_dataset = TrajectoryDataset(
        data_dir=data_dir, 
        valid_seeds=val_seeds, 
        input_key="X_gt", 
        scaler_X_mean=sx_m, scaler_X_std=sx_s, 
        scaler_Y_mean=sy_m, scaler_Y_std=sy_s
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print(f"Samples -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = LSTMPredictor(
        input_dim=7, hidden_dim=128, num_layers=2, output_steps=50, output_dim=2,
        scaler_X_mean=torch.from_numpy(sx_m).float(),
        scaler_X_std=torch.from_numpy(sx_s).float(),
        scaler_Y_mean=torch.from_numpy(sy_m).float(),
        scaler_Y_std=torch.from_numpy(sy_s).float()
    )
    
    train_lstm(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=30, 
        lr=1e-3, 
        device=device,
        save_path="lstm_baseline.pt"
    )

if __name__ == "__main__":
    run_training_pipeline()