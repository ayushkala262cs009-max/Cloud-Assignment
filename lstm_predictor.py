import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def compute_scalers(data_dir: str, train_seeds: set):
    """
    Computes unified normalization scalers strictly from the training split's 
    ground truth (X_gt and Y). This guarantees zero data leakage from val/test sets
    and establishes a pristine physical baseline for the downstream predictor.
    """
    scenarios = ["penetration_10", "penetration_20", "penetration_30"]
    raw_X_list, raw_Y_list = [], []

    for sc in scenarios:
        sc_path = os.path.join(data_dir, sc)
        if not os.path.exists(sc_path):
            continue
            
        for fname in sorted(os.listdir(sc_path)):
            if fname in train_seeds:
                fpath = os.path.join(sc_path, fname)
                try:
                    with np.load(fpath, allow_pickle=True) as data:
                        raw_X_list.append(data['X_gt'].astype(np.float32))
                        raw_Y_list.append(data['Y'].astype(np.float32))
                except Exception as e:
                    print(f"Error loading {fpath} for scalers: {e}")

    raw_X = np.concatenate(raw_X_list, axis=0)
    raw_Y = np.concatenate(raw_Y_list, axis=0)

    # Compute across batch and time dimensions, yielding a 7D scaler for X and 2D scaler for Y
    scaler_X_mean = np.nanmean(raw_X, axis=(0, 1))
    scaler_X_std = np.nanstd(raw_X, axis=(0, 1))
    scaler_X_std = np.clip(scaler_X_std, a_min=1e-6, a_max=None)
    
    scaler_Y_mean = np.mean(raw_Y, axis=(0, 1))
    scaler_Y_std = np.std(raw_Y, axis=(0, 1))
    scaler_Y_std = np.clip(scaler_Y_std, a_min=1e-6, a_max=None)

    return scaler_X_mean, scaler_X_std, scaler_Y_mean, scaler_Y_std

class TrajectoryDataset(Dataset):
    """
    Flexible dataset loader capable of targeting specific input arrays (X_gt, X_corrupted, etc.)
    Allows seamless reuse of the same class for training and downstream evaluation.
    """
    def __init__(self, data_dir: str, valid_seeds: set, input_key: str, 
                 scaler_X_mean: np.ndarray, scaler_X_std: np.ndarray,
                 scaler_Y_mean: np.ndarray, scaler_Y_std: np.ndarray):
        
        scenarios = ["penetration_10", "penetration_20", "penetration_30"]
        raw_X_list, raw_Y_list = [], []

        for sc in scenarios:
            sc_path = os.path.join(data_dir, sc)
            if not os.path.exists(sc_path):
                continue
                
            for fname in sorted(os.listdir(sc_path)):
                if fname in valid_seeds:
                    fpath = os.path.join(sc_path, fname)
                    try:
                        with np.load(fpath, allow_pickle=True) as data:
                            raw_X_list.append(data[input_key].astype(np.float32))
                            raw_Y_list.append(data['Y'].astype(np.float32))
                    except Exception as e:
                        pass

        raw_X = np.concatenate(raw_X_list, axis=0)
        raw_Y = np.concatenate(raw_Y_list, axis=0)

        norm_X = (raw_X - scaler_X_mean) / scaler_X_std
        norm_Y = (raw_Y - scaler_Y_mean) / scaler_Y_std

        # Safely neutralize structural NaNs (like Col 6) or physical blackouts (Cols 4-5) 
        # by converting to 0.0, placing them at the exact mean of the z-score distribution.
        norm_X = np.nan_to_num(norm_X, nan=0.0)

        self.samples_X = torch.from_numpy(norm_X)
        self.samples_Y = torch.from_numpy(norm_Y)

    def __len__(self):
        return len(self.samples_X)

    def __getitem__(self, idx):
        return self.samples_X[idx], self.samples_Y[idx]

class LSTMPredictor(nn.Module):
    """
    Standard Two-Layer LSTM.
    Takes (Batch, 30, 7), extracts final hidden state, predicts (Batch, 50, 2).
    """
    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, num_layers: int = 2,
                 output_steps: int = 50, output_dim: int = 2,
                 scaler_X_mean: torch.Tensor = None, scaler_X_std: torch.Tensor = None,
                 scaler_Y_mean: torch.Tensor = None, scaler_Y_std: torch.Tensor = None):
        super().__init__()
        self.output_steps = output_steps
        self.output_dim = output_dim
        
        # Register scalers to ensure they travel in the state_dict
        self.register_buffer('scaler_X_mean', scaler_X_mean)
        self.register_buffer('scaler_X_std', scaler_X_std)
        self.register_buffer('scaler_Y_mean', scaler_Y_mean)
        self.register_buffer('scaler_Y_std', scaler_Y_std)

        # Encoder: Consume the 30-step trajectory
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Decoder: Linear projection from the final hidden state to the flat future trajectory
        self.fc = nn.Linear(hidden_dim, output_steps * output_dim)

    def denormalise_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Reverts predicted coordinates back to physical meters."""
        return y_norm * self.scaler_Y_std + self.scaler_Y_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 30, 7)
        out, _ = self.lstm(x)
        
        # Extract the final hidden state of the top layer at t=30
        last_hidden = out[:, -1, :]  # shape: (B, hidden_dim)
        
        # Project to flat 100-dimensional vector
        pred_flat = self.fc(last_hidden)
        
        # Reshape to final spatial trajectory (B, 50, 2)
        pred_seq = pred_flat.view(-1, self.output_steps, self.output_dim)
        return pred_seq

def train_lstm(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 30, lr: float = 1e-3, device: str = "cpu", 
               save_path: str = "lstm_baseline.pt"):
    """Standard MSE training loop for the trajectory predictor."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    print("=" * 60)
    print("TRAINING DOWNSTREAM LSTM PREDICTOR")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            saved_str = " -> [Model Saved]"
        else:
            saved_str = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {avg_train_loss:.5f} | Val MSE: {avg_val_loss:.5f}{saved_str}")

    print("=" * 60)
    print(f"TRAINING COMPLETE. Best Val Loss: {best_val_loss:.5f}")
    print("=" * 60)
        
def load_lstm(path: str, device: str = 'cpu') -> LSTMPredictor:
    """Load a saved LSTMPredictor with scalers correctly restored."""
    model = LSTMPredictor(
        input_dim=7, hidden_dim=128, num_layers=2,
        scaler_X_mean=torch.zeros(7),
        scaler_X_std=torch.ones(7),
        scaler_Y_mean=torch.zeros(2),
        scaler_Y_std=torch.ones(2)
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model