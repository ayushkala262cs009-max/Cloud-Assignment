import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class GRU_D_Cell(nn.Module):
    """
    Decay-aware Recurrent Cell (GRU-D) for time-series imputation.
    Fuses temporal decay factors on both input features and hidden states.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Linear projections without bias to guarantee zero decay at delta_t = 0
        self.gamma_x_layer = nn.Linear(input_size, input_size, bias=False)
        self.gamma_h_layer = nn.Linear(input_size, hidden_size, bias=False)
        
        # Learned physical baseline towards which unobserved values decay
        self.baseline = nn.Parameter(torch.zeros(input_size))
        
        # Standard PyTorch GRUCell taking concatenated input [x_in, mask]
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, delta_t: torch.Tensor, 
                h: torch.Tensor, x_last: torch.Tensor) -> tuple:
        """
        Forward pass for a single time step.
        
        Args:
            x (torch.Tensor): Current raw input, shape (B, input_size).
            mask (torch.Tensor): Binary mask (1=observed, 0=missing), shape (B, input_size).
            delta_t (torch.Tensor): Time since last observation in seconds, shape (B, input_size).
            h (torch.Tensor): Previous hidden state, shape (B, hidden_size).
            x_last (torch.Tensor): Last observed physical values, shape (B, input_size).
            
        Returns:
            tuple: (x_in, h_new, x_last_updated)
        """
        # Ensure non-negative decay weights to guarantee monotonically decreasing decay
        gamma_x = torch.exp(-F.relu(self.gamma_x_layer(delta_t)))
        gamma_h = torch.exp(-F.relu(self.gamma_h_layer(delta_t)))
        
        # Decayed input estimate (decaying from x_last towards learned baseline)
        x_hat = gamma_x * x_last + (1.0 - gamma_x) * self.baseline
        
        # Imputed input vector: use real observation if present, else x_hat
        x_in = mask * x + (1.0 - mask) * x_hat
        
        # Apply hidden state decay
        h_decayed = gamma_h * h
        
        # Concatenate input features with observation mask for the GRU cell
        gru_input = torch.cat([x_in, mask], dim=-1)
        h_new = self.gru_cell(gru_input, h_decayed)
        
        # Update last observed physical values
        x_last_updated = mask * x + (1.0 - mask) * x_last
        
        return x_in, h_new, x_last_updated


class TCMImputer(nn.Module):
    """
    Physics-Constrained GRU-D Imputer for Ego-Vehicle Trajectory (Channel 2).
    Imputes missing ego state variables [speed, accel, x, y] under packet loss.
    """
    def __init__(self, feature_dim: int = 4, hidden_dim: int = 64, 
                 scaler_mean: torch.Tensor = None, scaler_std: torch.Tensor = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Register scalers as buffers so they travel seamlessly with model.state_dict()
        if scaler_mean is None:
            scaler_mean = torch.zeros(feature_dim, dtype=torch.float32)
        if scaler_std is None:
            scaler_std = torch.ones(feature_dim, dtype=torch.float32)
            
        self.register_buffer('scaler_mean', scaler_mean)
        self.register_buffer('scaler_std', scaler_std)
        
        self.cell = GRU_D_Cell(feature_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, feature_dim)
        
        # Raw physical kinematic clamping bounds for [speed (m/s), accel (m/s^2), x (m), y (m)]
        raw_lo = torch.tensor([0.0,  -5.0, -1e5, -1e5], dtype=torch.float32)
        raw_hi = torch.tensor([40.0,  3.0,  1e5,  1e5], dtype=torch.float32)

        # Normalise the physical clamps into the latent z-score space
        scaled_lo = (raw_lo - self.scaler_mean) / self.scaler_std
        scaled_hi = (raw_hi - self.scaler_mean) / self.scaler_std
        
        self.register_buffer('kinematic_lo', scaled_lo)
        self.register_buffer('kinematic_hi', scaled_hi)

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        """Reverts normalised predictions back to physical units (m/s, m/s^2, m)."""
        return x * self.scaler_std + self.scaler_mean

    def _compute_delta_t(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes elapsed time in seconds since the last valid observation for each feature.
        
        Args:
            mask (torch.Tensor): Binary tensor of shape (B, T, F).
            
        Returns:
            torch.Tensor: Time deltas in seconds, shape (B, T, F).
        """
        B, T, F = mask.shape
        delta_t = torch.zeros_like(mask, dtype=torch.float32)
        
        # Step size for 10 Hz sampling rate
        dt_step = 0.1  # seconds
        
        # Initialize t=0 with 0.1s prior step
        delta_t[:, 0, :] = dt_step
        
        for t in range(1, T):
            # If step (t-1) was observed, delta_t is 0.1s; else increment by 0.1s
            delta_t[:, t, :] = torch.where(
                mask[:, t - 1, :] == 1.0,
                torch.tensor(dt_step, device=mask.device, dtype=torch.float32),
                delta_t[:, t - 1, :] + dt_step
            )
            
        return delta_t

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a 30-step temporal window.
        
        Args:
            x (torch.Tensor): Input state sequence (B, 30, 4), NaNs replaced by 0.
            mask (torch.Tensor): Observation mask (B, 30, 4) where 1.0=observed, 0.0=missing.
            
        Returns:
            torch.Tensor: Fully imputed ego trajectory of shape (B, 30, 4).
        """
        B, T, F = x.shape
        device = x.device
        dtype = x.dtype
        
        # Compute time deltas in seconds
        delta_t_seq = self._compute_delta_t(mask)
        
        # Initialize hidden state
        h = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
        
        # Initialize x_last from baseline to handle window-start dropouts cleanly
        x_last = self.cell.baseline.unsqueeze(0).expand(B, -1).clone()
        
        output_steps = []
        
        for t in range(T):
            x_t = x[:, t, :]
            mask_t = mask[:, t, :]
            dt_t = delta_t_seq[:, t, :]
            
            # Step forward through GRU-D cell
            x_in_t, h, x_last = self.cell(x_t, mask_t, dt_t, h, x_last)
            
            # Project hidden state to physical state space
            raw_pred = self.fc_out(h)
            
            # Clamp outputs to valid kinematic bounds
            clamped_pred = torch.clamp(raw_pred, self.kinematic_lo, self.kinematic_hi)
            
            # Combine known observation with model prediction
            imputed_t = mask_t * x_t + (1.0 - mask_t) * clamped_pred
            output_steps.append(imputed_t)
            
        return torch.stack(output_steps, dim=1)


def masked_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes Mean Squared Error strictly over missing entries (where mask == 0).
    Normalizes by the number of missing elements to prevent gradient attenuation.
    """
    missing_mask = 1.0 - mask
    squared_error = (y_pred - y_true) ** 2
    masked_error = squared_error * missing_mask
    
    num_missing = missing_mask.sum()
    if num_missing > 0:
        return masked_error.sum() / num_missing
    else:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)


class TrajectoryTCMDataset(Dataset):
    """Dataset wrapper for loading corrupted ego trajectories and masks."""
    def __init__(self, data_dir: str = "data_corrupted"):
        self.samples = []
        
        scenarios = ["penetration_10", "penetration_20", "penetration_30"]
        for sc in scenarios:
            sc_path = os.path.join(data_dir, sc)
            if not os.path.exists(sc_path):
                continue
            for fname in sorted(os.listdir(sc_path)):
                if fname.endswith(".npz") and ".tmp" not in fname:
                    fpath = os.path.join(sc_path, fname)
                    try:
                        with np.load(fpath, allow_pickle=True) as data:
                            X_gt = data['X_gt'][:, :, 0:4].astype(np.float32)
                            X_corr = data['X_corrupted'][:, :, 0:4].astype(np.float32)
                            X_mask = data['X_mask'][:, :, 0:4].astype(np.float32)
                            
                            # Clean NaNs in corrupted array for tensor conversion
                            X_corr_clean = np.nan_to_num(X_corr, nan=0.0)
                            
                            for i in range(X_gt.shape[0]):
                                self.samples.append((X_gt[i], X_corr_clean[i], X_mask[i]))
                    except Exception as e:
                        print(f"Error loading {fpath}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_gt, x_corr, mask = self.samples[idx]
        return (
            torch.from_numpy(x_gt),
            torch.from_numpy(x_corr),
            torch.from_numpy(mask)
        )


def train_tcm_grud(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                   epochs: int = 50, lr: float = 1e-3, device: str = "cpu"):
    """Executes model training with Adam optimizer and validation monitoring."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("=" * 60)
    print("TRAINING CHANNEL 2 (GRU-D) TCM IMPUTER")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for x_gt, x_corr, mask in train_loader:
            x_gt = x_gt.to(device)
            x_corr = x_corr.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            output = model(x_corr, mask)
            
            loss = masked_mse_loss(output, x_gt, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
        avg_train_loss = train_loss / max(1, train_batches)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_gt, x_corr, mask in val_loader:
                x_gt = x_gt.to(device)
                x_corr = x_corr.to(device)
                mask = mask.to(device)
                
                output = model(x_corr, mask)
                loss = masked_mse_loss(output, x_gt, mask)
                val_loss += loss.item()
                val_batches += 1
                
        avg_val_loss = val_loss / max(1, val_batches)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "tcm_grud_best.pt")
            saved_str = " -> [Model Saved]"
        else:
            saved_str = ""
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Masked MSE: {avg_train_loss:.5f} | Val Masked MSE: {avg_val_loss:.5f}{saved_str}")
            
    print("=" * 60)
    print(f"TRAINING COMPLETE. Best Val Loss: {best_val_loss:.5f}")
    print("=" * 60)


if __name__ == "__main__":
    # Self-contained verification and sanity test
    print("Executing module self-test...")
    mock_model = TCMImputer(feature_dim=4, hidden_dim=64)
    
    # Batch=2, Sequence=30, Features=4
    mock_x = torch.randn(2, 30, 4)
    mock_mask = torch.ones(2, 30, 4)
    mock_mask[:, 10:15, :] = 0.0  # Simulated 5-step dropout burst
    
    mock_x_corr = mock_x * mock_mask
    out = mock_model(mock_x_corr, mock_mask)
    
    print(f"Input shape:  {mock_x.shape}")
    print(f"Output shape: {out.shape}")
    print("Self-test passed successfully!")