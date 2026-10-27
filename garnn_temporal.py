import torch
import torch.nn as nn


class GarnnTemporalBranch(nn.Module):
    """
    Forward-only temporal branch for GARNN-style reconstruction.
    Uses value/mask LSTM split + fusion + decay gate + residual estimate.
    """

    def __init__(self, feature_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.value_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.mask_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)

        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decay_gate = nn.Linear(hidden_dim + feature_dim, feature_dim)
        self.residual_head = nn.Linear(hidden_dim, feature_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.shape != mask.shape:
            raise ValueError(f"values shape {values.shape} must match mask shape {mask.shape}")

        value_state, _ = self.value_lstm(values)
        mask_state, _ = self.mask_lstm(mask)

        fused = torch.tanh(self.fusion(torch.cat([value_state, mask_state], dim=-1)))

        gate_input = torch.cat([fused, mask], dim=-1)
        decay = torch.sigmoid(self.decay_gate(gate_input))

        residual = self.residual_head(fused)

        # Keep observed entries unchanged and reconstruct only missing entries.
        reconstructed = values + (1.0 - mask) * decay * residual
        return reconstructed


def masked_reconstruction_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MAE computed only on masked cells (mask == 0)."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target and mask must have identical shapes")

    missing = 1.0 - mask
    denom = missing.sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)

    abs_err = torch.abs(prediction - target) * missing
    return abs_err.sum() / denom
