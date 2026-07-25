import torch
import numpy as np
from lstm_predictor import LSTMPredictor

model = LSTMPredictor(
    input_dim=7, hidden_dim=128, num_layers=2,
    scaler_X_mean=torch.zeros(7),
    scaler_X_std=torch.ones(7),
    scaler_Y_mean=torch.zeros(2),
    scaler_Y_std=torch.ones(2)
)
model.load_state_dict(torch.load('lstm_baseline.pt', map_location='cpu'))
model.eval()

print(f"Scaler Y mean: {model.scaler_Y_mean}")
print(f"Scaler Y std:  {model.scaler_Y_std}")

# Load one test window from a test seed (seed 17)
data = np.load('data_corrupted/penetration_20/run_seed_17.npz',
               allow_pickle=True)
X = data['X_gt'][0:1].astype(np.float32)
Y = data['Y'][0:1].astype(np.float32)
data.close()

mean_x = model.scaler_X_mean.numpy()
std_x  = model.scaler_X_std.numpy()
mean_y = model.scaler_Y_mean.numpy()
std_y  = model.scaler_Y_std.numpy()

X_norm = (X - mean_x) / std_x
X_norm = np.nan_to_num(X_norm, nan=0.0)

with torch.no_grad():
    pred_norm = model(torch.from_numpy(X_norm))
    pred_phys = model.denormalise_y(pred_norm).numpy()

print(f"\nPredicted trajectory shape: {pred_phys.shape}")
print(f"First 3 predicted positions (x, y):")
for t in range(3):
    print(f"  t+{t+1}: x={pred_phys[0,t,0]:.2f}m  y={pred_phys[0,t,1]:.2f}m")

print(f"\nGround truth first 3 future positions:")
for t in range(3):
    print(f"  t+{t+1}: x={Y[0,t,0]:.2f}m  y={Y[0,t,1]:.2f}m")