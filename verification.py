import numpy as np
import pickle

X = np.load('ground_truth_X.npy')
Y = np.load('ground_truth_Y.npy')
with open('type_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

print("=== SHAPE CHECK ===")
print(f"X: {X.shape}  — expected (N, 30, 7)")
print(f"Y: {Y.shape}  — expected (N, 50, 2)")
print(f"Meta: {len(meta)} records")

print("\n=== NaN CHECK (ground truth should have zero NaN in ego columns) ===")
for col, name in enumerate(['ego_speed','ego_accel','ego_x','ego_y']):
    n = np.isnan(X[:,:,col]).sum()
    print(f"  {name:15s} NaN count: {n}")

print("\n=== NaN CHECK (neighbor columns — NaN expected when HDV) ===")
for col, name in enumerate(['gap_to_leader','leader_speed','follower_speed'], start=4):
    n = np.isnan(X[:,:,col]).sum()
    pct = 100 * n / (X.shape[0] * X.shape[1])
    print(f"  {name:15s} NaN count: {n:6d}  ({pct:.1f}%)")

print("\n=== PHYSICAL BOUNDS CHECK ===")
print(f"  Speed    min={X[:,:,0].min():.2f}  max={X[:,:,0].max():.2f}  (expected 0-33.33)")
print(f"  Accel    min={X[:,:,1].min():.2f}  max={X[:,:,1].max():.2f}  (expected -5 to +3)")
print(f"  X pos    min={X[:,:,2].min():.2f}  max={X[:,:,2].max():.2f}  (expected 200-800)")
print(f"  Y pos    min={X[:,:,3].min():.2f}  max={X[:,:,3].max():.2f}  (expected -5 to +5)")

print("\n=== LABEL CHECK ===")
print(f"  Y x-pos  min={Y[:,:,0].min():.2f}  max={Y[:,:,0].max():.2f}")
print(f"  Y y-pos  min={Y[:,:,1].min():.2f}  max={Y[:,:,1].max():.2f}")

print("\n=== METADATA SAMPLE (window 0) ===")
print(f"  Leader types:   {set(meta[0]['leader'])}")
print(f"  Follower types: {set(meta[0]['follower'])}")

print("\n=== HDV NEIGHBOUR RATE ===")
hdv_leader = sum(
    1 for m in meta for t in m['leader'] if t == 'HDV'
)
total_steps = len(meta) * 30
print(f"  HDV leader steps: {hdv_leader}/{total_steps} ({100*hdv_leader/total_steps:.1f}%)")