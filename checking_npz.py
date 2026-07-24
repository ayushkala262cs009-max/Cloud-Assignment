import numpy as np
import pickle

data = np.load('data/penetration_20/run_seed_1.npz', allow_pickle=True)
meta = data['metadata']
print(f"metadata shape: {meta.shape}")
print(f"metadata dtype: {meta.dtype}")
print(f"metadata[0] type: {type(meta[0])}")
print(f"metadata[0] value: {meta[0]}")
data.close()