import numpy as np

data = np.load('data/penetration_30/run_seed_1.npz', allow_pickle=True)
meta = data['metadata']
leader_types = [m['leader'] for m in meta]

from collections import Counter
all_types = [t for window in leader_types for t in window]
print(Counter(all_types))
data.close()