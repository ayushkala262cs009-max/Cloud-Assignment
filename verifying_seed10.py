import numpy as np
from collections import Counter

for seed in [1, 5, 10, 15, 20]:
    data = np.load(f'data/penetration_30/run_seed_{seed}.npz', allow_pickle=True)
    meta = data['metadata']
    all_leaders = [t for m in meta for t in m['leader']]
    c = Counter(all_leaders)
    total = sum(c.values())
    print(f"seed {seed:2d}: CAV={100*c['CAV']/total:.1f}% HDV={100*c['HDV']/total:.1f}%")
    data.close()