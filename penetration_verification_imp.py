import numpy as np
from collections import Counter

for pct in ['penetration_10', 'penetration_20', 'penetration_30']:
    data = np.load(f'data/penetration_{pct.split("_")[1]}/run_seed_1.npz', 
                   allow_pickle=True)
    meta = data['metadata']
    all_leaders = [t for m in meta for t in m['leader']]
    c = Counter(all_leaders)
    total = sum(c.values())
    print(f"{pct}: CAV={100*c['CAV']/total:.1f}% HDV={100*c['HDV']/total:.1f}% none={100*c['none']/total:.1f}%")
    data.close()