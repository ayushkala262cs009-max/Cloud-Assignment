import numpy as np

for pct, seed in [('penetration_10', 1), ('penetration_20', 1), ('penetration_30', 1)]:
    data = np.load(f'data_corrupted/{pct}/run_seed_{seed}.npz', allow_pickle=True)
    mask = data['X_mask']
    print(f"\n{pct}:")
    cols = ['ego_speed','ego_accel','ego_x','ego_y','gap_to_leader','leader_speed','follower_speed']
    for i, name in enumerate(cols):
        missing = (mask[:,:,i] == 0).sum()
        total = mask[:,:,i].size
        print(f"  {name:15s}: {100*missing/total:.1f}% missing")
    data.close()