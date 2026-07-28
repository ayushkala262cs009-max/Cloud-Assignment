import pandas as pd

# Load the data
df = pd.read_csv("D:/highD-dataset-v1.0/data/01_tracks.csv")
meta = pd.read_csv("D:/highD-dataset-v1.0/data/01_tracksMeta.csv")

# Create a fast lookup dictionary for direction
dir_map = dict(zip(meta['id'], meta['drivingDirection']))

# Map the ego direction and the leader direction for every row
ego_dirs = df['id'].map(dir_map)
# Use map directly, 0 won't be in the map so it will yield NaN, which is fine
leader_dirs = df['precedingId'].map(dir_map)

# Filter for rows where a leader exists, but the directions don't match
mismatches = df[(df['precedingId'] != 0) & (ego_dirs != leader_dirs)]

print(f"Total Leader-Direction Mismatches: {len(mismatches)}")