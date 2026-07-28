import pandas as pd
import numpy as np

def inspect_highd_dataset(csv_path: str):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*50)
    print("1. COLUMN NAMES")
    print("="*50)
    print(df.columns.tolist())
    
    print("\n" + "="*50)
    print("2. FRAME CONTINUITY (np.diff == 1)")
    print("="*50)
    # Group by vehicle ID and calculate the maximum frame difference for each vehicle
    max_diffs = df.groupby('id')['frame'].apply(lambda x: np.diff(x).max() if len(x) > 1 else 1)
    max_overall_diff = max_diffs.max()
    
    if max_overall_diff == 1:
        print("SUCCESS: All vehicle tracks are perfectly continuous. No dropped frames!")
    else:
        print(f"WARNING: Discontinuities found! Maximum frame jump: {max_overall_diff}")
        bad_ids = max_diffs[max_diffs > 1].index.tolist()
        print(f"Number of vehicles with gaps: {len(bad_ids)}")
        
    print("\n" + "="*50)
    print("3. LEADER/FOLLOWER NULL REPRESENTATION")
    print("="*50)
    if 'precedingId' in df.columns:
        unique_preceding = df['precedingId'].unique()
        print(f"Sample of precedingId values: {unique_preceding[:10]}")
        if 0 in unique_preceding:
            print("Confirmed: '0' is used to represent 'no vehicle ahead'.")
        if df['precedingId'].isna().any():
            print("WARNING: NaN values found in precedingId!")
    else:
        print("ERROR: 'precedingId' column not found.")

    print("\n" + "="*50)
    print("4. DRIVING DIRECTION & COORDINATE SYSTEM")
    print("="*50)
    if 'drivingDirection' in df.columns and 'x' in df.columns:
        directions = df['drivingDirection'].unique()
        print(f"Driving directions found: {directions}")
        
        for d in directions:
            sample_veh = df[df['drivingDirection'] == d]['id'].iloc[0]
            veh_track = df[df['id'] == sample_veh].sort_values('frame')
            x_start = veh_track['x'].iloc[0]
            x_end = veh_track['x'].iloc[-1]
            movement = "Increasing (+)" if x_end > x_start else "Decreasing (-)"
            print(f"  Direction {d}: X-coordinate is {movement}")
    else:
        print("ERROR: 'drivingDirection' or 'x' not found.")

if __name__ == "__main__":
    # Change this to point to your actual local file, e.g., '01_tracks.csv'
    # MAKE SURE it is the tracks.csv, NOT the tracksMeta.csv
    TARGET_FILE = "D:\\highD-dataset-v1.0\\data\\01_tracks.csv" 
    
    try:
        inspect_highd_dataset(TARGET_FILE)
    except FileNotFoundError:
        print(f"File '{TARGET_FILE}' not found. Please update the TARGET_FILE path.")