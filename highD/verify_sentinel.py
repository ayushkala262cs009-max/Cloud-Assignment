import pandas as pd
import os

def verify_sentinel(tracks_file: str):
    print(f"Loading {tracks_file}...")
    df = pd.read_csv(tracks_file)
    
    print("\n" + "="*50)
    print("VERIFYING dhw == 0.0 EXCLUSIVITY")
    print("="*50)
    
    # 1. Does dhw == 0.0 ever occur when a leader genuinely exists?
    false_zero = df[(df['precedingId'] != 0) & (df['dhw'] == 0.0)]
    print(f"Rows with a real leader but dhw == 0.0: {len(false_zero)}")
    if len(false_zero) > 0:
        print("  -> WARNING: 0.0 is NOT exclusive! Real gaps hit 0.0.")
        
    # 2. Is precedingId ALWAYS 0 when dhw == 0.0?
    mismatch = df[(df['dhw'] == 0.0) & (df['precedingId'] != 0)]
    print(f"Mismatches (dhw=0 but leader exists): {len(mismatch)}")
    
    # 3. Just to be absolutely certain, what is the smallest non-zero gap?
    min_real_gap = df[df['dhw'] > 0.0]['dhw'].min()
    print(f"Smallest strictly positive dhw: {min_real_gap} meters")

if __name__ == "__main__":
    TRACKS_PATH = "D:/highD-dataset-v1.0/data/01_tracks.csv"
    if os.path.exists(TRACKS_PATH):
        verify_sentinel(TRACKS_PATH)
    else:
        print("File not found. Check path.")