import pandas as pd
import numpy as np
import os

def run_precheck(tracks_file: str, meta_file: str):
    print(f"Loading {tracks_file}...")
    df = pd.read_csv(tracks_file)
    meta = pd.read_csv(meta_file)
    
    print("\n" + "="*50)
    print("1. DHW SENTINEL VS PRECEDING_ID")
    print("="*50)
    # What is the value of dhw when precedingId is 0 (no leader)?
    no_leader_dhw = df[df['precedingId'] == 0]['dhw'].unique()
    print(f"Values of 'dhw' when precedingId == 0: {no_leader_dhw}")
    
    # Is -1.0 ever used when there IS a leader?
    has_leader_neg_dhw = df[(df['precedingId'] != 0) & (df['dhw'] < 0)]
    if len(has_leader_neg_dhw) > 0:
        print(f"WARNING: Found {len(has_leader_neg_dhw)} rows with negative DHW despite having a leader!")
    else:
        print("SUCCESS: Negative DHW perfectly correlates with having no leader.")
        
    print("\n" + "="*50)
    print("2. SURVIVAL RATE (MINIMUM 250 FRAMES)")
    print("="*50)
    # We need 8s (80 steps @ 10Hz) + 1s trim on both sides = 10s total
    # 10 seconds @ 25 Hz = 250 raw frames
    MIN_FRAMES = 250
    
    total_vehicles = len(meta)
    survivors = meta[meta['numFrames'] >= MIN_FRAMES]
    
    print(f"Total vehicles in recording: {total_vehicles}")
    print(f"Vehicles with >= {MIN_FRAMES} frames: {len(survivors)}")
    print(f"Survival Rate: {(len(survivors)/total_vehicles)*100:.2f}%")
    
    # Break down survivors by class
    cars = survivors[survivors['class'] == 'Car']
    trucks = survivors[survivors['class'] == 'Truck']
    print(f"  -> Surviving Cars (Eligible for CAV label): {len(cars)}")
    print(f"  -> Surviving Trucks (Forced HDV): {len(trucks)}")

if __name__ == "__main__":
    # Update paths to your local setup
    TRACKS_PATH = "D:/highD-dataset-v1.0/data/01_tracks.csv"
    META_PATH = "D:/highD-dataset-v1.0/data/01_tracksMeta.csv"
    
    if os.path.exists(TRACKS_PATH) and os.path.exists(META_PATH):
        run_precheck(TRACKS_PATH, META_PATH)
    else:
        print("Files not found. Please check paths.")