import os
import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# Configure these to match your local setup
HIGHD_DATA_DIR = "D:/highD-dataset-v1.0/data"
OUTPUT_DIR = "data_highd_extracted"

def process_recording(tracks_file: str, meta_file: str, output_file: str):
    """Processes a single HighD recording into physical 10Hz windows."""
    print(f"Loading {os.path.basename(tracks_file)}...")
    df = pd.read_csv(tracks_file)
    meta = pd.read_csv(meta_file)

    # BUG 1 FIX: Mirror Direction 2 so all vehicles travel in the +x direction
    MAX_X = df['x'].max()
    dir_map = dict(zip(meta['id'], meta['drivingDirection']))
    class_map = dict(zip(meta['id'], meta['class']))
    directions = df['id'].map(dir_map)

    # Transform x coordinates for Direction 2. This perfectly aligns kinematics
    # with SUMO's unimodal expectation, while preserving relative magnitudes (dhw).
    df.loc[directions == 2, 'x'] = MAX_X - df.loc[directions == 2, 'x']
    # CRITICAL: We must also mirror velocity and acceleration to match the new coordinate frame
    df.loc[directions == 2, 'xVelocity'] *= -1
    df.loc[directions == 2, 'xAcceleration'] *= -1

    print("Phase 1: Fitting independent C² continuous Cubic Splines for all vehicles...")
    splines = {}
    raw_data = {}

    grouped = df.groupby('id')
    
    for veh_id, group in grouped:
        # Ensure chronological order
        group = group.sort_values('frame')
        
        # HighD is 25 Hz. Convert frames to exact seconds.
        t_raw = group['frame'].values / 25.0
        x_raw = group['x'].values
        y_raw = group['y'].values

        # GAP CHECK ASSERTION
        frame_diffs = np.diff(group['frame'].values)
        assert np.all(frame_diffs == 1), f"Discontinuity found in vehicle {veh_id} track!"

        # Cannot fit a cubic spline with fewer than 3 points
        if len(t_raw) < 3:
            continue 

        # Fit independent C^2 continuous splines for position, velocity, and acceleration
        sx = CubicSpline(t_raw, x_raw)
        sy = CubicSpline(t_raw, y_raw)
        svx = CubicSpline(t_raw, group['xVelocity'].values)
        svy = CubicSpline(t_raw, group['yVelocity'].values)
        sax = CubicSpline(t_raw, group['xAcceleration'].values)
        say = CubicSpline(t_raw, group['yAcceleration'].values)

        splines[veh_id] = {
            'sx': sx, 'sy': sy, 
            'svx': svx, 'svy': svy, 
            'sax': sax, 'say': say, 
            't_min': t_raw[0], 't_max': t_raw[-1]
        }

        # Store discrete categorical data for nearest-neighbor lookups later
        raw_data[veh_id] = {
            't': t_raw,
            'precedingId': group['precedingId'].values,
            'followingId': group['followingId'].values,
            'dhw': group['dhw'].values
        }

    print("Phase 2: Resampling, matching, and extracting windows...")
    X_list, Y_list = [], []
    ego_ids, leader_ids, follower_ids = [], [], []

    for veh_id in splines.keys():
        # Ego vehicles MUST be Cars to qualify for our CAV trajectory prediction problem
        if class_map.get(veh_id) != 'Car':
            continue

        t_min = splines[veh_id]['t_min']
        t_max = splines[veh_id]['t_max']

        # Create uniform 10 Hz grid. Ceil to nearest 0.1s to lock onto the global grid.
        t_start = np.ceil(t_min * 10) / 10.0
        t_10hz = np.arange(t_start, t_max, 0.1)

        # Boundary Trim: Shave 1 second (10 frames) off both edges to prevent camera-edge tracking artifacts
        if len(t_10hz) <= 20:
            continue
        t_10hz = t_10hz[10:-10]

        # BUG 3 FIX: Guard against floating-point accumulated drift in np.arange
        assert np.allclose(np.diff(t_10hz), 0.1), f"10Hz grid drift detected for veh {veh_id}"

        # Require at least 80 steps (8 seconds) to extract one valid window
        if len(t_10hz) < 80:
            continue

        sx = splines[veh_id]['sx']
        sy = splines[veh_id]['sy']
        raw_t = raw_data[veh_id]['t']

        # Slice the track into 80-step windows with a stride of 30
        stride = 30
        for w_start in range(0, len(t_10hz) - 80 + 1, stride):
            t_window = t_10hz[w_start : w_start + 80]
            t_input = t_window[:30]
            t_label = t_window[30:]

            # --- Labels (Y) ---
            # 50 steps of future x, y positions
            y_window = np.zeros((50, 2), dtype=np.float32)
            y_window[:, 0] = sx(t_label)
            y_window[:, 1] = sy(t_label)

            # --- Inputs (X) ---
            # 30 steps, 7 columns
            x_window = np.zeros((30, 7), dtype=np.float32)
            lead_ids_window = np.zeros(30, dtype=int)
            foll_ids_window = np.zeros(30, dtype=int)

            for step_idx, t in enumerate(t_input):
                # 1. Ego Kinematics (Sampled from independent pre-smoothed splines)
                vx = splines[veh_id]['svx'](t)
                vy = splines[veh_id]['svy'](t)
                v = np.hypot(vx, vy) # Speed magnitude

                ax = splines[veh_id]['sax'](t)
                ay = splines[veh_id]['say'](t)
                # Signed longitudinal acceleration (dot product of velocity unit vector & acceleration)
                a_long = (vx * ax + vy * ay) / v if v > 1e-3 else 0.0

                x_window[step_idx, 0] = v
                x_window[step_idx, 1] = a_long
                x_window[step_idx, 2] = sx(t)
                x_window[step_idx, 3] = sy(t)

                # 2. Nearest Neighbor Lookup for Categorical IDs & Verified Gaps
                nearest_idx = np.argmin(np.abs(raw_t - t))
                lead_id = int(raw_data[veh_id]['precedingId'][nearest_idx])
                foll_id = int(raw_data[veh_id]['followingId'][nearest_idx])
                dhw = raw_data[veh_id]['dhw'][nearest_idx]

                lead_ids_window[step_idx] = lead_id
                foll_ids_window[step_idx] = foll_id

                # 3. Gap to leader (Using VERIFIED dhw==0.0 exclusivity sentinel)
                if lead_id == 0 or dhw == 0.0:
                    x_window[step_idx, 4] = np.nan
                else:
                    x_window[step_idx, 4] = dhw

                # 4. Leader Speed (Evaluated directly from leader's velocity spline)
                if lead_id != 0 and lead_id in splines:
                    l_t_min = splines[lead_id]['t_min']
                    l_t_max = splines[lead_id]['t_max']
                    # Prevent cubic extrapolation into the void if leader has exited the camera frame
                    if l_t_min <= t <= l_t_max:
                        lvx = splines[lead_id]['svx'](t)
                        lvy = splines[lead_id]['svy'](t)
                        x_window[step_idx, 5] = np.hypot(lvx, lvy)
                    else:
                        x_window[step_idx, 5] = np.nan
                else:
                    x_window[step_idx, 5] = np.nan

                # 5. Follower Speed (Evaluated directly from follower's velocity spline)
                if foll_id != 0 and foll_id in splines:
                    f_t_min = splines[foll_id]['t_min']
                    f_t_max = splines[foll_id]['t_max']
                    if f_t_min <= t <= f_t_max:
                        fvx = splines[foll_id]['svx'](t)
                        fvy = splines[foll_id]['svy'](t)
                        x_window[step_idx, 6] = np.hypot(fvx, fvy)
                    else:
                        x_window[step_idx, 6] = np.nan
                else:
                    x_window[step_idx, 6] = np.nan

            X_list.append(x_window)
            Y_list.append(y_window)
            ego_ids.append(veh_id)
            leader_ids.append(lead_ids_window)
            follower_ids.append(foll_ids_window)

    if len(X_list) > 0:
        temp_path = output_file + ".tmp"
        actual_temp_path = temp_path + ".npz"
        try:
            np.savez_compressed(
                temp_path,
                X_gt=np.array(X_list, dtype=np.float32),
                Y=np.array(Y_list, dtype=np.float32),
                ego_id=np.array(ego_ids, dtype=int),
                leader_id=np.array(leader_ids, dtype=int),
                follower_id=np.array(follower_ids, dtype=int)
            )
            # Atomic swap guarantees no corrupted file is left if interrupted
            shutil.move(actual_temp_path, output_file)
            print(f" -> Success: {len(X_list)} highly-curated windows saved to {os.path.basename(output_file)}")
            return len(X_list)
        except Exception as e:
            print(f" -> ERROR saving modifications to {output_file}: {e}")
            if os.path.exists(actual_temp_path):
                os.remove(actual_temp_path)
            return 0
    else:
        print(" -> No valid windows extracted (all filtered by bounds/duration).")
        return 0

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(HIGHD_DATA_DIR):
        print(f"ERROR: HighD data directory not found at {HIGHD_DATA_DIR}")
        exit(1)

    tracks_files = sorted([f for f in os.listdir(HIGHD_DATA_DIR) if f.endswith("_tracks.csv")])
    total_windows = 0
    
    print("=" * 60)
    print("STARTING HIGHD 25HZ -> 10HZ EXTRACTION PIPELINE")
    print("=" * 60)

    for track_file in tracks_files:
        # e.g., '01_tracks.csv' -> prefix '01'
        prefix = track_file.split('_')[0]
        meta_file = f"{prefix}_tracksMeta.csv"
        
        full_track_path = os.path.join(HIGHD_DATA_DIR, track_file)
        full_meta_path = os.path.join(HIGHD_DATA_DIR, meta_file)
        
        output_filename = f"highd_{prefix}_extracted.npz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # BUG 4 FIX: Fully restartable architecture
        if os.path.exists(output_path):
            print(f"Skipping {prefix}: Already processed.")
            continue
            
        if os.path.exists(full_track_path) and os.path.exists(full_meta_path):
            windows_extracted = process_recording(full_track_path, full_meta_path, output_path)
            total_windows += windows_extracted
        else:
            print(f"WARNING: Missing track or meta file for prefix {prefix}. Skipping.")

    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"Total Physical Windows Extracted: {total_windows}")
    print("=" * 60)