import os
import traci
import numpy as np

WINDOW = 30
LABEL_HORIZON = 50
REQUIRED_STEPS = WINDOW + LABEL_HORIZON  # 80 continuous frames
RSU_CENTER = 500.0
RSU_RADIUS = 300.0

def extract_trajectory_windows(route_file: str, seed: int, output_path: str) -> int:
    """
    Runs a single SUMO simulation instance, extracts continuous CAV trajectories,
    slices them into completely independent inputs and labels, and packs them into an NPZ archive.
    
    Returns:
        int: The number of valid windows saved.
    """
    active_episodes = {}
    completed_episodes = []

    dataset_X = []
    dataset_Y = []
    meta_leader = []
    meta_follower = []

    # Initialize TraCI with custom overrides for the route distribution and deterministic seed
    traci.start([
        'sumo', 
        '-c', 'highway.sumocfg', 
        '--route-files', route_file, 
        '--seed', str(seed)
    ])

    for step in range(6000):  # 600 seconds at 10 Hz
        traci.simulationStep()
        active_vehicles = traci.vehicle.getIDList()
        
        # Flush trajectories of vehicles that exited the network entirely mid-simulation
        for old_id in list(active_episodes.keys()):
            if old_id not in active_vehicles:
                if len(active_episodes[old_id]) >= REQUIRED_STEPS:
                    completed_episodes.append(active_episodes[old_id])
                del active_episodes[old_id]

        for veh_id in active_vehicles:
            x = traci.vehicle.getPosition(veh_id)[0]
            
            # Spatial Boundary & V2X Capabilities Verification
            if abs(x - RSU_CENTER) > RSU_RADIUS or traci.vehicle.getTypeID(veh_id) != "CAV":
                if veh_id in active_episodes:
                    if len(active_episodes[veh_id]) >= REQUIRED_STEPS:
                        completed_episodes.append(active_episodes[veh_id])
                    del active_episodes[veh_id]
                continue

            leader = traci.vehicle.getLeader(veh_id)
            follower = traci.vehicle.getFollower(veh_id)

            has_leader = leader and leader[0] != ""
            has_follower = follower and follower[0] != ""

            snap = {
                'ego_speed': traci.vehicle.getSpeed(veh_id),
                'ego_accel': traci.vehicle.getAcceleration(veh_id),
                'ego_x': x,
                'ego_y': traci.vehicle.getPosition(veh_id)[1],
                'gap_to_leader': leader[1] if has_leader else np.nan,
                'leader_speed': traci.vehicle.getSpeed(leader[0]) if has_leader else np.nan,
                'follower_speed': traci.vehicle.getSpeed(follower[0]) if has_follower else np.nan,
                'leader_type': traci.vehicle.getTypeID(leader[0]) if has_leader else 'none',
                'follower_type': traci.vehicle.getTypeID(follower[0]) if has_follower else 'none'
            }

            if veh_id not in active_episodes:
                active_episodes[veh_id] = []
                
            active_episodes[veh_id].append(snap)

    # Process remaining vehicles that were still active at step 6000
    for veh_id, episode in active_episodes.items():
        if len(episode) >= REQUIRED_STEPS:
            completed_episodes.append(episode)

    traci.close()

    # Matrix Extraction & Independent Slicing Loop
    for episode in completed_episodes:
        n_steps = len(episode)
        
        # Step by REQUIRED_STEPS to ensure that the label sequences (future states)
        # never bleed into the input features of a subsequent window.
        for i in range(0, n_steps - REQUIRED_STEPS + 1, REQUIRED_STEPS):
            input_slice = episode[i : i + WINDOW]
            label_slice = episode[i + WINDOW : i + REQUIRED_STEPS]

            x_features = np.array([[
                s['ego_speed'], s['ego_accel'], s['ego_x'], s['ego_y'],
                s['gap_to_leader'], s['leader_speed'], s['follower_speed']
            ] for s in input_slice])

            y_targets = np.array([[
                s['ego_x'], s['ego_y']
            ] for s in label_slice])

            dataset_X.append(x_features)
            dataset_Y.append(y_targets)
            
            # Save metadata as clean parallel flat lists of strings for conversion to clean matrices
            meta_leader.append([s['leader_type'] for s in input_slice])
            meta_follower.append([s['follower_type'] for s in input_slice])

    # Atomic saving block to eliminate incomplete records
    if dataset_X:
        np.savez_compressed(
            output_path,
            X=np.array(dataset_X, dtype=np.float32),
            Y=np.array(dataset_Y, dtype=np.float32),
            leader_meta=np.array(meta_leader),  # Structured matrix shape: (N, 30)
            follower_meta=np.array(meta_follower)  # Structured matrix shape: (N, 30)
        )
        return len(dataset_X)
    else:
        return 0


if __name__ == "__main__":
    scenarios = [
        {"route": "traffic_10pct.rou.xml", "dir": "data/penetration_10"},
        {"route": "mixed_traffic.rou.xml", "dir": "data/penetration_20"},
        {"route": "traffic_30pct.rou.xml", "dir": "data/penetration_30"},
    ]
    
    seeds = range(1, 21)
    running_window_total = 0
    current_run_count = 0
    total_runs = len(scenarios) * len(seeds)

    print("=" * 60)
    print("STARTING DETERMINISTIC MATRIX PIPELINE SWEEP")
    print("=" * 60)

    for run_idx, config in enumerate(scenarios):
        for seed in seeds:
            current_run_count += 1
            output_file = os.path.join(config["dir"], f"run_seed_{seed}.npz")
            
            # Restartable capability check: Skip existing runs
            if os.path.exists(output_file):
                try:
                    with np.load(output_file) as data:
                        existing_count = len(data['X'])
                    running_window_total += existing_count
                    print(f"[{current_run_count}/{total_runs}] SKIPPING: {output_file} already exists ({existing_count} windows).")
                except Exception:
                    print(f"[{current_run_count}/{total_runs}] CORRUPT FILE DETECTED: Removing {output_file} to force re-run.")
                    try:
                        os.remove(output_file)
                    except OSError as e:
                        print(f" -> Warning: Could not delete corrupt file ({e}). Attempting overwrite anyway.")
                else:
                    continue

            print(f"[{current_run_count}/{total_runs}] Executing Seed {seed} on {config['route']}...", end="", flush=True)
            
            # Execute simulation instance
            windows_extracted = extract_trajectory_windows(
                route_file=config["route"],
                seed=seed,
                output_path=output_file
            )
            
            if windows_extracted > 0:
                running_window_total += windows_extracted
                print(f" SUCCESS. Extracted: {windows_extracted} | Running Total: {running_window_total}")
            else:
                # Log failed run metrics to prevent silent execution gaps
                print(" FAILED (0 Windows).")
                with open("failed_runs.txt", "a") as f:
                    f.write(f"Scenario: {config['route']}, Seed: {seed}, OutputPath: {output_file} - Reason: 0 Windows Extracted\n")

    print("\n" + "=" * 60)
    print("PIPELINE PROCESSING COMPLETE")
    print(f"Total Cumulative Samples Gathered: {running_window_total}")
    if os.path.exists("failed_runs.txt"):
        print("WARNING: Some runs produced 0 data points. Check failed_runs.txt for details.")
    print("=" * 60)