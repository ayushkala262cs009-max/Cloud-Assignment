import pickle
import traci
import numpy as np

WINDOW = 30
LABEL_HORIZON = 50
REQUIRED_STEPS = WINDOW + LABEL_HORIZON  # 80 steps minimum to extract 1 sample
RSU_CENTER = 500.0
RSU_RADIUS = 300.0

# Tracks active vehicles inside the RSU zone
active_episodes = {}
# Stores finished trajectories that are ready for slicing
completed_episodes = []

# Diagnostic Telemetry Counters
total_episodes_seen = 0
total_episodes_discarded = 0

dataset_X = []
dataset_Y = []
type_metadata = []

traci.start(['sumo', '-c', 'highway.sumocfg'])

for step in range(6000):  # 600 seconds at 10 Hz
    traci.simulationStep()
    active_vehicles = traci.vehicle.getIDList()
    
    # Handle vehicles that completely left the simulation loop
    for old_id in list(active_episodes.keys()):
        if old_id not in active_vehicles:
            total_episodes_seen += 1
            if len(active_episodes[old_id]) >= REQUIRED_STEPS:
                completed_episodes.append(active_episodes[old_id])
            else:
                total_episodes_discarded += 1
            del active_episodes[old_id]

    for veh_id in active_vehicles:
        x = traci.vehicle.getPosition(veh_id)[0]
        
        # Spatial Filter & V2X Type Filter
        if abs(x - RSU_CENTER) > RSU_RADIUS or traci.vehicle.getTypeID(veh_id) != "CAV":
            # If it just crossed out of the RSU, flush its data and remove from active tracking
            if veh_id in active_episodes:
                total_episodes_seen += 1
                if len(active_episodes[veh_id]) >= REQUIRED_STEPS:
                    completed_episodes.append(active_episodes[veh_id])
                else:
                    total_episodes_discarded += 1
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

# Append any remaining active episodes that met the length criteria at step 6000
for veh_id, episode in active_episodes.items():
    total_episodes_seen += 1
    if len(episode) >= REQUIRED_STEPS:
        completed_episodes.append(episode)
    else:
        total_episodes_discarded += 1

traci.close()

# --- DIAGNOSTIC REPORT ---
print("\n" + "="*40)
print("TRAJECTORY EXTRACTION DIAGNOSTICS")
print("="*40)
print(f"Total CAV RSU Episodes Registered: {total_episodes_seen}")
print(f"Episodes Discarded (< {REQUIRED_STEPS} steps): {total_episodes_discarded}")
print(f"Episodes Kept for Post-Processing:  {len(completed_episodes)}")
print("="*40)

print(f"\nProcessing {len(completed_episodes)} valid trajectories into sliding windows...")

# --- POST-PROCESSING PHASE ---
for episode in completed_episodes:
    n_steps = len(episode)

    # Slice into distinct, non-overlapping input windows with their respective futures
    for i in range(0, n_steps - REQUIRED_STEPS + 1, WINDOW):
        input_slice = episode[i : i + WINDOW]
        label_slice = episode[i + WINDOW : i + REQUIRED_STEPS]

        # Extract features for X: shape (30, 7)
        x_features = np.array([[
            s['ego_speed'], s['ego_accel'], s['ego_x'], s['ego_y'],
            s['gap_to_leader'], s['leader_speed'], s['follower_speed']
        ] for s in input_slice])

        # Extract targets for Y: shape (50, 2)
        y_targets = np.array([[
            s['ego_x'], s['ego_y']
        ] for s in label_slice])

        dataset_X.append(x_features)
        dataset_Y.append(y_targets)
        
        type_metadata.append({
            'leader': [s['leader_type'] for s in input_slice],
            'follower': [s['follower_type'] for s in input_slice]
        })

# --- ARTIFACT SERIALIZATION ---
if dataset_X:
    X_arr = np.array(dataset_X)
    Y_arr = np.array(dataset_Y)
    
    np.save('ground_truth_X.npy', X_arr)
    np.save('ground_truth_Y.npy', Y_arr)
    
    with open('type_metadata.pkl', 'wb') as f:
        pickle.dump(type_metadata, f)
    
    print("\nDataset generation complete!")
    print(f"Inputs (X) Shape:       {X_arr.shape}")
    print(f"Labels (Y) Shape:       {Y_arr.shape}")
    print(f"Metadata Records:       {len(type_metadata)}")
else:
    print(f"\nError: Zero valid datasets generated. Of the {total_episodes_seen} episodes seen, "
          f"100% were discarded because they failed to meet the {REQUIRED_STEPS}-step threshold.")