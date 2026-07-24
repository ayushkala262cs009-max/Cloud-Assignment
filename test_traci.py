import traci

sumo_cmd = [
    "sumo",
    "--net-file", "highway.net.xml",
    "--route-files", "mixed_traffic.rou.xml",
    "--step-length", "0.1",
    "--no-warnings",
    "--no-step-log"
]

traci.start(sumo_cmd)

for step in range(200):
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()

    if step == 100:  # snapshot at 10 seconds
        print(f"Snapshot at step {step} — {len(vehicles)} vehicles\n")
        for veh_id in list(vehicles)[:5]:  # first 5 only
            x, y      = traci.vehicle.getPosition(veh_id)
            speed     = traci.vehicle.getSpeed(veh_id)
            accel     = traci.vehicle.getAcceleration(veh_id)
            vtype     = traci.vehicle.getTypeID(veh_id)
            leader    = traci.vehicle.getLeader(veh_id)
            follower  = traci.vehicle.getFollower(veh_id)

            print(f"  Vehicle : {veh_id}")
            print(f"  Type    : {vtype}")
            print(f"  Position: x={x:.2f}m, y={y:.2f}m")
            print(f"  Speed   : {speed:.2f} m/s")
            print(f"  Accel   : {accel:.2f} m/s²")
            print(f"  Leader  : {leader}")
            print(f"  Follower: {follower}")
            print()

traci.close()
print("Done.")