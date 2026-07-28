import pandas as pd

for rec_id in ['07', '25']:
    meta = pd.read_csv(f"D:/highD-dataset-v1.0/data/{rec_id}_recordingMeta.csv")
    print(f"\n--- Recording {rec_id} ---")
    print(meta[['frameRate', 'locationId', 'speedLimit', 'duration',
                 'numVehicles', 'numCars', 'numTrucks',
                 'upperLaneMarkings', 'lowerLaneMarkings']].to_string(index=False))