import os

CORRUPTED_DIR = "data_corrupted"
SCENARIOS = ["penetration_10", "penetration_20", "penetration_30"]

deleted_count = 0

print("=" * 60)
print("SCANNING FOR ORPHANED TEMPORARY ARCHIVES")
print("=" * 60)

for scenario in SCENARIOS:
    path = os.path.join(CORRUPTED_DIR, scenario)
    if not os.path.exists(path):
        continue
        
    for f in os.listdir(path):
        if f.endswith(".tmp.npz"):
            full = os.path.join(path, f)
            try:
                os.remove(full)
                print(f" -> Deleted: {full}")
                deleted_count += 1
            except OSError as e:
                print(f" -> ERROR deleting {full}: {e}")

print("=" * 60)
print(f"CLEANUP COMPLETE. Total files deleted: {deleted_count}")
print("=" * 60)