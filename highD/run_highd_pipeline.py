import os
import subprocess
import sys

def main():
    print("=" * 60)
    print("EXECUTING PHASE 3 PIPELINE ON HIGHD DATASET")
    print("=" * 60)

    # 1. Inject TCM/SFM Corruption into an isolated HighD directory
    print("\n[1/4] Injecting TCM/SFM Corruption...")
    subprocess.run([sys.executable, "inject_corruption.py",
                    "--input-dir", "data_highd",
                    "--output-dir", "data_highd_corrupted"], check=True)
    
    # 2. Simulate RSU (Channel 1 Observation)
    print("\n[2/4] Simulating RSU Cameras...")
    subprocess.run([sys.executable, "simulate_rsu.py",
                    "--data-dir", "data_highd_corrupted"], check=True)

    # 3. Channel 1 (Kalman Filter)
    print("\n[3/4] Executing Channel 1 (Kalman Filter)...")
    subprocess.run([sys.executable, "integrate_ch1.py",
                    "--data-dir", "data_highd_corrupted"], check=True)

    # 4. Channel 2 (GRU-D Inference)
    print("\n[4/4] Executing Channel 2 (GRU-D Dual Integration)...")
    subprocess.run([sys.executable, "integrate_ch2.py",
                    "--data-dir", "data_highd_corrupted"], check=True)

    print("\n" + "=" * 60)
    print("HIGHD PIPELINE EXECUTION COMPLETE")
    print("The corrupted directory is now fully staged with HighD data.")
    print("Ready for Step 11 evaluation scripts!")
    print("=" * 60)

if __name__ == "__main__":
    main()