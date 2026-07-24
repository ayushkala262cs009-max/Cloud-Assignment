import numpy as np
import os
from filterpy.kalman import KalmanFilter

def build_hdv_kf(position_noise_std: float = 0.5, speed_noise_std: float = 0.3) -> KalmanFilter:
    """
    Constructs a 1D Constant-Velocity Kalman Filter to track absolute state.
    State space X = [x_leader, v_leader]^T (Position, Velocity)
    Measurement Z = [z_x, z_v]^T (Absolute Position, Absolute Velocity)
    """
    kf = KalmanFilter(dim_x=2, dim_z=2)
    dt = 0.1  # 10 Hz sampling rate
    
    # State Transition Matrix F: x_t = x_{t-1} + v_{t-1}*dt
    kf.F = np.array([
        [1.0,  dt],
        [0.0, 1.0]
    ], dtype=np.float32)
    
    # Measurement Matrix H: Directly maps states to observations
    kf.H = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float32)
    
    # Measurement Noise Covariance Matrix R (Derived from physical sensors)
    kf.R = np.array([
        [position_noise_std**2,                  0.0],
        [0.0,                  speed_noise_std**2]
    ], dtype=np.float32)
    
    # Process Noise Covariance Matrix Q (Allows for acceleration fluctuations)
    # Using continuous white noise approximation for dt=0.1
    spectral_density = 0.15
    kf.Q = np.array([
        [(dt**3)/3.0, (dt**2)/2.0],
        [(dt**2)/2.0,          dt]
    ], dtype=np.float32) * spectral_density
    
    # Initial Covariance Matrix P (Large initial uncertainty)
    kf.P = np.eye(2, dtype=np.float32) * 10.0
    
    return kf

def sfm_impute_channel1(rsu_obs: np.ndarray, ego_states: np.ndarray) -> tuple:
    """
    Applies the Kalman Filter to impute and smooth Channel 1 features.
    
    Args:
        rsu_obs (np.ndarray): Shape (30, 2) containing [noisy_gap, noisy_leader_speed].
        ego_states (np.ndarray): Shape (30, 4) containing [speed, accel, x, y] for ego.
        
    Returns:
        tuple: (imputed_gaps, imputed_speeds) as shape (30,) arrays.
    """
    imputed_gaps = np.full(30, np.nan, dtype=np.float32)
    imputed_speeds = np.full(30, np.nan, dtype=np.float32)
    
    # Find the index of the first valid RSU detection
    first_idx = next((t for t in range(30) if not np.isnan(rsu_obs[t, 0])), None)
    
    # Case: No detections at all in this window
    if first_idx is None:
        return imputed_gaps, imputed_speeds
        
    kf = build_hdv_kf()
    init_pos = ego_states[first_idx, 2] + rsu_obs[first_idx, 0]  # ego_x + gap = leader_x
    init_vel = rsu_obs[first_idx, 1]                            # leader_speed
    kf.x = np.array([init_pos, init_vel], dtype=np.float32)
    
    # Execute state propagation and measurement updates sequentially
    for t in range(30):
        if t < first_idx:
            continue
            
        # Standard predict step (propagates physics forward)
        kf.predict()
        
        # Check if we have an active observation at this timestep (no missed detection)
        if not np.isnan(rsu_obs[t, 0]):
            obs_pos = ego_states[t, 2] + rsu_obs[t, 0]  # absolute position
            obs_vel = rsu_obs[t, 1]                     # absolute speed
            z = np.array([obs_pos, obs_vel], dtype=np.float32)
            kf.update(z)
            
        imputed_gaps[t] = kf.x[0] - ego_states[t, 2]  # leader_x - ego_x = gap
        imputed_speeds[t] = kf.x[1]
        
    return imputed_gaps, imputed_speeds

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING KALMAN FILTER HANDLER (CHANNEL 1)")
    print("=" * 60)
    
    # Mocking a realistic data window for verification
    mock_ego = np.zeros((30, 4), dtype=np.float32)
    mock_ego[:, 2] = np.linspace(100.0, 190.0, 30)  # ego_x moves from 100 to 190m
    
    # Leader is moving at constant 31.0 m/s with a true gap of 15.0m
    mock_rsu = np.full((30, 2), 15.0, dtype=np.float32)
    mock_rsu[:, 1] = 31.0
    
    # Injecting noise and dropouts (gaps at t=5 to t=10)
    np.random.seed(42)
    mock_rsu += np.random.normal(0, 0.2, (30, 2))
    mock_rsu[5:11, :] = np.nan
    
    gaps, speeds = sfm_impute_channel1(mock_rsu, mock_ego)
    
    print("\nVerification (First 12 Steps):")
    for t in range(12):
        print(f"Step {t:02d} | RSU Gap: {mock_rsu[t, 0]:.2f} | KF Imputed Gap: {gaps[t]:.2f} | Imputed Speed: {speeds[t]:.2f}")
    print("=" * 60)