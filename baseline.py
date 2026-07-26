import numpy as np
import pandas as pd
import warnings

def zero_fill(X_corrupted: np.ndarray) -> np.ndarray:
    """
    Replaces all NaNs with 0.0.
    This is the most naive baseline, equivalent to assuming the feature
    value instantly drops to the dataset mean (in z-score space) or zero.
    """
    return np.nan_to_num(X_corrupted, nan=0.0)

def forward_fill(X_corrupted: np.ndarray) -> np.ndarray:
    """
    Carries the last observed value forward.
    Uses backward-fill for leading NaNs, and zero-fill for completely empty columns.
    """
    N, T, F = X_corrupted.shape
    X_imputed = np.empty_like(X_corrupted)

    for i in range(N):
        df = pd.DataFrame(X_corrupted[i])
        # ffill carries forward, bfill catches leading NaNs, fillna(0.0) catches empty cols
        X_imputed[i] = df.ffill().bfill().fillna(0.0).values

    return X_imputed

def mean_fill(X_corrupted: np.ndarray) -> np.ndarray:
    """
    Fills NaNs with the mean of the observed values within that specific 30-step window.
    If the entire column is NaN, falls back to 0.0.
    """
    N, T, F = X_corrupted.shape
    X_imputed = np.empty_like(X_corrupted)

    # We suppress Mean of empty slice warnings that occur when an entire column is NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for i in range(N):
            # Compute mean along the time axis (axis=0 for this 2D slice)
            col_means = np.nanmean(X_corrupted[i], axis=0)

            # Identify where NaNs are in this specific window
            nan_mask = np.isnan(X_corrupted[i])

            # Broadcast means to the shape of the window
            mean_matrix = np.tile(col_means, (T, 1))

            # Replace NaNs with the corresponding column mean
            imputed_window = np.where(nan_mask, mean_matrix, X_corrupted[i])

            # Any remaining NaNs are due to empty columns (mean was NaN). Fill with 0.0.
            X_imputed[i] = np.nan_to_num(imputed_window, nan=0.0)

    return X_imputed

def linear_interp(X_corrupted: np.ndarray) -> np.ndarray:
    """
    Draws straight lines between observed points.
    Extrapolates to the boundaries. Falls back to zero for empty columns.
    """
    N, T, F = X_corrupted.shape
    X_imputed = np.empty_like(X_corrupted)

    for i in range(N):
        df = pd.DataFrame(X_corrupted[i])
        # limit_direction='both' ensures boundaries are extrapolated based on the nearest slope
        X_imputed[i] = df.interpolate(method='linear', limit_direction='both').fillna(0.0).values

    return X_imputed

def spline_interp(X_corrupted: np.ndarray) -> np.ndarray:
    """
    Fits a 2nd-order spline (polynomial curve) between observed points for smooth kinematic
    transitions. Cascades down to linear, then zero-fill if points are too sparse.
    """
    N, T, F = X_corrupted.shape
    X_imputed = np.empty_like(X_corrupted)

    for i in range(N):
        df = pd.DataFrame(X_corrupted[i])

        try:
            # Spline order 2 requires at least 3 points.
            # Often fails or outputs extreme values on sparse data.
            window = df.interpolate(method='spline', order=2, limit_direction='both')

            # If spline left NaNs (because it couldn't process a sparse column)
            if window.isna().any().any():
                window = window.interpolate(method='linear', limit_direction='both')

            X_imputed[i] = window.fillna(0.0).values

        except Exception:
            # If spline completely crashes, cascade directly to linear + zero
            X_imputed[i] = df.interpolate(method='linear', limit_direction='both').fillna(0.0).values

    return X_imputed
