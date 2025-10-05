import numpy as np
from numpy.lib.stride_tricks import as_strided


def to_col_vector(x: np.ndarray) -> np.ndarray:
    """
    Convert a 1D array to a column vector.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Column vector.
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x


def series_normalization(x: np.ndarray) -> np.ndarray:
    """
    Normalize the input series to zero mean and unit variance, then add small noise.

    Args:
        x (np.ndarray): Input series.

    Returns:
        np.ndarray: Normalized series with added noise.
    """
    x = to_col_vector(x).reshape(-1)  # flatten to 1D
    x = (x - np.mean(x)) / np.std(x)
    x = x + 1e-10 * np.random.rand(*x.shape)
    return x


def embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Perform Takens embedding for single-channel signals using stride tricks.

    Args:
        x (np.ndarray): Input signal (1D array or column vector).
        m (int): Embedding dimension (number of delay coordinates).
        tau (int): Time delay (number of samples between coordinates).

    Returns:
        np.ndarray: 2D array of shape (N_emb, m) containing embedded vectors.

    Raises:
        ValueError: If the signal is too short for the given embedding parameters.
    """
    x = to_col_vector(x).reshape(-1)  # flatten to 1D
    x = np.ascontiguousarray(x)  # ensure contiguous

    n_samples = x.shape[0]
    min_length = (m - 1) * tau + 1

    if n_samples < min_length:
        raise ValueError(
            f"Signal too short: Need {min_length} samples, got {n_samples}"
        )

    # Create a memory-efficient strided view
    itemsize = x.strides[0]
    strides = (
        itemsize,  # Step between embedded vectors
        tau * itemsize,  # Step between coordinates
    )

    embedded = as_strided(
        x, shape=(n_samples - (m - 1) * tau, m), strides=strides, writeable=False
    )

    return embedded.copy()  # Return a contiguous array for safe usage


def split_by_windows(x: np.ndarray, window_size: int, window_step: int) -> np.ndarray:
    """
    Split a 1-D (or single-column) time-series into overlapping (or disjoint) windows.

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (n,) or (n,1).
    window_size : int
        Number of samples in each window. Must be >= 1.
    window_step : int
        Step (stride) between consecutive windows (in samples). Must be >= 1.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, window_size)
        A 2-D array where each row is a contiguous window of the original series.

    Notes
    -----
    - If `window_step < window_size` the windows overlap.
    - Only **complete** windows are returned (no padding).
    - If `x` is shorter than `window_size` the function returns an empty array
      with shape (0, window_size).
    - Input is reshaped/flattened to 1-D before processing.
    """
    x = to_col_vector(x).reshape(-1)  # flatten to 1D
    n = x.size

    if window_size <= 0 or window_step <= 0:
        raise ValueError("window_size and window_step must be positive integers")

    if n < window_size:
        # no full window fits
        return np.empty((0, window_size), dtype=x.dtype)

    # compute start indices of each window
    starts = np.arange(0, n - window_size + 1, window_step)
    n_windows = starts.size

    # use stride_tricks for efficiency
    from numpy.lib.stride_tricks import as_strided

    stride = x.strides[0]
    windows = as_strided(
        x,
        shape=(n_windows, window_size),
        strides=(stride * window_step, stride),
        writeable=False,
    )

    return windows.copy()  # return a safe copy
