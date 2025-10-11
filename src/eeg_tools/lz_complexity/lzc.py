import numpy as np

from ..utils.utils import split_by_windows, to_col_vector
from .lzc_algorithms import lz_complexity_76_scheme1, lz_complexity_76_scheme2
from .lzc_utils import lzc_normalization, quantil_quantizer, to_char


def lzc(
    x: list | np.ndarray,
    lz_algorithm: str = "lz76_scheme1",
    alphabet_size: int = 2,
    lzc_norm: str | None = "scheme1",
) -> float:
    if isinstance(x, list):
        x = np.array(x)

    if x.size == 0:
        return 0

    x = to_col_vector(x)

    # quantization
    x_q = quantil_quantizer(x, n_quantiles=alphabet_size)
    # transform to string
    x_str = to_char(x_q)

    # Calculate lzc
    if lz_algorithm == "lz76_scheme1":
        lzc = lz_complexity_76_scheme1(x_str)
    elif lz_algorithm == "lz76_scheme2":
        lzc = lz_complexity_76_scheme2(x_str)
    else:
        print("Choose a valid algorithm")
        raise ValueError("Unsupported algorithm, choose lz76_scheme1 or lz76_scheme2")

    # Lzc normalization
    lzc_n = lzc_normalization(lzc, alphabet_size, x.size, lzc_norm)

    return lzc_n


def windowed_lzc(
    x: np.ndarray,
    window_size: int,
    window_step: int,
    lz_algorithm: str = "lz76_scheme1",
    alphabet_size: int = 2,
    lzc_norm: str | None = "scheme1",
):
    windows = split_by_windows(x, window_size, window_step)
    return np.array([lzc(w, lz_algorithm, alphabet_size, lzc_norm) for w in windows])
