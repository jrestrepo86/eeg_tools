import numpy as np


def quantil_quantizer(x: np.ndarray, n_quantiles: int) -> np.ndarray:
    samples, dim = x.shape
    flat = x.ravel()
    # α‑quantile thresholds
    q_levels = (np.arange(n_quantiles - 1) + 1) / n_quantiles
    thresh = np.quantile(flat, q=q_levels)
    return np.digitize(flat, thresh, right=True).reshape(samples, dim)


def to_char(x: np.ndarray, offset=65) -> str:
    _, mapped = np.unique(x, return_inverse=True)
    return "".join([chr(s_.item() + offset) for s_ in mapped])


def lzc_normalization(
    lzc: float, alphabet_size: int, sequence_length: int, norm_scheme: str | None = None
):
    if norm_scheme is None:
        return lzc
    elif norm_scheme == "scheme1":
        return (lzc / sequence_length) * np.log(sequence_length)
    if norm_scheme == "scheme2":
        return (lzc / sequence_length) * (
            np.log(sequence_length) / (np.log(alphabet_size))
        )
    elif norm_scheme == "scheme3":
        return (lzc / sequence_length) * (np.log(lzc) / (np.log(alphabet_size)))
    else:
        print("Choose a valid normalization scheme")
        raise ValueError("Unsupported normalization scheme, choose scheme1 or scheme2")
