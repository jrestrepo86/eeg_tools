import math

import numpy as np

from ..utils.embedding import embedding, to_col_vector


def permutation_entropy(
    x: np.ndarray, emb_dim: int = 3, emb_lag: int = 1, normalize: bool = False
) -> float:
    # Embedding
    X = embedding(to_col_vector(x), emb_dim, emb_lag)
    n, order = X.shape

    # Precompute factorial weights
    weights = np.array(
        [math.factorial(order - 1 - i) for i in range(order)], dtype=np.int64
    )

    patterns = np.argsort(X, axis=1, kind="mergesort")

    # Map each row's permutation (lexicographic over 0..m-1) to a Lehmer-code index.
    idx_vals = np.zeros(n, dtype=np.int64)
    for i in range(order):
        # Broadcast compare to the tail
        a = patterns[:, i][:, None]
        b = patterns[:, i + 1 :]
        c = (a > b).sum(axis=1) if b.shape[1] else 0
        idx_vals += c * weights[i]

    M = math.factorial(order)
    counts = np.bincount(idx_vals, minlength=M).astype(float)
    total = counts.sum()

    p = counts / total

    # Shannon entropy with chosen base; handle zero-prob gracefully
    nz = p > 0
    H = -(p[nz] * np.log(p[nz])).sum()

    if normalize:
        H /= np.log(M)

    return H
