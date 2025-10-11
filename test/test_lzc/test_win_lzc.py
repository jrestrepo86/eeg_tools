from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from eeg_tools.lz_complexity.lzc import windowed_lzc

FS = 100
DURATION = 10.0
PARAMS = dict(
    window_size=100,
    window_step=10,
    lz_algorithm="lz76_scheme1",
    alphabet_size=4,
    lzc_norm="scheme1",
)


def generate_signals(
    fs: int = 100, duration: float = 10.0, seed: int = 42
) -> Dict[str, np.ndarray]:
    N = int(fs * duration)
    t = np.arange(N) / fs
    rng = np.random.default_rng(seed)

    # Time-varying entropy signal (piecewise):
    # 0-2s: constant (very low PE)
    # 2-5s: pure sine (low PE)
    # 5-8s: sine + moderate noise (medium PE)
    # 8-10s: white noise (high PE)
    seg1 = np.full(int(2 * fs), 1.0, dtype=float)
    seg2 = np.sin(2 * np.pi * 2.0 * t[: int(3 * fs)])
    seg3 = np.sin(2 * np.pi * 2.0 * t[: int(3 * fs)]) + 0.3 * rng.standard_normal(
        int(3 * fs)
    )
    seg4 = rng.standard_normal(int(2 * fs))
    timevary = np.concatenate([seg1, seg2, seg3, seg4])

    signals = {
        "t": t,
        "constant": np.full(N, 1.0, dtype=float),
        "sine": np.sin(2 * np.pi * 2.0 * t),  # 2 Hz
        "random": rng.standard_normal(N),
        "timevary": timevary,
    }
    return signals


if __name__ == "__main__":
    sigs = generate_signals(fs=FS, duration=DURATION)
    df = pd.DataFrame(
        {
            "t": sigs["t"],
            "constant": sigs["constant"],
            "sine": sigs["sine"],
            "random": sigs["random"],
        }
    )

    LZC_const = windowed_lzc(sigs["constant"], **PARAMS)
    LZC_sine = windowed_lzc(sigs["sine"], **PARAMS)
    LZC_rand = windowed_lzc(sigs["random"], **PARAMS)
    LZC_timevary = windowed_lzc(sigs["timevary"], **PARAMS)

    K = len(LZC_const)
    win_idx = np.arange(K)

    pe_df = pd.DataFrame(
        {
            "window_index": win_idx,
            "constant": LZC_const,
            "sine": LZC_sine,
            "random": LZC_rand,
            "timevary": LZC_timevary,
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=win_idx, y=LZC_const, mode="lines+markers", name="constant")
    )
    fig.add_trace(go.Scatter(x=win_idx, y=LZC_sine, mode="lines+markers", name="sine"))
    fig.add_trace(
        go.Scatter(x=win_idx, y=LZC_rand, mode="lines+markers", name="random")
    )
    fig.add_trace(
        go.Scatter(x=win_idx, y=LZC_timevary, mode="lines+markers", name="time var")
    )

    fig.update_layout(
        title="Windowed LempelZiv Complexity vs Window Index",
        xaxis_title="Window index",
        yaxis_title="LempelZiv Complexity (normalized)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
