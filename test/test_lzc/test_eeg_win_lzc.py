from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from eeg_tools.eegio.data_handler import DataHandler
from eeg_tools.lz_complexity.lzc import windowed_lzc

OPENBCI_FILE = (
    Path(__file__).parent.parent.parent / "data" / "OpenBCI-RAW-2020-01-20_19-25-40.txt"
)
NEUTRONIC_FILE = Path(__file__).parent.parent.parent / "data" / "neutronic_data.txt"


def generate_plot(pent, window_centers, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    channels = pent.columns

    for ch in channels:
        fig.add_trace(
            go.Scatter(
                x=window_centers,
                y=pent[ch],
                mode="lines+markers",
                name=ch,
            )
        )

    fig.update_layout(
        title=title or "LZ Complexity vs Windows",
        xaxis_title="Windows",
        yaxis_title="Permutation Entropy",
        legend_title="Channels",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def neutronic_pentropy():
    window_size = 64
    window_step = 32
    lz_algorithm = "lz76_scheme1"
    alphabet_size = 4
    lzc_norm = "scheme1"

    handler = DataHandler(NEUTRONIC_FILE, hardware="neutronic")
    data = handler.data
    fs = handler.get_sampling_rate()
    data_samples = handler.get_series_lenght()

    lzc_df = pd.DataFrame()
    for channel in handler.channels:
        lzc = windowed_lzc(
            data[channel].to_numpy(),
            window_size=window_size,
            window_step=window_step,
            lz_algorithm=lz_algorithm,
            alphabet_size=alphabet_size,
            lzc_norm=lzc_norm,
        )
        lzc_df[channel] = lzc

    lzc = lzc_df.reset_index(drop=True)
    n_windows = (data_samples - window_size) // window_step + 1
    centers = (np.arange(n_windows) * window_step) + (window_size - 1) / 2.0

    title = f"Neutronic LZC | data samples= {data_samples}, sampling-rate={fs} , window_size={window_size}, window_step={window_step}, emb_dim={emb_dim}, emb_lag={emb_lag} "
    fig = generate_plot(lzc, centers, title=title)

    fig.show()


def openbci_pentropy():
    window_size = 128
    window_step = 64
    lz_algorithm = "lz76_scheme1"
    alphabet_size = 4
    lzc_norm = "scheme1"

    handler = DataHandler(OPENBCI_FILE, hardware="openbci")
    data = handler.data
    fs = handler.get_sampling_rate()
    data_samples = handler.get_series_lenght()

    lzc_df = pd.DataFrame()
    for channel in handler.channels:
        pent = windowed_lzc(
            data[channel].to_numpy(),
            window_size=window_size,
            window_step=window_step,
            lz_algorithm=lz_algorithm,
            alphabet_size=alphabet_size,
            lzc_norm=lzc_norm,
        )
        lzc_df[channel] = pent

    pent = lzc_df.reset_index(drop=True)

    n_windows = (data_samples - window_size) // window_step + 1
    centers = (np.arange(n_windows) * window_step) + (window_size - 1) / 2.0

    title = f"OpenBCI LZC | data samples= {data_samples}, sampling-rate={fs}, window_size={window_size}, window_step={window_step}, emb_dim={emb_dim}, emb_lag={emb_lag} "
    fig = generate_plot(pent, centers, title=title)

    fig.show()


if __name__ == "__main__":
    neutronic_pentropy()
    openbci_pentropy()
