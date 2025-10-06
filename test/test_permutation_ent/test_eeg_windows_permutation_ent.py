from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from eeg_tools.eegio.data_handler import DataHandler
from eeg_tools.entropy.permutation_entropy import windowed_permutation_entropy
from eeg_tools.utils.utils import embedding

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
        title=title or "Permutation Entropy vs Time",
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
    emb_dim = 3
    emb_lag = 1

    handler = DataHandler(NEUTRONIC_FILE, hardware="neutronic")
    data = handler.data
    fs = handler.get_sampling_rate()
    data_samples = handler.get_series_lenght()

    pent_df = pd.DataFrame()
    for channel in handler.channels:
        pent = windowed_permutation_entropy(
            data[channel].to_numpy(),
            window_size=window_size,
            window_step=window_step,
            emb_dim=emb_dim,
            emb_lag=emb_lag,
            normalize=True,
        )
        pent_df[channel] = pent

    pent = pent_df.reset_index(drop=True)
    n_windows = (data_samples - window_size) // window_step + 1
    centers = (np.arange(n_windows) * window_step) + (window_size - 1) / 2.0

    title = f"Neutronic PEntropy | data samples= {data_samples}, sampling-rate={fs} , window_size={window_size}, window_step={window_step}, emb_dim={emb_dim}, emb_lag={emb_lag} "
    fig = generate_plot(pent, centers, title=title)

    fig.show()


def openbci_pentropy():
    window_size = 128
    window_step = 64
    emb_dim = 3
    emb_lag = 1

    handler = DataHandler(OPENBCI_FILE, hardware="openbci")
    data = handler.data
    fs = handler.get_sampling_rate()
    data_samples = handler.get_series_lenght()

    pent_df = pd.DataFrame()
    for channel in handler.channels:
        pent = windowed_permutation_entropy(
            data[channel].to_numpy(),
            window_size=window_size,
            window_step=window_step,
            emb_dim=emb_dim,
            emb_lag=emb_lag,
            normalize=False,
        )
        pent_df[channel] = pent

    pent = pent_df.reset_index(drop=True)

    n_windows = (data_samples - window_size) // window_step + 1
    centers = (np.arange(n_windows) * window_step) + (window_size - 1) / 2.0

    title = f"OpenBCI PEntropy | data samples= {data_samples}, sampling-rate={fs}, window_size={window_size}, window_step={window_step}, emb_dim={emb_dim}, emb_lag={emb_lag} "
    fig = generate_plot(pent, centers, title=title)

    fig.show()


if __name__ == "__main__":
    neutronic_pentropy()
    openbci_pentropy()
