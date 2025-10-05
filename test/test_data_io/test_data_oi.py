from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from eeg_tools.eegio.data_handler import DataHandler

HARDWARE = "neutronic"
OPENBCI_FILE = (
    Path(__file__).parent.parent.parent / "data" / "OpenBCI-RAW-2020-01-20_19-25-40.txt"
)

NEUTRONIC_FILE = Path(__file__).parent.parent.parent / "data" / "neutronic_data.txt"


def generate_plot(handler, title: str | None = None) -> go.Figure:
    df = handler.data
    channels = handler.channels
    fs = handler.get_sampling_rate()  # Hz
    if fs is None or fs <= 0:
        raise ValueError(f"Invalid sampling rate from meta: {fs}")
    t = np.arange(len(df), dtype=float) / float(fs)  # time in seconds, starts at 0
    fig = go.Figure()
    for ch in channels:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=df[ch].to_numpy(),
                mode="lines",
                name=ch,
            )
        )

    fig.update_layout(
        title=title or "EEG Channels vs Time",
        xaxis_title="Time [s]",
        yaxis_title="EEG amplitude (raw units)",
        legend_title="Channels",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def neutronic():
    neutronic_handler = DataHandler(NEUTRONIC_FILE, hardware="neutronic")
    fig = generate_plot(neutronic_handler, title="Neutronic Channels vs Time")
    fig.show()


def openbci():
    openbci_handler = DataHandler(OPENBCI_FILE, hardware="openbci")
    fig = generate_plot(openbci_handler, title="OpenBCI Channels vs Time")
    fig.show()


if __name__ == "__main__":
    neutronic()
    openbci()
