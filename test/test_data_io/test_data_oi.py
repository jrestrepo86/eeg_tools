from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from eeg_tools.eegio.data_handler import DataHandler

OPENBCI_FILE = (
    Path(__file__).parent.parent.parent / "data" / "OpenBCI-RAW-2020-01-20_19-25-40.txt"
)

NEUTRONIC_FILE = Path(__file__).parent.parent.parent / "data" / "neutronic_data.txt"

EDF_FILE = file = Path(__file__).parent.parent.parent / "data" / "eeg_data.edf"


def generate_plot(handler, title: str | None = None) -> go.Figure:
    channels = handler.channels
    fig = go.Figure()
    for ch in channels:
        fs = handler.get_sampling_rate(ch)  # Hz
        data = handler.get_data_by_channel(ch)
        if data.size > 10000:
            data = data[:10000]
        t = np.arange(data.size, dtype=float) / fs  # time in seconds, starts at 0
        fig.add_trace(
            go.Scatter(
                x=t,
                y=data,
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


def edf():
    edf_handler = DataHandler(EDF_FILE, hardware="edf")
    fig = generate_plot(edf_handler, title="EDF Channels vs Time")
    fig.show()


if __name__ == "__main__":
    # neutronic()
    # openbci()
    edf()
