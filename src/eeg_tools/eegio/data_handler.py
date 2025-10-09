from pathlib import Path

from .edf_loader import EDF
from .neutronic import Neutronic
from .open_bci import OpenBci

HARDWARE_OPTIONS = ["neutronic", "openbci", "edf"]


class DataHandler:
    def __init__(self, source_file: Path | str, hardware: str):
        self.hardware_source = hardware.lower()
        self.source_file = Path(source_file)
        # chack source file
        if self.source_file.exists():
            self.file_name = self.source_file.stem
        else:
            raise ValueError(f"No source file {source_file} found")

        if self.hardware_source == "neutronic":
            handler = Neutronic(self.source_file)
        elif self.hardware_source == "openbci":
            handler = OpenBci(self.source_file)
        elif self.hardware_source == "edf":
            handler = EDF(self.source_file)
        else:
            handler = None
            raise ValueError(
                f"No hardware {hardware} found. Options {HARDWARE_OPTIONS}"
            )
        if handler is not None:
            self.data = handler.load_data()
            self.channels = handler.set_channels()
            self.meta = handler.set_meta()
            self.sampling_rate = handler.set_sampling_rate()

    def get_data_by_channel(self, channel: str):
        if channel not in self.channels:
            raise ValueError(f"No channel {channel} found. Channels {self.channels}")
        return self.data[channel].to_numpy()

    def get_sampling_rate(self, channel=None) -> float:
        if channel is None:
            return self.sampling_rate[self.channels[0]].values[0]
        else:
            return self.sampling_rate[channel].values[0]

    def get_series_lenght(self):
        return self.data.shape[0]
