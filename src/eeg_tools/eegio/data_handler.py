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
            self.channels = handler.get_channels()
            self.meta = handler.get_meta()

    def get_data_by_channel(self, channel: str):
        if channel not in self.channels:
            raise ValueError(f"No channel {channel} found. Channels {self.channels}")
        return self.data[channel]

    def get_sampling_rate(self):
        return self.meta["sampling_rate"]

    def get_series_lenght(self):
        return self.data.shape[0]


if __name__ == "__main__":
    file = Path(__file__).parent.parent.parent.parent / "data" / "eeg_data.edf"
    handler = DataHandler(file, "edf")
    data = handler.data
    meta = handler.meta
    pass
