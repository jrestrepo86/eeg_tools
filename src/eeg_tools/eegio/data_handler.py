from pathlib import Path

from .neutronic import Neutronic
from .open_bci import OpenBci

HARDWARE_OPTIONS = ["neutronic", "openbci"]


class DataHandler:
    def __init__(self, source_file: str, hardware: str):
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
        else:
            handler = None
            raise ValueError(
                f"No hardware {hardware} found. Options {HARDWARE_OPTIONS}"
            )
        if handler is not None:
            self.data = handler.load_data()
            self.channels = handler.channels
            self.meta = handler.meta
