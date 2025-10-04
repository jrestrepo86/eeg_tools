from pathlib import Path

import pandas as pd

META = {"fz": None}


class OpenBci:
    def __init__(self, source_file: Path):
        # Define the EEG channel labels in the specified order
        self._check_source_file(source_file)
        self.channels = [
            "F1",
            "F3",
            "C3",
            "P3",
            "O1",
            "F7",
            "T3",
            "T5",
            "Fz",
            "Cz",
            "Pz",
            "Oz",
            "T6",
            "T4",
            "F8",
            "O2",
            "P4",
            "C4",
            "F4",
            "F2",
        ]
        self.meta = META

    def _check_source_file(self, source_file: Path):
        if source_file.suffix == "txt":
            self.source_file = source_file
        else:
            raise ValueError("OpenBCI file is not a .txt file")

    def load_data(self) -> pd.DataFrame:
        return pd.DataFrame()
