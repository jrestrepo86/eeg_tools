from pathlib import Path
from typing import Dict

import pandas as pd

CHANNELS_MAP = {
    "F1": 3,
    "F3": 4,
    "C3": 5,
    "P3": 6,
    "O1": 7,
    "F7": 8,
    "T3": 9,
    "T5": 10,
    "Fz": 11,
    "Cz": 12,
    "Pz": 13,
    "Oz": 14,
    "T6": 15,
    "T4": 16,
    "F8": 17,
    "O2": 18,
    "P4": 19,
    "C4": 20,
    "F4": 21,
    "F2": 22,
}


class Neutronic:
    def __init__(self, source_file: Path):
        self.source_file = Path(source_file)
        if self.source_file.suffix != ".txt":
            raise ValueError("Neutronic file must be a .txt file")

    def get_meta(self):
        meta: Dict[str, object] = {}
        meta["sampling_rate"] = 65
        return meta

    def get_channels(self):
        return list(CHANNELS_MAP.keys())

    def load_data(self) -> pd.DataFrame:
        raw_data = pd.read_csv(
            self.source_file,
            sep=r"\s+",
            skiprows=2,
            encoding="utf-8",
        )
        data = pd.DataFrame()
        for channel_name, position in CHANNELS_MAP.items():
            data[channel_name] = raw_data.iloc[:, position]

        return data.reset_index(drop=True)


if __name__ == "__main__":
    file = Path(__file__).parent.parent.parent.parent / "data" / "neutronic_data.txt"
    handler = Neutronic(file)
    data = handler.load_data()
    pass
