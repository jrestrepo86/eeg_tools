import re
from pathlib import Path
from typing import Dict

import pandas as pd

CHANNELS_MAP = {
    "FP1": 1,
    "FP2": 2,
    "F7": 3,
    "F3": 4,
    "F4": 5,
    "F8": 6,
    "T7": 7,
    "C3": 8,
    "C4": 9,
    "T8": 10,
    "P7": 11,
    "P3": 12,
    "P4": 13,
    "P8": 14,
    "O1": 15,
    "O2": 16,
}


class OpenBci:
    def __init__(self, source_file: Path | str):
        self.source_file = Path(source_file)
        if self.source_file.suffix != ".txt":
            raise ValueError("OpenBCI file must be a .txt file")

    def get_meta(self):
        meta: Dict[str, object] = {}
        with open(self.source_file, "r", encoding="utf-8") as file:
            header = [next(file) for _ in range(5)]

        fs = float(re.search(r"=\s*([0-9.]+)", header[2]).group(1))
        meta["sampling_rate"] = fs
        meta["header"] = "\n".join(header)
        return meta

    def get_channels(self):
        return list(CHANNELS_MAP.keys())

    def load_data(self) -> pd.DataFrame:
        raw_data = pd.read_csv(
            self.source_file,
            sep=",",
            skiprows=6,
            encoding="utf-8",
        )
        data = pd.DataFrame()
        for channel_name, position in CHANNELS_MAP.items():
            data[channel_name] = raw_data.iloc[:, position]

        return data.reset_index(drop=True)


if __name__ == "__main__":
    file = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "OpenBCI-RAW-2020-01-20_19-25-40.txt"
    )
    handler = OpenBci(file)
    data = handler.load_data()
    pass
