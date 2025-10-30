from pathlib import Path
from typing import Dict

import chardet
import pandas as pd
from pandas._config import detect_console_encoding

CHANNELS_MAP = [
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
    "OPTO",
]


class Neutronic:
    def __init__(self, source_file: Path, data_col_index=2):
        self.source_file = Path(source_file)
        if self.source_file.suffix != ".txt":
            raise ValueError("Neutronic file must be a .txt file")
        self.file_encoding, _ = self._detect_encoding_chardet()
        self.data_col_index = data_col_index
        self.channel_map = self._set_channel_map()

    def _detect_encoding_chardet(self, sample_size=1_000_000):
        with open(self.source_file, "rb") as f:
            raw = f.read(sample_size)
        guess = chardet.detect(raw)
        return guess.get("encoding"), guess.get("confidence", 0.0)

    def _set_channel_map(self):
        return {ch: self.data_col_index + i for i, ch in enumerate(CHANNELS_MAP)}

    def set_meta(self):
        meta: Dict[str, object] = {}
        meta["sampling_rate"] = 65
        return meta

    def set_channels(self):
        return list(self.channel_map.keys())

    def set_sampling_rate(self) -> pd.DataFrame:
        rates = {ch: float(65) for _, ch in enumerate(self.channel_map.keys())}
        return pd.DataFrame([rates])

    def load_data(self) -> pd.DataFrame:
        raw_data = pd.read_csv(
            self.source_file,
            sep=r"\s+",
            skiprows=2,
            encoding=self.file_encoding,
            names=[f"col{i}" for i in range(50)],
            header=None,
        )
        data = pd.DataFrame()
        for channel_name, position in self.channel_map.items():
            data[channel_name] = raw_data.iloc[:, position]

        return data.reset_index(drop=True)


if __name__ == "__main__":
    # file = Path(__file__).parent.parent.parent.parent / "data" / "neutronic_data.txt"
    file = Path(__file__).parent.parent.parent.parent / "data" / "ID_18_EEG_02_new.txt"
    handler = Neutronic(file)
    data = handler.load_data()
    pass
