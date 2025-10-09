from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyedflib


class EDF:
    def __init__(self, source_file: Path):
        self.source_file = Path(source_file)

    def set_meta(self):
        meta: Dict[str, object] = {}
        with pyedflib.EdfReader(str(self.source_file)) as f:
            n = f.signals_in_file
            meta["sampling-rate"] = [f.getSampleFrequency(i) for i in range(n)]
            meta["header"] = f.getHeader()  # file-level header dict
            meta["annotations"] = f.readAnnotations()  # list of annotations
            meta["equitment"] = f.getEquipment()
            meta["patient-code"] = f.getPatientCode()
            meta["patient-name"] = f.getPatientName()
            meta["units"] = [f.getPhysicalDimension(i) for i in range(n)]
        return meta

    def set_channels(self) -> list:
        with pyedflib.EdfReader(str(self.source_file)) as f:
            ch_names = f.getSignalLabels()  # list of channel labels
        return ch_names

    def set_sampling_rate(self) -> pd.DataFrame:
        with pyedflib.EdfReader(str(self.source_file)) as f:
            ch_names = f.getSignalLabels()  # list of channel labels
            rates = {
                ch: float(f.getSampleFrequency(i)) for i, ch in enumerate(ch_names)
            }
        return pd.DataFrame([rates])

    def load_data(self) -> pd.DataFrame:
        data = pd.DataFrame()
        with pyedflib.EdfReader(str(self.source_file)) as f:
            n = f.signals_in_file
            ch_names = f.getSignalLabels()  # list of channel labels
            sigbufs = np.zeros((n, f.getNSamples()[0]))
            for i in range(n):
                temp = f.readSignal(i)
                sigbufs[i, : temp.size] = temp

            for i, ch in enumerate(ch_names):
                data[ch] = sigbufs[i, :]

        return data.reset_index(drop=True)


if __name__ == "__main__":
    file = Path(__file__).parent.parent.parent.parent / "data" / "eeg_data.edf"
    handler = EDF(file)
    data = handler.load_data()
    pass
