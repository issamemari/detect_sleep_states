import pandas as pd
import torch
from torch.utils.data import Dataset


class DetectSleepStatesDataset(Dataset):
    def __init__(self, series: str, events: str, transform=None):
        self.series = pd.read_parquet(series)
        self.events = pd.read_csv(events)

        self.series.set_index("series_id", inplace=True)
        self.events.set_index("series_id", inplace=True)

        self.series_ids = list(set(self.series.index))

        self.transform = transform

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        series_id = self.series_ids[idx]
        series = self.series.loc[series_id]
        events = self.events.loc[series_id]

        signal = torch.tensor(series[["anglez", "enmo"]].values.T)
        signal_len = signal.shape[1]

        bboxes = []
        # loop over events in pairs
        for i in range(0, len(events), 2):
            start = events.iloc[i]["step"] / signal_len
            end = events.iloc[i + 1]["step"] / signal_len

            bboxes.append([start, end])

        bboxes = torch.tensor(bboxes).float()

        if self.transform:
            signal = self.transform(signal)

        return signal, bboxes


def main():
    dataset = DetectSleepStatesDataset(
        series="./data/train_series.parquet",
        events="./data/train_events.csv",
    )

    import code

    code.interact(local=locals())

    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
