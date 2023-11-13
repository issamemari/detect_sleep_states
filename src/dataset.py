import pandas as pd
import torch
from torch.utils.data import Dataset


class DetectSleepStatesDataset(Dataset):
    def __init__(self, train_series: str, train_events: str, transform=None):
        self.train_series = pd.read_csv(train_series)
        self.train_events = pd.read_csv(train_events)

        self.train_series.set_index("series_id", inplace=True)
        self.train_events.set_index("series_id", inplace=True)

        self.series_ids = list(set(self.train_series.index))

        self.transform = transform

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        series_id = self.series_ids[idx]
        series = self.train_series.loc[series_id]
        events = self.train_events.loc[series_id]

        signal = torch.tensor(series[["anglez", "enmo"]].values.T)
        signal_len = signal.shape[1]

        bboxes = []
        # loop over events in pairs
        for i in range(0, len(events), 2):
            start = events.iloc[i]["step"] / signal_len
            end = events.iloc[i + 1]["step"] / signal_len

            bboxes.append([start, end])

        bboxes = torch.tensor(bboxes)

        if self.transform:
            signal = self.transform(signal)

        return signal, bboxes


def main():
    dataset = DetectSleepStatesDataset(
        train_series="./data/train_series_preprocessed_sample.csv",
        train_events="./data/train_events_preprocessed_sample.csv",
    )

    import code

    code.interact(local=locals())

    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
