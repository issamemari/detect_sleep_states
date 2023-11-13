from torch.utils.data import Dataset
import pandas as pd


class DetectSleepStatesDataset(Dataset):
    def __init__(self, train_series: str, train_events: str):
        self.train_series = pd.read_csv(train_series)
        self.train_events = pd.read_csv(train_events)

        self.train_series.set_index("series_id", inplace=True)
        self.train_events.set_index("series_id", inplace=True)

    def __len__(self):
        return len(self.train_series)

    def __getitem__(self, idx):
        series_id = self.train_series.index[idx]
        series = self.train_series.loc[series_id]
        events = self.train_events.loc[series_id]
        return series, events


def main():
    dataset = DetectSleepStatesDataset(
        train_series="./data/train_series_preprocessed_sample.csv",
        train_events="./data/train_events_preprocessed_sample.csv"
    )

    import code
    code.interact(local=locals())

    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()