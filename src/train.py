from dataset import DetectSleepStatesDataset
from model import OneDObjectDetectionCNN
from transforms import resize


def main():
    signal_length = 40000
    model = OneDObjectDetectionCNN(
        signal_length=signal_length, num_classes=1, num_anchors=10
    )

    dataset = DetectSleepStatesDataset(
        train_series="./data/train_series_preprocessed_sample.csv",
        train_events="./data/train_events_preprocessed_sample.csv",
        transform=resize(signal_length),
    )


if __name__ == "__main__":
    main()
