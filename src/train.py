from dataset import DetectSleepStatesDataset
from model import OneDObjectDetectionCNN
from transforms import resize


def main():
    signal_length = 40000
    model = OneDObjectDetectionCNN(
        signal_length=signal_length,
        input_channels=2,
        num_classes=2,
        anchor_scales=[512, 2048, 4096, 8192, 12288, 16384],
    )

    dataset = DetectSleepStatesDataset(
        train_series="./data/train_series_preprocessed_sample.csv",
        train_events="./data/train_events_preprocessed_sample.csv",
        transform=resize(signal_length),
    )

    import code
    code.interact(local=locals())


if __name__ == "__main__":
    main()
