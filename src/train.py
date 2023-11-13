from dataset import DetectSleepStatesDataset
from model import OneDObjectDetectionCNN, OneDObjectDetectionLoss
from transforms import resize

import torch


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

    loss_function = OneDObjectDetectionLoss()

    signal, gt_bboxes = dataset[0]
    scores, bboxes = model(signal.unsqueeze(0))
    gt_classes = torch.tensor([[1]])


    import code
    code.interact(local=locals())


if __name__ == "__main__":
    main()
