import torch
from tqdm import tqdm

from dataset import DetectSleepStatesDataset
from model import OneDObjectDetectionCNN, OneDObjectDetectionLoss
from transforms import resize


def collate_fn(batch):
    signal, gt_bboxes = zip(*batch)

    signal = torch.stack(signal)

    return signal, gt_bboxes


def create_data_loaders(dataset, batch_size: int):
    split_point = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [split_point, len(dataset) - split_point]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


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

    train_loader, val_loader = create_data_loaders(dataset, batch_size=4)

    loss_function = OneDObjectDetectionLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    batch_size = 4

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(10):
        tqdm_iterator = tqdm(train_loader, total=len(train_loader))
        tqdm_iterator.set_description(f"Epoch {epoch}: ")

        for batch in tqdm_iterator:
            signal, gt_bboxes = batch

            scores, bboxes = model(signal)

            gt_classes = torch.tensor([[1]] * batch_size)

            loss = loss_function(scores, bboxes, gt_classes, gt_bboxes, model.anchors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss)


if __name__ == "__main__":
    main()
