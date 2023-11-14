import functools
import logging
import os

import hydra
import torch
from tqdm import tqdm

import wandb
from dataset import DetectSleepStatesDataset
from metrics import mean_average_precision
from model import (
    OneDObjectDetectionCNN,
    OneDObjectDetectionLoss,
    non_maximum_suppression,
)
from transforms import resize

logger = logging.getLogger(__name__)


def collate_fn(batch):
    signal, gt_bboxes = zip(*batch)

    signal = torch.stack(signal)

    return signal, gt_bboxes


def create_data_loaders(dataset, *, batch_size: int, num_workers: int):
    split_point = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [split_point, len(dataset) - split_point]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.2")
def main(config):
    wandb.init(
        project="detect_sleep_states",
        config=dict(config),
        entity="issa-memari",
    )
    model = OneDObjectDetectionCNN(**config.model)

    dataset = DetectSleepStatesDataset(
        **config.dataset,
        transform=functools.partial(resize, size=config.model.signal_length),
    )

    train_loader, val_loader = create_data_loaders(dataset, **config.data_loader)

    loss_function = OneDObjectDetectionLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    for epoch in range(config.training.epochs):
        print("Epoch", epoch)

        tqdm_iterator = tqdm(train_loader, total=len(train_loader))

        model.train()
        mean_training_loss = 0
        for step, batch in enumerate(tqdm_iterator):
            signal, gt_bboxes = batch

            gt_classes = torch.tensor([[1]] * config.data_loader.batch_size)

            scores, bboxes = model(signal)

            loss = loss_function(scores, bboxes, gt_classes, gt_bboxes, model.anchors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_training_loss = mean_training_loss + (
                loss.item() - mean_training_loss
            ) / (step + 1)
            tqdm_iterator.set_description(f"loss {mean_training_loss:.4f}")

        # Validation
        tqdm_iterator = tqdm(val_loader, total=len(val_loader))

        model.eval()
        mean_val_loss = 0
        map = 0
        for step, batch in enumerate(tqdm_iterator):
            signal, gt_bboxes = batch

            gt_classes = torch.tensor([[1]] * config.data_loader.batch_size)

            with torch.no_grad():
                scores, bboxes = model(signal)

                loss = loss_function(
                    scores, bboxes, gt_classes, gt_bboxes, model.anchors
                )

            bboxes_filtered = non_maximum_suppression(
                scores, bboxes, iou_threshold=config.training.iou_threshold
            )

            batch_map = mean_average_precision(bboxes_filtered, gt_bboxes)

            map = map + (batch_map - map) / (step + 1)

            mean_val_loss = mean_val_loss + (loss.item() - mean_val_loss) / (step + 1)

            tqdm_iterator.set_description(
                f"val_loss {mean_val_loss:.4f}, map {map:.4f}"
            )

        # save model to output dir
        output_dir = config.training.output_dir

        os.makedirs(output_dir, exist_ok=True)

        torch.save(model.state_dict(), f"{output_dir}/model_{epoch}.pth")


if __name__ == "__main__":
    main()
