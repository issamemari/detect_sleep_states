from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchors import generate_anchors
from .loss import OneDObjectDetectionLoss


class OneDObjectDetectionCNN(nn.Module):
    def __init__(
        self,
        *,
        signal_length: int,
        input_channels: int,
        num_classes: int,
        anchor_scales: List[int],
    ):
        super(OneDObjectDetectionCNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=33, padding=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=33, padding=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=33, padding=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=33, padding=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )

        # Generate anchors
        self.anchors = generate_anchors(signal_length, anchor_scales)
        self.num_anchors = len(self.anchors)

        # Prediction layers
        self.score_pred = nn.Conv1d(64, self.num_anchors * num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv1d(
            64, self.num_anchors * 2, kernel_size=1
        )  # Each anchor has 2 coordinates (start, end)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the network
        x = self.layers(x)

        # Predict class scores
        scores = self.score_pred(x)

        # Predict bounding box adjustments
        bboxes = self.bbox_pred(x)
        bboxes = self.sigmoid(bboxes)

        # Reshape the output for easy interpretation
        scores = scores.permute(0, 2, 1).contiguous()
        scores = scores.view(scores.size(0), self.num_anchors, self.num_classes)

        bboxes = bboxes.permute(0, 2, 1).contiguous()
        bboxes = bboxes.view(bboxes.size(0), self.num_anchors, 2)

        return scores, bboxes


def main():
    # Example usage
    signal_length = 10000  # Example signal length
    model = OneDObjectDetectionCNN(
        signal_length=signal_length,
        input_channels=2,
        num_classes=2,
        anchor_scales=[512, 2048, 4096, 8192, 12288, 16384],
    )

    # Example input (batch size of 1, 2 channel, signal length of signal_length)
    input_signal = torch.randn(4, 2, signal_length)

    # Get model predictions
    scores, bboxes = model(input_signal)

    # Example ground truth
    gt_classes = torch.tensor([[1], [1], [1], [1]])
    gt_bboxes = (
        torch.tensor([[[0, 120]], [[5, 1024]], [[3060, 4000]], [[5, 1024]]]).float()
        / signal_length
    )

    # Compute loss
    loss_fn = OneDObjectDetectionLoss()

    loss = loss_fn(scores, bboxes, gt_classes, gt_bboxes, model.anchors)

    print(loss)


if __name__ == "__main__":
    main()
