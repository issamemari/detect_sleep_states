from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from anchors import generate_anchors


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

    def forward(self, x):
        # Forward pass through the network
        x = self.layers(x)

        # Predict class scores and bounding box adjustments
        scores = self.score_pred(x)
        bboxes = self.bbox_pred(x)

        # Reshape the output for easy interpretation
        scores = scores.permute(0, 2, 1).contiguous()
        scores = scores.view(scores.size(0), self.num_anchors, self.num_classes)

        bboxes = bboxes.permute(0, 2, 1).contiguous()
        bboxes = bboxes.view(bboxes.size(0), self.num_anchors, 2)

        return scores, bboxes


class OneDObjectDetectionLoss(nn.Module):
    def calculate_ious(self, anchors: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """
        Calculate the IoU between a ground truth box and a set of anchors
        """
        # Calculate the intersection of the box and anchors
        intersection_start = torch.max(anchors[:, 0], box[0])
        intersection_end = torch.min(anchors[:, 1], box[1])
        intersection = torch.clamp(intersection_end - intersection_start, min=0)

        # Calculate the areas
        anchor_areas = anchors[:, 1] - anchors[:, 0]
        box_area = box[1] - box[0]

        # Calculate the union of the box and anchors
        union = anchor_areas + box_area - intersection

        # Calculate the IoU
        ious = intersection / union

        return ious

    def find_best_anchor(self, anchors: torch.Tensor, box: torch.Tensor) -> int:
        """
        Find the best matching anchor for a given ground truth box
        """
        # Calculate the IoU between the ground truth box and each anchor
        ious = self.calculate_ious(anchors, box)

        # Find the anchor with the highest IoU
        best_anchor = torch.argmax(ious)

        return best_anchor

    def forward(self, scores, bboxes, gt_classes, gt_bboxes, anchors) -> float:
        num_anchors = anchors.size(0)
        num_classes = scores.size(2)

        scores = scores.view(-1, num_anchors, num_classes)

        gt_bboxes = gt_bboxes.view(-1, 2)
        gt_classes = gt_classes.view(-1)

        # get the best anchor for each ground truth box
        best_anchors = []
        for b in range(gt_bboxes.size(0)):
            best_anchors.append(self.find_best_anchor(anchors, gt_bboxes[b]))

        scores_ready = torch.zeros_like(scores)
        scores_ready[range(scores.size(0)), best_anchors, gt_classes] = 1

        bboxes_ready = torch.zeros_like(bboxes)
        bboxes_ready[range(bboxes.size(0)), best_anchors, :] = gt_bboxes

        # for bboxes, keep only best anchors
        bboxes_ready = bboxes_ready[range(bboxes.size(0)), best_anchors, :]
        bboxes = bboxes[range(bboxes.size(0)), best_anchors, :]

        classification_loss = F.binary_cross_entropy_with_logits(
            scores, scores_ready, reduction="sum"
        )

        regression_loss = F.smooth_l1_loss(bboxes, bboxes_ready, reduction="sum")

        return classification_loss + regression_loss


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
