import torch
import torch.nn as nn
import torch.nn.functional as F


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
