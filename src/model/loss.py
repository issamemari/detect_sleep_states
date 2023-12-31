import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import intersection_over_union


class OneDObjectDetectionLoss(nn.Module):
    def find_best_anchor(self, anchors: torch.Tensor, box: torch.Tensor) -> int:
        """
        Find the best matching anchor for a given ground truth box
        """
        # Calculate the IoU between the ground truth box and each anchor
        ious = intersection_over_union(anchors, box)

        # Find the anchor with the highest IoU
        best_anchor = torch.argmax(ious)

        return best_anchor

    def forward(self, scores, bboxes, gt_classes, gt_bboxes, anchors) -> float:
        num_anchors = anchors.size(0)
        num_classes = scores.size(2)

        scores = scores.view(-1, num_anchors, num_classes)

        # get the best anchor for each ground truth box
        best_anchors = []
        for b in range(len(gt_bboxes)):  # batch items
            best_anchors_item = []
            for box in gt_bboxes[b]:
                best_anchors_item.append(self.find_best_anchor(anchors, box))

            best_anchors.append(best_anchors_item)

        scores_ready = torch.zeros_like(scores)
        for i, b in enumerate(best_anchors):
            scores_ready[i, b, gt_classes[i]] = 1

        bboxes_ready = torch.zeros_like(bboxes)
        for i, b in enumerate(best_anchors):
            bboxes_ready[i, b, :] = gt_bboxes[i]

        # keep only the best anchor for each ground truth box
        # set values of bboxes to 0 for anchors that are not the best
        mask = bboxes_ready != 0
        bboxes = bboxes * mask

        classification_loss = F.binary_cross_entropy_with_logits(
            scores, scores_ready, reduction="sum"
        )

        regression_loss = F.smooth_l1_loss(bboxes, bboxes_ready, reduction="sum")

        return regression_loss + classification_loss
