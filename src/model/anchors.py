from typing import List

import torch


# Generate anchors for a 1D signal
def generate_anchors(signal_length: int, anchor_scales: List[int]):
    anchors = []
    for scale in anchor_scales:
        anchor_points = torch.arange(scale / 2, signal_length, scale / 2)
        for point in anchor_points:
            anchors.append((point - scale / 2, point + scale / 2))

    return torch.tensor(anchors) / signal_length


def construct_ground_truth(anchors, ground_truth_boxes, num_classes):
    spatial = 625
    batch_size = 1
    num_anchors = 56

    gt_classes = torch.zeros((batch_size, spatial, num_anchors, num_classes))
    gt_bboxes = torch.zeros((batch_size, spatial, num_anchors, 2))

    for b in range(batch_size):
        for _, box in enumerate(ground_truth_boxes[b]):
            # Find the best matching anchor for each ground truth box
            best_anchor = find_best_anchor(anchors, box)

            # Calculate the offsets for the bounding box relative to the anchor
            gt_bboxes[b, :, best_anchor, :] = encode_offsets(anchors[best_anchor], box)

            # Set the class label for the matched anchor
            gt_classes[b, :, best_anchor, box.class_id] = 1

            # Mark negative and ignore anchors in gt_classes appropriately

    return gt_classes, gt_bboxes
