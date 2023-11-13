import torch


def calculate_ious(bboxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """
    Calculate the IoUs between a bounding box and a batch of bounding boxes.

    Args:
        bboxes: The anchors, shape (num_anchors, 2)
        box: The ground truth box, shape (2,)

    Returns:
        The IoU between the box and each bounding box, shape (num_anchors,)
    """
    # Calculate the intersection of the box and anchors
    intersection_start = torch.max(bboxes[:, 0], box[0])
    intersection_end = torch.min(bboxes[:, 1], box[1])
    intersection = torch.clamp(intersection_end - intersection_start, min=0)

    # Calculate the areas
    anchor_areas = bboxes[:, 1] - bboxes[:, 0]
    box_area = box[1] - box[0]

    # Calculate the union of the box and anchors
    union = anchor_areas + box_area - intersection

    # Calculate the IoU
    ious = intersection / union

    return ious
