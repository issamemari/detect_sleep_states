import torch


def intersection_over_union(bboxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
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


def mean_average_precision(
    pred_bboxes: torch.Tensor,
    gt_bboxes: torch.Tensor,
) -> float:
    """
    Calculate the mean average precision for a batch of predictions

    Args:
        pred_bboxes: The predicted bounding boxes, shape (num_boxes_pred, 2)
        gt_bboxes: The ground truth bounding boxes, shape (num_boxes_gt, 2)

    Returns:
        Mean average precision
    """

    all_precisions = []
    for batch_idx in range(len(gt_bboxes)):
        batch_pred_bboxes = pred_bboxes[batch_idx]
        batch_gt_bboxes = gt_bboxes[batch_idx]

        # calculate iou between every predicted bbox and every gt bbox
        best_ious = []
        for gt_bbox in batch_gt_bboxes:
            ious = intersection_over_union(batch_pred_bboxes, gt_bbox)

            best_iou_idx = torch.argmax(ious)
            best_iou = ious[best_iou_idx]
            best_ious.append(best_iou)

        precisions = []
        for threshold in torch.arange(0.5, 1, 0.05):
            tp = torch.tensor([best_iou > threshold for best_iou in best_ious]).sum()
            precision = tp / len(batch_pred_bboxes)

            precisions.append(precision)

        all_precisions.append(torch.tensor(precisions).mean().item())

    return torch.tensor(precisions).mean().item()
