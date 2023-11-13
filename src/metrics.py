import torch


def iou(bboxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
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

    # Sort the predictions by confidence
    pred_bboxes = pred_bboxes[torch.argsort(pred_bboxes[:, 0], descending=True)]

    # Calculate the IoU between each prediction and the ground truth boxes
    ious = iou(pred_bboxes[:, 1:], gt_bboxes)

    # Calculate the precision and recall for each prediction
    num_predictions = pred_bboxes.size(0)
    precisions = torch.zeros(num_predictions)
    recalls = torch.zeros(num_predictions)
    for idx, iou in enumerate(ious):
        # Calculate the number of true positives
        true_positives = torch.sum(iou > 0.5)

        # Calculate the precision and recall
        precision = true_positives / (idx + 1)
        recall = true_positives / gt_bboxes.size(0)

        precisions[idx] = precision
        recalls[idx] = recall

    # Calculate the average precision
    average_precision = torch.sum(precisions * recalls) / torch.sum(recalls)

    return average_precision
