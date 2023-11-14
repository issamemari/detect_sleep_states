import torch

from metrics import intersection_over_union


def non_maximum_suppression(
    scores: torch.Tensor,
    bboxes: torch.Tensor,
    iou_threshold: float,
):
    """
    Perform non-maximum suppression on the predicted bounding boxes

    Args:
        scores: The predicted class scores, shape (batch size, num_anchors, num_classes)
        bboxes: The predicted bounding boxes, shape (batch size, num_anchors, 2)
        iou_threshold: The IoU threshold to use for NMS

    Returns:
        A list of the predicted bounding boxes after NMS
    """

    output = []
    for batch_idx in range(scores.size(0)):
        batch_scores = scores[batch_idx]
        batch_bboxes = bboxes[batch_idx]

        batch_output = []

        while len(batch_bboxes) > 0:
            max_score_idx = torch.argmax(batch_scores.max(dim=1)[0])
            max_score_bbox = batch_bboxes[max_score_idx]

            batch_output.append(max_score_bbox)

            to_remove = []
            for idx, bbox in enumerate(batch_bboxes):
                if idx == max_score_idx:
                    to_remove.append(idx)
                    continue

                iou = intersection_over_union(bbox.unsqueeze(0), max_score_bbox)[0]

                if iou > iou_threshold:
                    to_remove.append(idx)

            to_remove = torch.tensor(to_remove, dtype=torch.int32)

            to_keep = torch.ones(batch_bboxes.shape[0], dtype=torch.bool)
            to_keep[to_remove] = False

            batch_bboxes = batch_bboxes[to_keep]
            batch_scores = batch_scores[to_keep]

        batch_output = torch.vstack(batch_output)
        output.append(batch_output)

    return output
