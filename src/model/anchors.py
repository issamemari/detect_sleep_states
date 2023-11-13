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
