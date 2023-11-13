import torch
from typing import List


# Parameters
ANCHOR_SCALES = [512, 2048, 4096, 8192, 12288, 16384] # TODO: Move to model config

# Generate anchors for a 1D signal
def generate_anchors(signal_length: int, anchor_scales: List[int]):
    anchors = []
    for scale in anchor_scales:
        anchor_points = torch.arange(scale / 2, signal_length, scale / 2)
        for point in anchor_points:
            anchors.append((point - scale / 2, point + scale / 2))

    return torch.tensor(anchors)
