import torch
import torch.nn.functional as F


def resize(signal: torch.Tensor, *, size: int) -> torch.Tensor:
    return F.interpolate(
        signal.unsqueeze(0).float(),
        size=size,
        mode="linear",
        align_corners=False,
    ).squeeze(0)
