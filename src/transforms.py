import torch
import torch.nn.functional as F


def resize(size: int) -> callable:
    def resize_fn(signal: torch.Tensor) -> torch.Tensor:
        return F.interpolate(signal, size=size, mode="linear", align_corners=False)

    return resize_fn
