import torch
import torch.nn.functional as F


def resize(size: int) -> callable:
    def resize_fn(signal: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            signal.unsqueeze(0).float(),
            size=size,
            mode="linear",
            align_corners=False,
         ).squeeze(0)

    return resize_fn
