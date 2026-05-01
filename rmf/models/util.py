"""
Utility functions.
"""

from torch import Tensor


def match_dims(tensor: Tensor, size: tuple[int, ...]) -> Tensor:
    """Match the dimensions of a tensor to a given size."""
    while tensor.dim() < len(size):
        tensor = tensor.unsqueeze(-1)
    return tensor
