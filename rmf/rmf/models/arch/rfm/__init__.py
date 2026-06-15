"""
Adapted RFM architectures for 2 time inputs.

From: https://github.com/facebookresearch/riemannian-fm.
"""
import torch

from .arch import TMLP
torch.serialization.add_safe_globals([TMLP])