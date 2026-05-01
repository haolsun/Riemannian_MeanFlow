"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
import torch.nn as nn

import gfm.models.arch.rfm.diffeq_layers as diffeq_layers
from gfm.models.arch.rfm.actfn import Sine, Softplus


ACTFNS = {
    "swish": diffeq_layers.TimeDependentSwish,
    "sine": Sine,
    "srelu": Softplus,
}


class TMLP(nn.Module):
    """A time-dependent MLP with optional Fourier features."""

    def __init__(
        self, d_in, d_out=None, d_model=256, num_layers=6, actfn="swish", fourier=None
    ):
        super().__init__()
        assert num_layers > 1, "No weak linear nets here"
        d_out = d_in if d_out is None else d_out
        actfn = ACTFNS[actfn]
        if fourier:
            layers = [
                diffeq_layers.diffeq_wrapper(
                    PositionalEncoding(n_fourier_features=fourier)
                ),
                diffeq_layers.ConcatLinear_v2(d_in * fourier * 2, d_model),
            ]
        else:
            layers = [diffeq_layers.ConcatLinear_v2(d_in, d_model)]

        for _ in range(num_layers - 2):
            layers.append(actfn(d_model))
            layers.append(diffeq_layers.ConcatLinear_v2(d_model, d_model))
        layers.append(actfn(d_model))
        layers.append(diffeq_layers.ConcatLinear_v2(d_model, d_out))
        self.net = diffeq_layers.SequentialDiffEq(*layers)

    def forward(self, x, r, t):
        return self.net(r, t, x)


class PositionalEncoding(nn.Module):
    """Assumes input is in [0, 2pi]."""

    def __init__(self, n_fourier_features):
        super().__init__()
        self.n_fourier_features = n_fourier_features

    def forward(self, x):
        feature_vector = [
            torch.sin((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        feature_vector += [
            torch.cos((i + 1) * x) for i in range(self.n_fourier_features)
        ]
        return torch.cat(feature_vector, dim=-1)


class Unbatch(nn.Module):
    def __init__(self, vecfield):
        super().__init__()
        self.vecfield = vecfield

    def forward(self, t, x):
        has_batch = x.ndim > 1
        if not has_batch:
            x = x.reshape(1, -1)
            t = t.reshape(-1)
        v = self.vecfield(t, x)
        if not has_batch:
            v = v[0]
        return v
