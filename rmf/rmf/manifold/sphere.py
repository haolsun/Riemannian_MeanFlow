"""
Adjusted geoopt Sphere, following https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/manifolds/sphere.py.
Meta, all rights reserved, etc.
"""

import geoopt
import torch


class FixedGeooptSphere(geoopt.Sphere):
    def transp(self, x, y, v):
        denom = 1 + self.inner(x, x, y, keepdim=True)
        res = v - self.inner(x, y, v, keepdim=True) / denom * (x + y)
        cond = denom.gt(1e-3)
        return torch.where(cond, res, -v)
