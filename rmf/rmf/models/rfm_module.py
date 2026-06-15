"""
Riemannian flow matching module.
"""

from torch import Tensor
import torch
from gfm.models import RiemannianGenerativeModule, match_dims


class RFMModule(RiemannianGenerativeModule):
    """
    Simple re-implementation of RFM (https://github.com/facebookresearch/riemannian-fm).
    """

    def get_loss(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            x0 = self.manifold.prior(x.shape, device=x.device)
            t = torch.rand(x.shape[0], device=x.device)
            # t = match_dims(t, x0.shape)
            xt, vt = self.manifold.geodesic_with_tangent(x0, x, t)
            # assert self.manifold.all_belong(xt)
            # assert self.manifold.all_belong_tangent(xt, vt)
        diff = vt - self.net(xt, t, t)
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        return self.manifold.inner(xt, diff, diff).mean()

    @property
    def val_nll(self) -> bool:
        return True

    def get_loss_alpha_flow(self,x: Tensor | list[Tensor]):
        print("flow map no this loss")
        return 0

    def get_loss_alpha_p(self,x: Tensor | list[Tensor]):
        print("flow map no this loss")
        return 0

    def get_loss_imf(self,x: Tensor | list[Tensor]):
        print("flow map no this loss")
        return 0

    def get_loss_pc_grad(self,x: Tensor | list[Tensor]):
        print("flow map no this loss")
        return 0


    @torch.inference_mode()
    def sample_batch(self, steps: int, sz: int) -> Tensor:
        # Euler sampling
        ts = torch.linspace(0, 1, steps + 1, device=self.device)
        x = self.manifold.prior(
            shape=(sz,) + tuple(self.hparams.in_shape), device=self.device
        )
        for s, t in zip(ts[:-1], ts[1:]):
            x = self.manifold.exp(
                x,
                (t - s) * self.net(x, t, t),
            )
        return x

    def name(self) -> str:
        return "rfm"
