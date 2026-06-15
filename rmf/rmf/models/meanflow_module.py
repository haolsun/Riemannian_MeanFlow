"""
Generalised Mean Flow module.
"""

import torch
from torch import Tensor

from gfm.models import match_dims
from gfm.models import RiemannianGenerativeModule


class MeanFlowModule(RiemannianGenerativeModule):
    """
    Generalised Mean Flow module.

    :param prop_equal_times: Proportion of times that should be equal (`r = t`).
    :param time_distribution: Distribution to sample times from. One of "uniform" or "lognormal".
    """

    def __init__(
        self,
        prop_equal_times: float = 0.0,
        time_distribution: str = "uniform",
        *args,
        **kwargs,
    ):
        """
        Args:
            prop_equal_times (float): Proportion of times that should be equal (`r = t`).
        """
        # assert len(in_shape) >= 2, f"Input shape must have at least 2 dimensions `(n_products, manifold_dims...)`, got shape: {in_shape}"
        super().__init__(*args, **kwargs)
        if time_distribution == "uniform":
            self.time_distribution = torch.distributions.Uniform(
                0.0, 1.0, validate_args=False
            )
        else:
            self.time_distribution = torch.distributions.LogNormal(
                -0.4, 1.0, validate_args=False
            )

    @property
    def is_sampling_time_zero_to_one(self) -> bool:
        return False

    @torch.no_grad()
    def sample_r_t(self, n: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Sample random times `r` and `t` for the mean flow.
        `r` is sampled uniformly in [0, 1] and `t` is sampled uniformly in [0, 1].
        """
        if self.hparams.time_distribution == "uniform":
            t = torch.rand(n, device=device)
            r = torch.rand(n, device=device) * t
            return r, t
        t = self.time_distribution.sample((n,)).to(device)
        r = self.time_distribution.sample((n,)).to(device)
        msk = r > t
        tmp = r[msk]
        r[msk] = t[msk]
        t[msk] = tmp
        return r.sigmoid(), t.sigmoid()

    # come from alpha-flow
    @torch.no_grad()
    def alpha_sample(self):
        k_s = 0
        k_e = self.epoch
        gamma = 25
        eta = 0.005
        k = self.current_epoch + 1
        scale = 1/(k_e - k_s)
        mid = - (k_s + k_e) / 2 / (k_e - k_s)
        alpha = 1 - (1 / (1 + torch.exp(torch.tensor((scale * k + mid) * gamma))))
        alpha = 1 if alpha > 1 - eta else (0 if alpha < eta else alpha)
        return alpha

    def get_loss(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(self.hparams.prop_equal_times * x.shape[0])
            eps = self.manifold.prior(shape=x.shape, device=x.device)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

            xt, v = torch.func.jvp(
                geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
            )

        u, dudt = torch.func.jvp(
            self.forward,
            (xt, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )
        with torch.no_grad():
            # both v and dudt are tangent vectors already
            u_tgt = v - (t - r) * dudt.detach()

        diff = u - u_tgt
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        return self.manifold.inner(xt, diff, diff).mean()

    def get_loss_imf(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(0.5 * x.shape[0])
            eps = self.manifold.prior(shape=x.shape, device=x.device)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

        xt, v = torch.func.jvp(
            geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
        )
        v_tt = self.forward(xt, t, t)
        u, dudt = torch.func.jvp(
            self.forward,
            (xt, r, t),
            (v_tt, torch.zeros_like(r), torch.ones_like(t)),
        )
        V = u + (t - r) * dudt.detach()

        diff = V - v
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        return self.manifold.inner(xt, diff, diff).mean()


    def get_loss_alpha_p(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(self.hparams.prop_equal_times * x.shape[0])
            # n_equal = int(0 * x.shape[0])
            eps = self.manifold.prior(shape=x.shape, device=x.device)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

            xt, v = torch.func.jvp(
                geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
            )

        u, dudt = torch.func.jvp(
            self.forward,
            (xt, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )
        with torch.no_grad():
            # both v and dudt are tangent vectors already
            u_tgt = v - (t - r) * dudt.detach()
            alpha = self.alpha_sample()

        diff = u - v
        diff_dudt = (t - r) * dudt.detach()
        xt_1 = xt
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        l_1 = self.manifold.inner(xt, diff, diff).mean()
        if diff_dudt.isnan().any():
            nans = diff_dudt.isnan().any(dim=(-1, -2))
            xt_1 = xt_1[~nans]
            diff_dudt = diff_dudt[~nans]
            u = u[~nans]
        l_2 = (1 - alpha) * self.manifold.inner(xt_1, u, 2 * diff_dudt).mean()
        return l_1 + l_2

    def get_loss_alpha_flow(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(self.hparams.prop_equal_times * x.shape[0])
            # n_equal = int(0 * x.shape[0])
            eps = self.manifold.prior(shape=x.shape, device=x.device)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

            xt, v = torch.func.jvp(
                geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
            )

        with torch.no_grad():
            # alpha flow edit
            alpha = self.alpha_sample()
            if alpha == 0:
                u, dudt = torch.func.jvp(
                    self.forward,
                    (xt, r, t),
                    (v, torch.zeros_like(r), torch.ones_like(t)),
                )
                u_tgt = v - (t - r) * dudt
            else:
                s = r * alpha + (1 - alpha) * t
                xs, v_s = torch.func.jvp(
                    geodesic, primals=(s,), tangents=(torch.ones_like(s),), has_aux=False
                )
                u_tgt = alpha * v_s + (1 - alpha) * self.forward(xs, s, t)
        diff = self.forward(xt, r, t) - u_tgt.detach()
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        return self.manifold.inner(xt, diff, diff).mean()

    def get_loss_pc_grad(self, x: Tensor):
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(self.hparams.prop_equal_times * x.shape[0])
            eps = self.manifold.prior(shape=x.shape, device=x.device)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

            xt, v = torch.func.jvp(
                geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
            )
        u, dudt = torch.func.jvp(
            self.forward,
            (xt, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )
        diff = u - v
        diff_dudt = (t - r) * dudt.detach()
        xt_1 = xt
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        l_1 = self.manifold.inner(xt, diff, diff).mean()
        if diff_dudt.isnan().any():
            nans = diff_dudt.isnan().any(dim=(-1, -2))
            xt_1 = xt_1[~nans]
            diff_dudt = diff_dudt[~nans]
            u = u[~nans]
        l_2 = self.manifold.inner(xt_1, u, 2 * diff_dudt).mean()
        return l_1, l_2

    @torch.inference_mode()
    def sample_batch(self, steps: int, sz: int) -> Tensor:
        z = self.manifold.prior(
            shape=(sz,) + tuple(self.hparams.in_shape), device=self.device
        )
        ts = torch.linspace(0, 1, steps + 1, device=self.device).flip(0)

        for r, t in zip(ts[1:], ts[:-1]):
            t = match_dims(t.expand(sz), z.shape)
            r = match_dims(r.expand(sz), z.shape)
            z = self.manifold.exp(z, -(t - r) * self.forward(z, r, t))
            z = self.manifold.project(z)

        return z

    def name(self) -> str:
        return "mf"

    def forward(self, x: Tensor, s: Tensor, t: Tensor) -> Tensor:
        return self.net(x, s, t)


class AminogfmModule(MeanFlowModule):
    @property
    def val_nll(self):
        return True


class ImageMeanFlowModule(MeanFlowModule):
    @property
    def val_images(self):
        return True
