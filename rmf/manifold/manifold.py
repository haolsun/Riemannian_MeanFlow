"""
Definitions of manifolds.

Substantial parts of code were taken from FoldFlow (1 & 2) for SO(3) and SE(3).
"""

import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from abc import ABC, abstractmethod
from enum import Enum
import math

import numpy as np

import torch
from torch.distributions import Dirichlet
from torch import Tensor
from torch.nn import functional as F

from geoopt.manifolds import (
    PoincareBall as GeooptPoincareBall,
)
from geoopt.manifolds.stereographic import math as stereo_math

import geomstats
from geomstats import backend
backend.set_default_dtype("float32")
from einops import rearrange


from gfm.manifold import FixedGeooptSphere


_eps = 1e-8


def group_rot_trans(rot: Tensor, trans: Tensor) -> Tensor:
    """Convert rotation matrices and translations to grouped SE(3) representation.

    Args:
        rot: Rotation matrices of shape (..., 3, 3).
        trans: Translation vectors of shape (..., 3).

    Returns:
        Grouped SE(3) representation of shape (..., 4, 4).
    """
    # Create an empty tensor for the grouped representation
    grouped = torch.eye(4, device=rot.device, dtype=rot.dtype).expand(*rot.shape[:-2], 4, 4).clone()
    grouped[..., :3, :3] = rot
    grouped[..., :3, 3] = trans
    return grouped


def _time_ndims(t: Tensor, y: Tensor) -> Tensor:
    return t.view((t.shape[0],) + (1,) * (len(y.shape) - 1))


DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def so3_rotation_angle(
    R: torch.Tensor,
) -> torch.Tensor:
    assert R.shape[-1] == 3 and R.shape[-2] == 3
    # rot_trace ... trace of rotation matrix
    rot_trace = torch.diagonal(R, dim1=-1, dim2=-2).sum(dim=-1)
    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5
    return acos_linear_extrapolation(phi_cos, (-0.999, 0.999))


def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12)



def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    return _so3_exp_map(log_rot, eps=eps)[0]


def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


class ManifoldType(Enum):
    """
    Defines a few essential functions for manifolds.
    """

    NONE = 1
    SIMPLEX = 2
    SPHERE = 3
    FLAT_TORUS = 4
    SO3 = 5
    SE3 = 6
    FLAT_SIMPLEX = 7
    POINCARE_BALL = 8


class Manifold(ABC):
    """Defines the manifold abstraction."""

    @abstractmethod
    def get_type(self) -> ManifoldType:
        """
        Returns the type of the manifold.
        """

    @abstractmethod
    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        """
        Defines the exponential map at `p` in the direction `v`.
        """

    @abstractmethod
    def log(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Defines the logarithmic map from `p` to `q`.
        """

    @abstractmethod
    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        """
        Returns the geodesic distance of points `p` and `q` on the manifold.
        """

    @abstractmethod
    def all_belong(self, p: Tensor) -> bool:
        """
        Returns `true` iff every point in `p` (batched) is in the manifold `self`.
        """

    @abstractmethod
    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        """
        Returns `true` iff every point in `p` (batched) and tangent vector `v`
        belong to the tangent space of the manifold at `p`.
        """

    @abstractmethod
    def project(self, x: Tensor) -> Tensor:
        """
        Projects the points `x` to the manifold.
        """

    @abstractmethod
    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """
        Projects the tangent vector `v` at point `x` to the tangent space of the manifold.
        """

    def geodesic_interpolant(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Returns the geodesic interpolant at time `t`, i.e.,
        `exp_{x_0}(t log_{x_0}(x_1))`.
        """
        # assert self.all_belong(x0)
        # assert self.all_belong(x1)
        lg = self.log(x0, x1)
        while len(t.shape) < len(lg.shape):
            t = t[..., None]
        return self.project(self.exp(x0, t * lg))

    def geodesic_with_tangent(
        self, x0: Tensor, x1: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Returns `xt` with its time derivative.
        """
        return torch.func.jvp(
            lambda _t: self.geodesic_interpolant(x0, x1, _t),
            primals=(t,),
            tangents=(torch.ones_like(t),),
        )

    @abstractmethod
    def prior(
        self, shape: tuple[int, ...], device: str | torch.device = "cpu"
    ) -> Tensor:
        """
        Returns samples from the default prior on the given manifold.
        """

    @abstractmethod
    def logp0(self, x: Tensor) -> Tensor:
        """
        Returns the log-probability of the point `x` under the default prior on the manifold.
        """

    @abstractmethod
    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """
        Returns the inner product a point `p` between `u` and `v`.
        """

    @abstractmethod
    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        Returns the parallely transported `v` from `p` to `q`.
        """


class EuclideanSpace(Manifold):
    """
    Euclidean space.
    """

    def get_type(self) -> ManifoldType:
        return ManifoldType.NONE

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        return p + v

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        return q - p

    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        return torch.norm(p - q, dim=-1)

    def all_belong(self, x: Tensor) -> bool:
        return True

    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        return True

    def project_tangent(self, x, v):
        return v

    def project(self, x: Tensor) -> Tensor:
        return x

    def prior(
        self, shape: tuple[int, ...], device: str | torch.device = "cpu"
    ) -> Tensor:
        return torch.randn(shape, device=device)

    def logp0(self, x: Tensor) -> Tensor:
        raise NotImplementedError("logp0 not implemented for R^n")

    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return torch.sum(u * v, dim=-1)

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        return v


class FRSimplex(Manifold):
    """
    A manifold for the Fisher-Rao-equipped simplex.
    """

    def get_type(self) -> ManifoldType:
        return ManifoldType.SIMPLEX

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        s = p.sqrt()
        xs = v / s.clamp(min=_eps) / 2.0
        theta = torch.norm(xs, dim=-1, keepdim=True)
        return (s * torch.cos(theta) + xs * torch.sinc(theta / torch.pi)) ** 2

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        z = torch.sqrt(p * q)
        s = z.sum(-1, keepdim=True)
        dist = 2.0 * torch.acos(s.clamp(0, 1 - _eps))
        u = dist / torch.sqrt((1 - s**2).clamp(min=_eps)) * (z - s * p)
        return torch.where(dist > _eps, u, q - p)

    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        return 2.0 * torch.acos((p * q).sqrt().sum(dim=-1).clamp(0.0, 1.0))

    def all_belong(self, x: Tensor) -> bool:
        return bool((x >= 0).all().item()) and torch.allclose(
            x.sum(dim=-1), torch.tensor(1.0)
        )

    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        return torch.allclose(v.sum(dim=-1), torch.tensor(0.0, device=v.device))

    def project(self, x: Tensor) -> Tensor:
        return x / (x.sum(dim=-1, keepdim=True) + _eps)

    def project_tangent(self, x, v):
        return v - v.mean(dim=-1, keepdim=True)

    def prior(
        self, shape: tuple[int, ...], device: str | torch.device = "cpu"
    ) -> Tensor:
        return Dirichlet(torch.ones(shape, device=device), validate_args=False).sample()

    def logp0(self, x: Tensor) -> Tensor:
        raise NotImplementedError("logp0 not implemented for FRSimplex")

    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return ((u * v) / p.clamp(min=1e-8)).sum(dim=-1)

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        """
        See `Manifold.parallel_transport`. Based on the parallel transport of
        `NSphere`.
        """
        sphere = Sphere()
        q_s = q.sqrt()
        y_s = sphere.parallel_transport(
            p.sqrt(),
            q_s,
            v / p.sqrt(),
        )
        return y_s * q_s


class FlatSimplex(EuclideanSpace):
    """
    Simplex endowed with simple "flat" Euclidean metric.
    """

    def get_type(self) -> ManifoldType:
        return ManifoldType.FLAT_SIMPLEX

    def prior(self, *args, **kwargs) -> Tensor:
        return FRSimplex().prior(*args, **kwargs)

    def project_tangent(self, x, v):
        return FRSimplex().project_tangent(x, v)

    def project(self, x: Tensor) -> Tensor:
        return FRSimplex().project(x)

    def logp0(self, x: Tensor) -> Tensor:
        return FRSimplex().logp0(x)


class FlatTorus(Manifold):
    """
    A flat torus manifold.
    """

    def get_type(self) -> ManifoldType:
        return ManifoldType.FLAT_TORUS

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        return (p + v) % (2.0 * torch.pi)

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        return torch.atan2(
            torch.sin(q - p),
            torch.cos(q - p),
        )

    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        return torch.min(torch.abs(q - p), 2.0 * torch.pi - torch.abs(q - p))

    def all_belong(self, x: Tensor) -> bool:
        return bool((x >= 0).all().item()) and bool((x < 2.0 * torch.pi).all().item())

    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        return True

    def project(self, x: Tensor) -> Tensor:
        return x % (2.0 * torch.pi)

    def project_tangent(self, x, v):
        return v

    def prior(
        self, shape: tuple[int, ...], device: str | torch.device = "cpu"
    ) -> Tensor:
        return torch.rand(shape, device=device) * 2.0 * torch.pi

    def logp0(self, x: Tensor) -> Tensor:
        dim = x.shape[-1]
        return torch.full_like(x[..., 0], -dim * math.log(2 * math.pi))

    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return (u * v).sum(dim=-1)

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        return v


class Sphere(Manifold):
    """
    A sphere manifold.
    """

    def __init__(self):
        super().__init__()
        self.sphere = FixedGeooptSphere()

    def get_type(self) -> ManifoldType:
        return ManifoldType.SPHERE

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        return self.sphere.expmap(p, v)

    def log(self, p: Tensor, q: Tensor, eps=1e-8) -> Tensor:
        return self.sphere.logmap(p, q)

    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        return self.sphere.dist(p, q)

    def all_belong(self, x: Tensor) -> bool:
        return x.square().sum(dim=-1).allclose(torch.tensor(1.0))

    def project(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1)

    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        return self.sphere.proju(x, v)

    def prior(
        self, shape: tuple[int, ...], device: str | torch.device = "cpu"
    ) -> Tensor:
        # return self.sphere.random_uniform(shape).to(device)
        return F.normalize(torch.randn(shape, device=device), dim=-1)

    def logp0(self, x: Tensor) -> Tensor:
        dim = x.shape[-1]
        return torch.full_like(
            x[..., 0],
            math.lgamma(dim / 2) - (math.log(2) + (dim / 2) * math.log(math.pi)),
        )

    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        return self.sphere.inner(p, u, v)

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        return self.sphere.transp(p, q, v)

    def all_belong_tangent(self, x: Tensor, v: Tensor) -> bool:
        return torch.allclose((x * v).sum(dim=-1), torch.tensor(0.0), atol=1e-5)


class POSphere(Sphere):
    """Positive orthant of the sphere manifold."""

    def get_type(self) -> ManifoldType:
        return ManifoldType.SPHERE

    def project(self, x: Tensor) -> Tensor:
        return super().project(x).abs()

    def prior(self, *args, **kwargs) -> Tensor:
        return super().prior(*args, **kwargs).abs()

    def all_belong(self, x: Tensor) -> bool:
        return super().all_belong(x) and bool((x >= 0).all().item())


def _flatten_so3_ret(func):
    def wrapper(*args, **kwargs) -> Tensor:
        x = func(*args, **kwargs)
        if isinstance(x, Tensor) and x.shape[-2:] == (3, 3):
            return x.reshape(x.shape[0], 9)
        return x

    return wrapper


def _unflatten_so3_args(func):
    def wrapper(*args, **kwargs) -> Tensor:
        adjusted_args = []
        for arg in args:
            if isinstance(arg, Tensor) and arg.shape[-1] == 9:
                adjusted_args.append(arg.reshape(-1, 3, 3))
            else:
                adjusted_args.append(arg)
        return func(*adjusted_args, **kwargs)

    return wrapper

  
def matrix_to_quaternion(matrix):
    num_rots = matrix.shape[0]
    matrix_diag = torch.diagonal(matrix, dim1=-2, dim2=-1)
    matrix_trace = torch.sum(matrix_diag, dim=-1, keepdim=True)
    decision = torch.cat((matrix_diag, matrix_trace), dim=-1)
    choice = torch.argmax(decision, dim=-1)
    quat = torch.zeros((num_rots, 4), dtype=matrix.dtype, device=matrix.device)

    # Indices where choice is not 3
    not_three_mask = choice != 3
    i = choice[not_three_mask]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[not_three_mask, i] = (
        1 - decision[not_three_mask, 3] + 2 * matrix[not_three_mask, i, i]
    )
    quat[not_three_mask, j] = (
        matrix[not_three_mask, j, i] + matrix[not_three_mask, i, j]
    )
    quat[not_three_mask, k] = (
        matrix[not_three_mask, k, i] + matrix[not_three_mask, i, k]
    )
    quat[not_three_mask, 3] = (
        matrix[not_three_mask, k, j] - matrix[not_three_mask, j, k]
    )

    # Indices where choice is 3
    three_mask = ~not_three_mask
    quat[three_mask, 0] = matrix[three_mask, 2, 1] - matrix[three_mask, 1, 2]
    quat[three_mask, 1] = matrix[three_mask, 0, 2] - matrix[three_mask, 2, 0]
    quat[three_mask, 2] = matrix[three_mask, 1, 0] - matrix[three_mask, 0, 1]
    quat[three_mask, 3] = 1 + decision[three_mask, 3]

    return _normalize_quaternion(quat)


def _normalize_quaternion(quat):
    return quat / torch.norm(quat, dim=-1, keepdim=True)


def quaternion_to_axis_angle(quat, degrees=False, eps=1e-6):
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    angle = 2.0 * torch.atan2(torch.norm(quat[..., :3], dim=-1), quat[..., 3])
    angle2 = angle * angle
    small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_scale = angle / torch.sin(angle / 2 + eps)
    scale = torch.where(angle <= 1e-3, small_scale, large_scale)

    if degrees:
        scale = torch.rad2deg(scale)

    return scale[..., None] * quat[..., :3]


def matrix_to_axis_angle(matrix):
    # Check if matrix has 3 dimensions and last two dimensions have shape 3
    if len(matrix.shape) != 3 or matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


# Orthonormal basis of SO(3) with shape [3, 3, 3]
basis = torch.tensor(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
)


# hat map from vector space R^3 to Lie algebra so(3)
def my_hat(v):
    return torch.einsum("...i,ijk->...jk", v, basis.to(v))


# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
# def Log(R): return torch.tensor(Rotation.from_matrix(R.numpy()).as_rotvec())


def Log(R):
    return matrix_to_axis_angle(R)


# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R):
    return my_hat(Log(R))


class SO3(Manifold):
    def __init__(self):
        self.so = geomstats.geomstats.geometry.special_orthogonal.SpecialOrthogonal(
            3,
            point_type="matrix",
        )
        self.so_vec = geomstats.geomstats.geometry.special_orthogonal.SpecialOrthogonal(
            3,
            point_type="vector",
        )

    def get_type(self) -> ManifoldType:
        return ManifoldType.SO3

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        return torch.einsum(
            "...ij,...jk->...ik",
            p,
            torch.linalg.matrix_exp(p.transpose(-1, -2) @ v)
        )

    def log(
        self, p: Tensor, q: Tensor, back_to_mat: bool = True
    ) -> Tensor | tuple[Tensor, Tensor]:
        rot_x0 = matrix_to_axis_angle(p)
        rot_x1 = matrix_to_axis_angle(q)
        # NOTE: geomstats is (point, base_point) order
        ret = self.so_vec.log_not_from_identity(rot_x1, rot_x0)
        return (
            self.so_vec.matrix_from_rotation_vector(ret) if back_to_mat else ret
        ), rot_x0

    def geodesic_interpolant(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        lg, rot_x0 = self.log(x0, x1, back_to_mat=False)
        t = t.view((t.shape[0],) + (1,) * (len(lg.shape) - 1))
        xt = self.so_vec.exp_not_from_identity(t * lg, rot_x0)
        return self.so_vec.matrix_from_rotation_vector(xt)

    def prior(
        self, shape: tuple[int, ...], device: str | torch.device = "cpu"
    ) -> Tensor:
        return self.so.random_uniform(n_samples=shape[0]).to(
            device, dtype=torch.float32
        )

    def project(self, x: Tensor) -> Tensor:
        return x  # TODO

    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        R = x
        M = v

        skew_symmetric_part = 0.5 * (M - M.transpose(-2, -1))

        # Project onto the tangent space at R
        T = R @ skew_symmetric_part

        return T

    def logp0(self, x: Tensor) -> Tensor:
        # TODO: verify formula
        return torch.full_like(x[..., :1], 2.0 * math.log(8.0 * math.pi))

    def all_belong(self, p: Tensor) -> bool:
        det = torch.allclose(
            torch.vmap(torch.linalg.det)(p),
            torch.tensor(1.0).to(p),
            atol=1e-5,
            rtol=1e-5,
        )
        orth = torch.allclose(
            torch.vmap(lambda x: x.T @ x)(p.view(-1, 3, 3)), torch.eye(3).to(p), atol=1e-5, rtol=1e-5
        )
        return det and orth

    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        x = self._pt_to_identity(p, v)
        return torch.allclose(self.calc_tangent_error(p, v), torch.zeros_like(x), atol=1e-5)

    def calc_tangent_error(self, p: Tensor, v: Tensor) -> Tensor:
        x = self._pt_to_identity(p, v)
        return x + x.transpose(-2, -1)


    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        return so3_relative_angle(p, q)

    def _pt_to_identity(self, R, v):
        return torch.transpose(R, dim0=-2, dim1=-1) @ v

    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # assert torch.allclose(u, v), "not implemented for non-norm"
        # calulate the norm squared of matrix T_R in the tangent space of R
        if torch.allclose(u, v):
            r = self._pt_to_identity(p, u)  # matrix r is in so(3)
            norm = -torch.diagonal(r @ r, dim1=-2, dim2=-1).sum(dim=-1) / 2 # -trace(rTr)/2
        else:
            r = self._pt_to_identity(p, u)  # matrix r is in so(3)
            m = self._pt_to_identity(p, v)  # matrix m is in so(3)
            norm = -torch.diagonal(r @ m, dim1=-2, dim2=-1).sum(dim=-1) / 2  # -trace(rTm)/2
        return norm

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        raise NotImplementedError("not implemented yet")



class SE3(Manifold):
    def __init__(self):
        self.so3 = SO3()
        self.se3 = geomstats.geomstats.geometry.special_euclidean.SpecialEuclidean(
            n=3, point_type="matrix"
        )

    def _flatten_seq_to_batch(self, x: Tensor) -> Tensor:
        """
        Flattens a sequence of matrices to a batch of matrices.
        """
        return rearrange(x, "n l ... -> (n l) ...", n=x.size(0), l=x.size(1))

    def _rearrange_batch_to_seq(self, x: Tensor, n: int) -> Tensor:
        """
        Rearranges a batch of matrices to a sequence of matrices.

        Args:
            x: A tensor of shape (N*L, ...).
            n: The batch size.
        """
        l = x.size(0) // n
        return rearrange(x, "(n l) ... -> n l ...", n=n, l=l)

    def get_type(self) -> ManifoldType:
        return ManifoldType.SE3

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        rot_p = self._flatten_seq_to_batch(p[..., :3, :3])
        rot_v = self._flatten_seq_to_batch(v[..., :3, :3])
        so3_exp = self._rearrange_batch_to_seq(self.so3.exp(rot_p, rot_v), p.size(0))
        t_exp = p[..., :3, 3] + v[..., :3, 3]
        return group_rot_trans(so3_exp, t_exp)

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        rot_p = p[..., :3, :3].view(-1, 3, 3)
        rot_q = q[..., :3, :3].view(-1, 3, 3)
        so3_log, _ = self.so3.log(rot_p, rot_q)
        so3_log = so3_log.view(*p.shape[:-2], 3, 3)
        t_log = q[..., :3, 3] - p[..., :3, 3]
        return group_rot_trans(so3_log, t_log)

    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        # NOTE: not really distance on SE(3)
        p_rot = p[..., :3, :3]
        q_rot = q[..., :3, :3]
        p_rot = self._flatten_seq_to_batch(p_rot)
        q_rot = self._flatten_seq_to_batch(q_rot)
        rot_dist = self._rearrange_batch_to_seq(self.so3.distance(p_rot, q_rot), p.size(0)).square().sum(dim=-1)
        trans_dist = torch.square(p[..., :3, 3] - q[..., :3, 3]).sum(dim=(-1, -2))
        return torch.sqrt(rot_dist + trans_dist)

    def geodesic_interpolant(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # first do on SO(3)
        rot_x0 = self._flatten_seq_to_batch(x0[..., :3, :3])
        rot_x1 = self._flatten_seq_to_batch(x1[..., :3, :3])
        rot_xt = self._rearrange_batch_to_seq(
            self.so3.geodesic_interpolant(
                rot_x0,
                rot_x1,
                t.squeeze().repeat_interleave(x0.size(1)),
            ), x0.size(0)
        )
        # do on translations
        trans_x0 = x0[..., :3, 3]
        trans_x1 = x1[..., :3, 3]
        t = _time_ndims(t, trans_x0)
        trans_xt = (1.0 - t) * trans_x0 + t * trans_x1
        return group_rot_trans(rot_xt, trans_xt)

    def all_belong(self, x: Tensor) -> bool:
        return self.so3.all_belong(x[..., :-1, :-1])

    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        # return self.so3.all_belong_tangent(p[..., :3, :3], v[..., :3, :3])
        raise NotImplementedError("not implemented yet")

    def project(self, x: Tensor) -> Tensor:
        rot = self.so3.project(x[..., :3, :3])
        trans = x[..., :3, 3]
        return group_rot_trans(rot, trans)

    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        rot = self.so3.project_tangent(x[..., :3, :3], v[..., :3, :3])
        trans = v[..., :3, 3]
        return group_rot_trans(rot, trans)

    def prior(
        self, shape: tuple[int, ...], device: str | torch.device = "cpu"
    ) -> Tensor:
        n_samples = int(np.prod(shape[:-2]))
        rot = self.so3.prior((n_samples,), device=device).reshape(*shape[:-2], 3, 3)
        trans = torch.randn(*shape[:-2], 3, device=device)
        return group_rot_trans(rot, trans)

    def logp0(self, x: Tensor) -> Tensor:
        raise NotImplementedError("not implemented yet")

    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        assert torch.allclose(u, v), "not implemented for non-norm"
        rot_inner = self.so3.inner(p[..., :3, :3], u[..., :3, :3], v[..., :3, :3])
        trans_inner = (u[..., :3, 3] * v[..., :3, 3]).sum(dim=-1)
        return rot_inner + trans_inner

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        raise NotImplementedError("not implemented yet")


class PoincareBall(Manifold):
    # NOTE: k is explicited almost everywhere for JVP

    def __init__(self):
        self.geoopt = GeooptPoincareBall()

    def get_type(self) -> ManifoldType:
        return ManifoldType.POINCARE_BALL

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1):
        k = -1.0
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
        denom = 1 - 2 * k * xy + k**2 * x2 * y2
        # minimize denom (omit K to simplify th notation)
        # 1)
        # {d(denom)/d(x) = 2 y + 2x * <y, y> = 0
        # {d(denom)/d(y) = 2 x + 2y * <x, x> = 0
        # 2)
        # {y + x * <y, y> = 0
        # {x + y * <x, x> = 0
        # 3)
        # {- y/<y, y> = x
        # {- x/<x, x> = y
        # 4)
        # minimum = 1 - 2 <y, y>/<y, y> + <y, y>/<y, y> = 0
        return num / denom.clamp_min(1e-15)

    def exp(self, p: Tensor, v: Tensor) -> Tensor:
        x = p
        u = v
        dim = -1
        # copy content to avoid JVP issues
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
        lam = self._lambda_x(x)
        second_term = self.tan_k((lam / 2.0) * u_norm) * (u / u_norm)
        y = self._mobius_add(x, second_term, dim=dim)
        return y
        # return GeooptPoincareBall().expmap(p, v)

    def artanh(self, x: Tensor):
        x = x.clamp(-1 + 1e-7, 1 - 1e-7)
        return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)

    def artan_k(self, x: Tensor):
        return self.artanh(x)

    def log(self, p: Tensor, q: Tensor) -> Tensor:
        x = p
        y = q
        dim = -1
        # return self.geoopt.logmap(p, q)
        # return stereo_math.logmap(p, q, k=torch.tensor(-1.0, device=p.device))
        sub = self._mobius_add(-x, y, dim=dim)
        sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
        lam = self._lambda_x(x)
        return 2.0 * self.artan_k(sub_norm) * (sub / (lam * sub_norm))
        # return GeooptPoincareBall().logmap(p, q)

    def inner(self, p: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # return self.geoopt.inner(p, u, v)
        return stereo_math.inner(p, u, v, k=torch.tensor(-1.0, device=p.device))
        # return GeooptPoincareBall().inner(p, u, v)

    def prior(self, shape: tuple[int, ...], device: str | torch.device = "cpu") -> Tensor:
        distance = 0.6
        std = 0.7
        sign0 = torch.tensor(-1.0, device=device)
        mean0 = torch.tensor([distance, distance], device=device) * sign0

        v = torch.randn(shape, device=device) * std
        lambda_x = self._lambda_x(mean0).unsqueeze(-1)
        return self.exp(mean0, v / lambda_x)

    def logp0(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def parallel_transport(self, p: Tensor, q: Tensor, v: Tensor) -> Tensor:
        raise NotImplementedError()

    def project_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        return self.geoopt.proju(x, v)

    def all_belong(self, p: Tensor) -> bool:
        return self.geoopt.check_point_on_manifold(p)

    def all_belong_tangent(self, p: Tensor, v: Tensor) -> bool:
        return self.geoopt.check_vector_on_tangent(p, v)

    def tan_k(self, x: torch.Tensor):
        k_sqrt = torch.tensor(1.0, device=x.device)
        scaled_x = x * k_sqrt

        return k_sqrt.reciprocal() * scaled_x.clamp(-15, 15).tanh()

    def project(self, x: Tensor) -> Tensor:
        dim = -1
        # return self.geoopt.projx(x)
        # return stereo_math.project(x, k=torch.tensor(-1.0, device=x.device))
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
        maxnorm = (1 - eps)
        # maxnorm = torch.where(k.lt(0), maxnorm, k.new_full((), 1e15))
        norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
        # return GeooptPoincareBall().projx(x)

    def distance(self, p: Tensor, q: Tensor) -> Tensor:
        # return self.geoopt.dist(p, q)
        return GeooptPoincareBall().dist(p, q)

    def _lambda_x(self, x) -> Tensor:
        # lambda_x = 2 / (1 + k * x.pow(2).sum(dim=-1, keepdim=True)).clamp_min(1e-15)
        # for k = -1
        return 2 / (x.pow(2).sum(dim=-1, keepdim=True)).clamp_min(1e-15)

    def metric_normalize(self, x: Tensor, v: Tensor) -> Tensor:
        # return v / self.geoopt.lambda_x(x, keepdim=True)
        return v / self._lambda_x(x)
        # return v / GeooptPoincareBall().lambda_x(x, keepdim=True)
        # return v


def manifold_from_name(name: str) -> Manifold:
    """
    Returns a manifold instance from its name.
    """
    return {
        "euclidean": EuclideanSpace(),
        "simplex": FRSimplex(),
        "sphere": Sphere(),
        "flat_torus": FlatTorus(),
        "so3": SO3(),
        # NOTE: do not use se3
        # "se3": SE3(),
        "po-sphere": POSphere(),
        "flat-simplex": FlatSimplex(),
        "hyperbolic": PoincareBall(),
    }[name]
if "__main__" == __name__:
    print('success import')