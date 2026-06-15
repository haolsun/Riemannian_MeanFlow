import math
from functools import partial
from typing import Optional

import ot  # POT library for optimal transport
import torch


def pairwise_euclidean_distance(x0, x1):
    """
    Compute pairwise Euclidean distances between two sets of points in ℝ³.

    Parameters
    ----------
    x0 : torch.Tensor of shape (n_samples_x0, 3)
        Source points in ℝ³.
    x1 : torch.Tensor of shape (n_samples_x1, 3)
        Target points in ℝ³.

    Returns
    -------
    torch.Tensor of shape (n_samples_x0, n_samples_x1)
        Pairwise Euclidean distance matrix.
    """
    diff = x0[:, None, :] - x1[None, :, :]
    M = torch.norm(diff, dim=2)
    return M


def pairwise_geodesic_distance(x0, x1):
    """
    Compute pairwise geodesic distances between two sets of rotations in SO(3).

    Parameters
    ----------
    x0 : torch.Tensor of shape (n_samples_x0, 3, 3)
        Source rotations as rotation matrices.
    x1 : torch.Tensor of shape (n_samples_x1, 3, 3)
        Target rotations as rotation matrices.

    Returns
    -------
    torch.Tensor of shape (n_samples_x0, n_samples_x1)
        Pairwise geodesic distance matrix.
    """
    n_samples_x0 = x0.shape[0]
    n_samples_x1 = x1.shape[0]

    # Expand dimensions to compute all pairwise products
    x0_exp = x0.unsqueeze(1).expand(n_samples_x0, n_samples_x1, 3, 3)
    x1_exp = x1.unsqueeze(0).expand(n_samples_x0, n_samples_x1, 3, 3)

    # Compute relative rotations
    #TODO: Check if everything is at a certain point(collapse)
    rel_rot = torch.matmul(x0_exp.transpose(-1, -2), x1_exp)

    # Compute the trace of the relative rotations
    trace_rel_rot = rel_rot.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)

    # Compute the angle between rotations
    cos_theta = (trace_rel_rot - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp to avoid numerical errors
    theta = torch.acos(cos_theta)

    return theta


def wasserstein_distance(
    x0: torch.Tensor,
    x1: torch.Tensor,
    space: str = "so3",
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """
    Compute the Wasserstein distance between two distributions in either ℝ³ or SO(3).

    Parameters
    ----------
    x0 : torch.Tensor
        Source samples.
        - For 'r3': shape (n_samples_x0, 3)
        - For 'so3': shape (n_samples_x0, 3, 3)
    x1 : torch.Tensor
        Target samples.
        - For 'r3': shape (n_samples_x1, 3)
        - For 'so3': shape (n_samples_x1, 3, 3)
    space : str, optional
        The space to compute distances in ('r3' or 'so3'). Default is 'so3'.
    method : str, optional
        Method for computing the Wasserstein distance ('exact' or 'sinkhorn'). Default is 'exact'.
    reg : float, optional
        Regularization coefficient for the 'sinkhorn' method. Default is 0.05.
    power : int, optional
        Power of the Wasserstein distance (1 or 2). Default is 2.
    **kwargs
        Additional arguments passed to the optimal transport function.

    Returns
    -------
    float
        The computed Wasserstein distance.
    """
    assert power in [1, 2], "Power must be 1 or 2."
    assert space in ["r3", "so3"], "Space must be 'r3' or 'so3'."

    # Select the optimal transport function
    if method == "exact" or method is None:
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create uniform weight vectors for the distributions
    a = ot.unif(x0.shape[0])
    b = ot.unif(x1.shape[0])

    # Compute the cost matrix based on the specified space
    if space == "r3":
        M = pairwise_euclidean_distance(x0, x1)
    elif space == "so3":
        M = pairwise_geodesic_distance(x0, x1)

    # Adjust the cost matrix based on the specified power
    if power == 2:
        M = M**2

    # Compute the Wasserstein distance
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7, **kwargs)

    # Adjust the result for Wasserstein-2 distance
    if power == 2:
        ret = math.sqrt(ret)

    return ret


if __name__ == "__main__":
    # Example usage for distributions in ℝ³
    x0_r3 = torch.randn(100, 3)  # Source samples in ℝ³
    x1_r3 = torch.randn(100, 3)  # Target samples in ℝ³

    wd_r3 = wasserstein_distance(x0_r3, x1_r3, space="r3", method="exact", power=2)
    print(f"Wasserstein distance in ℝ³: {wd_r3}")

    # Example usage for distributions in SO(3)
    # Generate random rotation matrices for SO(3)
    def random_rotation_matrices(n):
        from scipy.stats import special_ortho_group

        matrices = [torch.tensor(special_ortho_group.rvs(3)) for _ in range(n)]
        return torch.stack(matrices)

    x0_so3 = random_rotation_matrices(100)  # Source rotations in SO(3)
    x1_so3 = random_rotation_matrices(100)  # Target rotations in SO(3)

    wd_so3 = wasserstein_distance(x0_so3, x1_so3, space="so3", method="exact", power=2)
    print(f"Wasserstein distance in SO(3): {wd_so3}")
