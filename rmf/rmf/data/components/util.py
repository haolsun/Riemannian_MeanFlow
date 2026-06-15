import torch


def cartesian_from_latlon(x):
    """Embedded 3D unit vector from spherical polar coordinates.

    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.

    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    assert x.shape[-1] == 2
    lat = x.select(-1, 0)
    lon = x.select(-1, 1)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)


def lonlat_from_cartesian(x):
    r = x.pow(2).sum(-1).sqrt()
    x, y, z = x[..., 0], x[..., 1], x[..., 2]
    lat = torch.asin(z / r)
    lon = torch.atan2(y, x)
    return torch.cat([lon.unsqueeze(-1), lat.unsqueeze(-1)], dim=-1)
