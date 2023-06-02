import warnings

import numpy as np
from scipy.special import binom


def bernstein(n, k):
    """Bernstein polynomial."""
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x**k * (1 - x) ** (n - k)

    return _bpoly


def bezier(points, at):
    """Build Bézier curve from points."""
    warnings.warn(
        message="Deprecated. CatmulClark builds nicer splines.",
        category=FutureWarning,
        stacklevel=1,
    )

    at = np.asarray(at)
    at_flat = at.ravel()
    n = len(points)
    curve = np.zeros((at_flat.shape[0], 2))
    for ii in range(n):
        curve += np.outer(bernstein(n - 1, ii)(at_flat), points[ii])
    return curve.reshape((*at.shape, 2))


def catmul_clark(points, at):
    points = np.asarray(points)

    while len(points) < len(at):
        new_p = np.zeros((2 * len(points), 2))
        new_p[0] = points[0]
        new_p[-1] = points[-1]
        new_p[1:-2:2] = 3 / 4.0 * points[:-1] + 1 / 4.0 * points[1:]
        new_p[2:-1:2] = 1 / 4.0 * points[:-1] + 3 / 4.0 * points[1:]
        points = new_p
    xp, yp = zip(*points)
    xp = np.interp(at, np.linspace(0, 1, len(xp)), xp)
    yp = np.interp(at, np.linspace(0, 1, len(yp)), yp)
    return np.asarray(list(zip(xp, yp)))


curve_method = {
    "Bezier": bezier,
    "CatmulClark": catmul_clark,
}
