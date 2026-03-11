from typing import Sequence, Optional
from numpy.typing import ArrayLike

import numpy
import scipy

from ..core.rays import Rays


def optimize_rays_intersect(
    seq_rays: Sequence[Rays],
) -> numpy.ndarray:
    r"""
    Optimize the intersection point of multiple rays.

    Lets consider :math:`N` rays defined by their origins :math:`\mathbf{o}_i`
    and directions :math:`\mathbf{d}_i` for :math:`i = 1, 2, ..., N`.
    The goal is to find the point :math:`\mathbf{X}` that minimizes the sum of
    squared distances to all rays, which can be formulated as:

    .. math::

        \mathbf{X} = \arg\min_{\mathbf{X}} \sum_{i=1}^{N} \left\| (\mathbf{I} - \mathbf{d}_i \mathbf{d}_i^T) (\mathbf{X} - \mathbf{o}_i) \right\|^2


    where :math:`\mathbf{I}` is the identity matrix.
    This optimization problem can be solved using least squares, leading to a closed-form solution for the optimal intersection point.

    .. note::

        At least two rays are required for this optimization to be meaningful,
        but more rays can be used to improve the accuracy of the intersection point.


    Parameters
    ----------
    seq_rays : Sequence[Rays]
        A sequence of Rays objects, each representing n_points rays.
        The intersection point is optimized to minimize the distance to all rays.


    Returns
    -------
    numpy.ndarray
        The optimized intersection point, shape (n_points, 3).

    """
    if not isinstance(seq_rays, Sequence) or len(seq_rays) < 2:
        raise ValueError("seq_rays must be a sequence of at least two Rays objects.")
    if not all(isinstance(rays, Rays) for rays in seq_rays):
        raise ValueError("All elements of seq_rays must be Rays objects.")

    seq_origins = [rays.origins for rays in seq_rays]
    seq_directions = [rays.directions for rays in seq_rays]
    n_points = seq_rays[0].origins.shape[0]

    if not all(rays.origins.shape[0] == n_points for rays in seq_rays):
        raise ValueError(
            "All Rays objects in seq_rays must have the same number of points."
        )

    I = numpy.eye(3)

    A = numpy.zeros((n_points, 3, 3), dtype=numpy.float64)
    b = numpy.zeros((n_points, 3), dtype=numpy.float64)

    for origins, directions in zip(seq_origins, seq_directions):

        # normalize directions
        d = directions / numpy.linalg.norm(directions, axis=1, keepdims=True)

        # outer product d d^T
        ddT = numpy.einsum("ni,nj->nij", d, d)

        # projection matrix
        M = I - ddT

        A += M
        b += numpy.einsum("nij,nj->ni", M, origins)

    X = numpy.linalg.solve(A, b[..., None])[..., 0]

    return X
