from dataclasses import dataclass
from typing import Optional
import numpy

@dataclass(slots=True)
class Rays:
    r"""
    A class to represent a collection of rays in 3D space.
    
    This class is returned by the `compute_rays` method of the :class:`pycvcam.core.Extrinsic` class, which computes the rays in the world coordinate system for the given normalized points.

    The rays are storred in a numpy array with shape (..., 6), where the last dimension represents the x, y, and z coordinates of the origin and the dx, dy, and dz components of the direction of the ray.
    
    To view on the rays are available:

    - ``origins``: The origins of the rays in the world coordinate system with shape (..., 3).
    - ``directions``: The directions of the rays in the world coordinate system with shape (..., 3).
    
    .. note::

        If ``transpose`` is set to True during the transformation, the output rays will have shape (6, ...) instead of (..., 6). Same for the origins and directions (i.e., shape (3, ...) instead of (..., 3) respectively).

    Attributes
    ----------
    rays : numpy.ndarray
        The rays in the world coordinate system. Shape (..., 6).

    transpose : bool, optional
        If True, the output rays will have shape (6, ...) instead of (..., 6). True if set during the transformation, otherwise False.

    """
    rays: numpy.ndarray
    transpose: bool = False

    @property
    def origins(self) -> numpy.ndarray:
        r"""
        Returns the origins of the rays in the world coordinate system.

        Returns
        -------
        numpy.ndarray
            The origins of the rays in the world coordinate system. Shape (..., 3).
        """
        if self.rays is None:
            raise ValueError("Rays are not computed yet.")
        if self.transpose:
            return self.rays[:3, ...]
        return self.rays[..., :3]
    
    @property
    def directions(self) -> numpy.ndarray:
        r"""
        Returns the directions of the rays in the world coordinate system.

        Returns
        -------
        numpy.ndarray
            The directions of the rays in the world coordinate system. Shape (..., 3).
        """
        if self.rays is None:
            raise ValueError("Rays are not computed yet.")
        if self.transpose:
            return self.rays[3:, ...]
        return self.rays[..., 3:]



