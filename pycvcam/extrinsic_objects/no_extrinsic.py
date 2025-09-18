# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple
import numpy

from ..core import Extrinsic
from ..core.package import Package

class NoExtrinsic(Extrinsic):
    r"""

    Subclass of the :class:`pycvcam.core.Extrinsic` class that represents a no extrinsic model.

    .. note::

        This class represents the extrinsic transformation, which is the first step of the process from the ``world_points`` to the ``image_points``.

    The ``NoExtrinsic`` model is a special case of the extrinsic transformation where no extrinsic transformations are applied.

    Lets consider ``world_points`` in the global coordinate system :math:`\vec{X}_w = (X_w, Y_w, Z_w)`, the corresponding ``normalized_points`` in the camera normalized coordinate system are given by :math:`\vec{x}_n = (X_w, Y_w)`. 
    Simply ignoring the z-coordinate, which is always set to 1 for the normalization plane.

    """
    def __init__(self):
        super().__init__(parameters=None, constants=None)

    # =============================================
    # Overwritten properties
    # =============================================
    @property
    def parameters(self) -> None:
        r"""
        Always returns None, as there are no parameters for the no extrinsic model.
        """
        return None
    
    @parameters.setter
    def parameters(self, value: None):
        if value is not None:
            raise ValueError("NoExtrinsic model has no parameters, must be set to None.")
        self._parameters = None

    @property
    def constants(self) -> None:
        r"""
        Always returns None, as there are no constants for the no extrinsic model.
        """
        return None
    
    @constants.setter
    def constants(self, value: None):
        if value is not None:
            raise ValueError("NoExtrinsic model has no constants, must be set to None.")
        self._constants = None

    def is_set(self) -> bool:
        r"""
        Always returns True, as the no extrinsic model is always set and does not require any parameters or constants.
        """
        return True
    
    # =============================================
    # Implementing the transform and inverse_transform methods
    # =============================================
    def _transform(self, world_points: numpy.ndarray, *, dx = False, dp = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the ``world_points`` to the ``normalized_points``.

        Lets consider ``world_points`` in the global coordinate system :math:`\vec{X}_w = (X_w, Y_w, Z_w)`, the corresponding ``normalized_points`` in the camera normalized coordinate system are given by :math:`\vec{x}_n = (X_w, Y_w)`. 
        Simply ignoring the z-coordinate, which is always set to 1 for the normalization plane.

        The jacobians with respect to the extrinsic parameters is an empty array with shape (Npoints, 2, 0), as there are no parameters to compute the jacobian for.
        The jacobian with respect to the world points is set to the identity matrix (for x and y) and zero for z, as the world points are equal to the normalized points.

        .. warning::

            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``world_points`` is (Npoints, 3) before calling this method.

        Parameters
        ----------
        world_points : numpy.ndarray
            The world points in global coordinates to be transformed. Shape (Npoints, 3).

        dx : bool, optional
            If True, the jacobian with respect to the world points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the extrinsic parameters is computed. Default is False

        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points in camera normalized coordinates, which are equal to the x and y componants of the world points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the normalized points with respect to the world points. Shape (Npoints, 2, 3) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the normalized points with respect to the extrinsic parameters. Shape (Npoints, 2, 0) if dp is True, otherwise None.
        """
        normalized_points = world_points[:, :2].copy() # shape (Npoints, 2)
        jacobian_dx = None # shape (Npoints, 2, 2)
        jacobian_dp = None # shape (Npoints, 2, Nparams)
        if dx:
            jacobian_dx = numpy.zeros((normalized_points.shape[0], 2, 3), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 3)
            jacobian_dx[:, 0, 0] = 1.0
            jacobian_dx[:, 1, 1] = 1.0
        if dp:
            jacobian_dp = numpy.empty((normalized_points.shape[0], 2, 0), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 0)
        return normalized_points, jacobian_dx, jacobian_dp
    
    
    def _inverse_transform(self, normalized_points: numpy.ndarray, *, dx = False, dp = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from the ``normalized_points`` to the ``world_points``.

        Lets consider ``normalized_points`` in the camera normalized coordinate system :math:`\vec{x}_n = (x_n, y_n)`, the corresponding ``world_points`` in the global coordinate system are given by :math:`\vec{X}_w = (x_n, y_n, 1)`.
        Simply adding a z-coordinate of 1 for the normalization plane.

        The jacobians with respect to the extrinsic parameters is an empty array with shape (Npoints, 2, 0), as there are no parameters to compute the jacobian for.
        The jacobian with respect to the normalized points is set to the identity matrix (for x and y), as the normalized points are equal to the world points.

        .. warning::

            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``normalized_points`` is (Npoints, 2) before calling this method.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in camera normalized coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the normalized points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the extrinsic parameters is computed. Default is False

        Returns
        -------
        world_points : numpy.ndarray
            The world 3D points in global coordinates, which are equal to the normalized points with z=1. Shape (Npoints, 3).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the world 3D points with respect to the normalized points. Shape (Npoints, 3, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the world 3D points with respect to the extrinsic parameters. Shape (Npoints, 3, 0) if dp is True, otherwise None.
        """
        world_points = numpy.empty((normalized_points.shape[0], 3), dtype=Package.get_float_dtype()) # shape (Npoints, 3)
        world_points[:, :2] = normalized_points.copy() # copy x and y coordinates
        world_points[:, 2] = 1.0 # set z coordinate
        jacobian_dx = None # shape (Npoints, 2, 2)
        jacobian_dp = None # shape (Npoints, 2, Nparams)
        if dx:
            jacobian_dx = numpy.zeros((normalized_points.shape[0], 3, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 3, 2)
            jacobian_dx[:, 0, 0] = 1.0
            jacobian_dx[:, 1, 1] = 1.0
        if dp:
            jacobian_dp = numpy.empty((normalized_points.shape[0], 2, 0), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 0)
        return normalized_points, jacobian_dx, jacobian_dp

    # =============================================
    # Implementing the rays computation
    # =============================================
    def _compute_rays(self, normalized_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Computes the rays from the camera to the scene for the no extrinsic model in the world coordinate system.

        The ray structure is as follows:

        - The first 3 elements are the origin of the ray in the world coordinate system (the normalized points with z=1)
        - The last 3 elements are the direction of the ray in the world coordinate system, which is always (0, 0, 1) for the no extrinsic model. The direction vector is normalized.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in the camera coordinate system. Shape (Npoints, 2).

        Returns
        -------
        rays : numpy.ndarray
            The rays in the world coordinate system. Shape (Npoints, 6).
        """
        rays = numpy.empty((normalized_points.shape[0], 6), dtype=Package.get_float_dtype())
        rays[:, :2] = normalized_points.copy()  # copy x and y coordinates
        rays[:, 2] = 1.0  # set z coordinate to 1
        rays[:, 3] = 0.0  # direction x
        rays[:, 4] = 0.0  # direction y
        rays[:, 5] = 1.0  # direction z
        return rays