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

from ..core import Intrinsic
from ..core.package import Package


class NoIntrinsic(Intrinsic):
    r"""

    Subclass of the :class:`pycvcam.core.Intrinsic` class that represents a no intrinsic model.

    .. note::

        This class represents the intrinsic transformation, which is the last step of the process from the ``world_points`` to the ``image_points``.

    The ``NoIntrinsic`` model is a special case of the intrinsic transformation where no intrinsic transformations are applied.

    Lets consider ``distorted_points`` in the camera normalized coordinate system :math:`\vec{x}_d = (x_d, y_d)`, the corresponding ``image_points`` in the image coordinate system are given by :math:`\vec{x}_i = (x_d, y_d)`. 
    Simply applying an identity transformation, which means that the image points are equal to the distorted points.

    """
    def __init__(self):
        super().__init__(parameters=None, constants=None)

    # =============================================
    # Overwritten properties
    # =============================================
    @property
    def parameters(self) -> None:
        r"""
        Always returns None, as there are no parameters for the no intrinsic model.
        """
        return None
    
    @parameters.setter
    def parameters(self, value: None):
        if value is not None:
            raise ValueError("NoIntrinsic model has no parameters, must be set to None.")
        self._parameters = None

    @property
    def constants(self) -> None:
        r"""
        Always returns None, as there are no constants for the no intrinsic model.
        """
        return None
    
    @constants.setter
    def constants(self, value: None):
        if value is not None:
            raise ValueError("NoIntrinsic model has no constants, must be set to None.")
        self._constants = None

    def is_set(self) -> bool:
        r"""
        Always returns True, as the no intrinsic model is always set and does not require any parameters or constants.
        """
        return True
    
    # =============================================
    # Implementing the transform and inverse_transform methods
    # =============================================
    def _transform(self, distorted_points: numpy.ndarray, *, dx = False, dp = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the ``distorted_points`` to the ``image_points``.

        Lets consider ``distorted_points`` in the camera normalized coordinate system :math:`\vec{x}_d = (x_d, y_d)`, the corresponding ``image_points`` in the image coordinate system are given by :math:`\vec{x}_i = (x_d, y_d)`. 
        Simply applying an identity transformation, which means that the image points are equal to the distorted points.

        The jacobians with respect to the intrinsic parameters is an empty array with shape (Npoints, 2, 0), as there are no parameters to compute the jacobian for.
        The jacobian with respect to the distorted points is set to the identity matrix, as the distorted points are equal to the image points.

        .. warning::

            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``distorted_points`` is (Npoints, 2) before calling this method.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points in camera normalized coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the distorted points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the intrinsic parameters is computed. Default is False

        Returns
        -------
        image_points : numpy.ndarray
            The image points in image coordinates, which are equal to the x and y componants of the distorted points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the image points with respect to the distorted points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the image points with respect to the intrinsic parameters. Shape (Npoints, 2, 0) if dp is True, otherwise None.
        """
        image_points = distorted_points.copy() # shape (Npoints, 2)
        jacobian_dx = None # shape (Npoints, 2, 2)
        jacobian_dp = None # shape (Npoints, 2, Nparams)
        if dx:
            jacobian_dx = numpy.zeros((image_points.shape[0], 2, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 2)
            jacobian_dx[:, 0, 0] = 1.0
            jacobian_dx[:, 1, 1] = 1.0
        if dp:
            jacobian_dp = numpy.empty((image_points.shape[0], 2, 0), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 0)
        return image_points, jacobian_dx, jacobian_dp
    
    
    def _inverse_transform(self, image_points: numpy.ndarray, *, dx = False, dp = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from the ``image_points`` to the ``distorted_points``.

        Lets consider ``image_points`` in the image coordinate system :math:`\vec{x}_i = (x_i, y_i)`, the corresponding ``distorted_points`` in the camera normalized coordinate system are given by :math:`\vec{x}_d = (x_i, y_i)`. 
        Simply applying an identity transformation, which means that the image points are equal to the distorted points.

        The jacobians with respect to the intrinsic parameters is an empty array with shape (Npoints, 2, 0), as there are no parameters to compute the jacobian for.
        The jacobian with respect to the image points is set to the identity matrix, as the distorted points are equal to the image points.

        .. warning::

            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``image_points`` is (Npoints, 2) before calling this method.

        Parameters
        ----------
        image_points : numpy.ndarray
            The image points in image coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the image points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the intrinsic parameters is computed. Default is False

        Returns
        -------
        distorted_points : numpy.ndarray
            The distorted points in camera normalized coordinates, which are equal to the x and y components of the image points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the image points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the intrinsic parameters. Shape (Npoints, 2, 0) if dp is True, otherwise None.
        """
        return self._transform(image_points, dx=dx, dp=dp)