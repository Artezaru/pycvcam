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

from typing import ClassVar, List
from abc import abstractmethod
import numpy

from .transform import Transform, TransformResult
from .rays import Rays
from .package import Package

class Extrinsic(Transform):
    r"""
    .. note::

        This class represents the extrinsic transformation, which is the first step of the process from the ``world_points`` to the ``image_points``.
        This is the abstract class that defines the interface for extrinsic transformations.

    This base class, set the ``input_dim`` and ``output_dim`` to 3 and 2 respectively and add the aliases :

    - ``normalized_points`` and ``x_n`` for the result of the transformation.
    - ``world_points`` and ``x_w`` for the result of the inverse transformation.

    To process the transformation, the method ``project`` and ``unproject`` are provided, which are aliases for the ``transform`` and ``inverse_transform`` methods respectively.

    The extrinsic transformation also implement the method ``compute_rays`` to compute the rays emitted by the camera from the world points, which is useful for ray tracing or other applications.
    """
    _input_dim : ClassVar[int] = 3
    _output_dim : ClassVar[int] = 2

    # =============================================
    # Addind aliases for the transformation
    # =============================================
    def _get_transform_aliases(self) -> List[str]:
        r"""
        Property to return a list of aliases for the transformed points.

        - ``normalized_points`` and ``x_n`` for the result of the transformation.
        
        Returns
        -------
        List[str]
            A list of aliases for the transformed points.
        """
        return ["normalized_points", "x_n"]

    def _get_inverse_transform_aliases(self) -> List[str]:
        r"""
        Property to return a list of aliases for the inverse transformed points.

        - ``world_points`` and ``x_w`` are added.

        Returns
        -------
        List[str]
            A list of aliases for the inverse transformed points.
        """
        return ["world_points", "x_w"]
    
    def project(
        self, 
        world_points: numpy.ndarray, 
        *, 
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        **kwargs
        ) -> TransformResult:
        r"""
        Alias for the ``transform`` method, which applies the extrinsic transformation to the points.

        .. seealso::

            - :meth:`pycvcam.core.Transform.transform` for applying the transformation to points.

        .. code-block:: python

            extrinsic = ... # An instance of a subclass of Extrinsic

            import numpy

            world_points = numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]) # shape (Npoints, 3)
            result = extrinsic.project(world_points)
            normalized_points = result.normalized_points  # shape (Npoints, 2)

            # SAME AS:
            result = extrinsic.transform(world_points)
            normalized_points = result.transformed_points  # shape (Npoints, 2)

        Parameters
        ----------
        world_points : numpy.ndarray
            The world points to be transformed. Shape (..., 3).

        transpose : bool, optional
            If True, the input points are assumed to have shape (3, ...) instead of (..., 3) and the output points will have shape (2, ...). Default is False.

        dx : bool, optional
            If True, the jacobian with respect to the world points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the distortion parameters is computed. Default is False

        Returns
        -------
        TransformResult
            The result of the transformation, which includes the ``normalized_points`` and the Jacobian matrices if available.
        """
        return self.transform(world_points, transpose=transpose, dx=dx, dp=dp, **kwargs)

    def unproject(
        self,
        normalized_points: numpy.ndarray,
        *,
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        **kwargs
    ) -> TransformResult:
        r"""
        Alias for the ``inverse_transform`` method, which applies the inverse extrinsic transformation to the points.

        .. seealso::

            - :meth:`pycvcam.core.Transform.inverse_transform` for applying the inverse transformation to points.

        .. code-block:: python

            extrinsic = ... # An instance of a subclass of Extrinsic

            import numpy

            normalized_points = numpy.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]) # shape (Npoints, 2)
            result = extrinsic.unproject(normalized_points)
            world_points = result.world_points  # shape (Npoints, 3)

            # SAME AS:
            result = extrinsic.inverse_transform(normalized_points)
            world_points = result.transformed_points  # shape (Npoints, 3)

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in the camera coordinate system to be transformed. Shape (..., 2).

        transpose : bool, optional
            If True, the input points are assumed to have shape (2, ...) instead of (..., 2) and the output points will have shape (3, ...). Default is False.

        dx : bool, optional
            If True, the jacobian with respect to the normalized points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the extrinsic parameters is computed. Default is False

        Returns
        -------
        TransformResult
            The result of the transformation, which includes the ``world_points`` and the Jacobian matrices if available.
        """
        return self.inverse_transform(normalized_points, transpose=transpose, dx=dx, dp=dp, **kwargs)

    # =============================================
    # Implementing the rays computation
    # =============================================
    def compute_rays(
        self, 
        normalized_points: numpy.ndarray,
        *,
        transpose: bool = False,
        _skip: bool = False,
        **kwargs
        ) -> Rays:
        r"""
        Compute the rays emitted from the camera to the scene.

        The rays are the concatenation of the ``normalized_points`` converted in the world coordinates and the direction of the rays.

        .. code-block:: python

            result = compute_rays(normalized_points)
            rays = result.rays  # (..., 6)

            # The last dimension is the ray structure: (origin_x, origin_y, origin_z, direction_x, direction_y, direction_z)
            # Where the coordinates of the origin and the direction are in the world coordinate system.

            result.origins  # (..., 3)  # The origins of the rays in the world coordinate system
            result.directions  # (..., 3)  # The directions of the rays in the world coordinate system

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in the camera coordinate system. Shape (..., 2).

        transpose : bool, optional
            If True, the input and output arrays are transposed to shape (2, ...) and (6, ...), respectively. Default is False.

        _skip : bool, optional
            If True, skip the checks and transformations. Default is False.

        kwargs : dict
            Additional arguments to be passed to the transformation method.

        Returns
        -------
        Rays
            The rays in the world coordinate system. The shape is (..., 6), where the last dimension represents the origin and direction of the rays.

        """
        if not _skip:
            # Check the boolean flags
            if not isinstance(transpose, bool):
                raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
            
            # Check if the transformation is set
            if not self.is_set():
                raise ValueError("Transformation parameters are not set. Please set the parameters before transforming points.")

            # Convert input points to float
            points = numpy.asarray(normalized_points, dtype=Package.get_float_dtype())

            # Check the shape of the input points
            if points.ndim < 2:
                raise ValueError(f"Input points must have at least 2 dimensions, got {points.ndim} dimensions.")
            
            # Transpose the input points if requested
            if transpose:
                points = numpy.moveaxis(points, 0, -1) # (output_dim, ...) -> (..., output_dim)
            
            # Save the shape of the input points
            shape = points.shape # (..., output_dim)

            # Check the last dimension of the input points
            if shape[-1] != self.output_dim:
                raise ValueError(f"Input points must have {self.output_dim} dimensions, got {shape[-1]} dimensions.")
            
            # Flatten the input points to 2D for processing
            points = points.reshape(-1, self.output_dim) # (Npoints, output_dim)

        # Apply the inverse transformation
        rays = self._compute_rays(points, **kwargs) # (Npoints, 6)

        if not _skip:
            # Reshape the transformed points to the original shape
            rays = rays.reshape(*shape[:-1], 6) # (Npoints, 6) -> (..., 6)

            # Transpose the transformed points if requested
            if transpose:
                rays = numpy.moveaxis(rays, -1, 0) # (..., 6) -> (6, ...)

        # Return the result as a InverseTransformResult object
        return Rays(rays, transpose=transpose)
    
    @abstractmethod
    def _compute_rays(self, normalized_points: numpy.ndarray, **kwargs) -> numpy.ndarray:
        r"""
        Computes the rays in the world coordinate system for the given normalized points.

        A ray is the concatenation of the normalized points with a z-coordinate of 1.0 representing the origin of the ray in the world coordinate system and a direction vector of (0, 0, 1) representing the direction of the ray in the world coordinate system.

        The ray structure is as follows:

        - The first 3 elements are the origin of the ray in the world coordinate system (the normalized points with z=1).
        - The last 3 elements are the direction of the ray in the world coordinate system, which is always (0, 0, 1) for this model. The direction vector is normalized.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in the camera coordinate system. Shape (Npoints, 2).

        Returns
        -------
        numpy.ndarray
            The rays in the world coordinate system. Shape (Npoints, 6).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")