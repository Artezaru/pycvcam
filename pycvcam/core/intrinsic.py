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
import numpy

from .transform import Transform, TransformResult

class Intrinsic(Transform):
    r"""
    .. note::

        This class represents the intrinsic transformation, which is the last step of the process from the ``world_points`` to the ``image_points``.
        This is the abstract class that defines the interface for intrinsic transformations.

    This base class, set the ``input_dim`` and ``output_dim`` to 2 and add the aliases :

    - ``image_points`` and ``x_i`` for the result of the transformation.
    - ``distorted_points`` and ``x_d`` for the result of the inverse transformation.

    .. note::

        If the camera don't have distortion, the ``distorted_points`` are same as the ``normalized_points``.

    To process the transformation, the methods ``scale`` and ``unscale`` are provided, which are aliases for the ``transform`` and ``inverse_transform`` methods respectively.

    """
    _input_dim : ClassVar[int] = 2
    _output_dim : ClassVar[int] = 2

    # =============================================
    # Addind aliases for the transformation
    # =============================================
    def _get_transform_aliases(self) -> List[str]:
        r"""
        Property to return a list of aliases for the transformed points.

        - ``image_points`` and ``x_i`` are added.
        
        Returns
        -------
        List[str]
            A list of aliases for the transformed points.
        """
        return ["image_points", "x_i"]
    
    def _get_inverse_transform_aliases(self) -> List[str]:
        r"""
        Property to return a list of aliases for the inverse transformed points.

        - ``distorted_points`` and ``x_d`` are added.

        Returns
        -------
        List[str]
            A list of aliases for the inverse transformed points.
        """
        return ["distorted_points", "x_d"]

    def scale(
        self, 
        distorted_points: numpy.ndarray, 
        *, 
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        **kwargs
        ) -> TransformResult:
        r"""
        Alias for the ``transform`` method, which applies the intrinsic transformation to the points.

        .. seealso::

            - :meth:`pycvcam.core.Transform.transform` for applying the transformation to points.

        .. code-block:: python

            intrinsic = ... # An instance of a subclass of Intrinsic

            import numpy

            distorted_points = numpy.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]) # shape (Npoints, 2)
            result = intrinsic.scale(distorted_points)
            image_points = result.image_points  # shape (Npoints, 2)

            # SAME AS:
            result = intrinsic.transform(distorted_points)
            image_points = result.transformed_points  # shape (Npoints, 2)

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points to be transformed. Shape (..., 2).

        transpose : bool, optional
            If True, the input points are assumed to have shape (2, ...) instead of (..., 2) and the output points will have shape (2, ...). Default is False.

        dx : bool, optional
            If True, the jacobian with respect to the distorted points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the intrinsic parameters is computed. Default is False

        Returns
        -------
        TransformResult
            The result of the transformation, which includes the ``image_points`` and the Jacobian matrices if available.
        """
        return self.transform(distorted_points, transpose=transpose, dx=dx, dp=dp, **kwargs)

    def unscale(
        self,
        image_points: numpy.ndarray,
        *,
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        **kwargs
        ) -> TransformResult:
        r"""
        Alias for the ``inverse_transform`` method, which applies the inverse intrinsic transformation to the points.

        .. seealso::

            - :meth:`pycvcam.core.Transform.inverse_transform` for applying the inverse transformation to points.

        .. code-block:: python

            intrinsic = ... # An instance of a subclass of Intrinsic

            import numpy

            image_points = numpy.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]) # shape (Npoints, 2)
            result = intrinsic.unscale(image_points)
            distorted_points = result.distorted_points  # shape (Npoints, 2)

            # SAME AS:
            result = intrinsic.inverse_transform(image_points)
            distorted_points = result.transformed_points  # shape (Npoints, 2)

        Parameters
        ----------
        image_points : numpy.ndarray
            The image points to be transformed. Shape (..., 2).

        transpose : bool, optional
            If True, the input points are assumed to have shape (2, ...) instead of (..., 2) and the output points will have shape (2, ...). Default is False.

        dx : bool, optional
            If True, the jacobian with respect to the image points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the intrinsic parameters is computed. Default is False

        Returns
        -------
        TransformResult
            The result of the inverse transformation, which includes the ``distorted_points`` and the Jacobian matrices if available.

        """
        return self.inverse_transform(image_points, transpose=transpose, dx=dx, dp=dp, **kwargs)
