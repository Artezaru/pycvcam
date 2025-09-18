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

from __future__ import annotations
from typing import Optional, Tuple, Dict, List
from numbers import Number
import numpy

from ..core import Intrinsic
from ..core.package import Package



class Cv2Intrinsic(Intrinsic):
    r"""

    Subclass of the :class:`pycvcam.core.Intrinsic` class that represents the OpenCV intrinsic model.

    .. note::

        This class represents the intrinsic transformation, which is the last step of the process from the ``world_points`` to the ``image_points``.

    The ``Cv2Intrinsic`` model used an intrinsic matrix to transform the ``distorted_points`` to the ``image_points``.

    The equation used for the intrinsic transformation is:    

    .. math::

        \begin{align*}
        \vec{x}_i &= K \cdot \vec{x}_d \\
        \end{align*}

    where :math:`\vec{x}_d` is the distorted points, :math:`\vec{x}_i` is the image points, and :math:`K` is the intrinsic matrix defined as:

    .. math::

        K = \begin{bmatrix}
        f_x & 0 & c_x \\
        0 & f_y & c_y \\
        0 & 0 & 1
        \end{bmatrix}

    where :math:`f_x` and :math:`f_y` are the focal lengths in pixels in x and y direction, :math:`c_x` and :math:`c_y` are the principal point coordinates in pixels.

    .. note::

        If no distortion is applied, the ``distorted_points`` are equal to the ``normalized_points``.

    .. warning::

        No ``skew`` parameter is included in the intrinsic matrix in this implementation. If you need to include a skew parameter, you can use :class:`pycvcam.SkewIntrinsic`.

    This transformation is caracterized by 4 parameters and 0 constants:

    - 2 parameters as focal length :math:`\vec{f} = (f_x, f_y)`.
    - 2 parameters as principal point :math:`\vec{c} = (c_x, c_y)`.

    Two short-hand notations are provided to access the jacobian with respect to the focal length and principal point in the results class:

    - ``jacobian_df``: The Jacobian of the image points with respect to the focal length parameters. It has shape (..., 2, 2), where the last dimension represents (df_x, df_y).
    - ``jacobian_dc``: The Jacobian of the image points with respect to the principal point parameters. It has shape (..., 2, 2), where the last dimension represents (dc_x, dc_y).

    .. note::

        The ``Cv2Intrinsic`` class can be instantiated with 2 different ways:

        - Setting directly the parameters as a numpy array of shape (4,) (__init__ method) containing the focal length and principal point concatenated.
        - Using the classmethod ``from_matrix`` to set the intrinsic matrix.

    Parameters
    ----------
    parameters : Optional[numpy.ndarray]
        The parameters of the intrinsic transformation. It should be a numpy array of shape (4,) containing the focal length and principal point concatenated.

    Examples
    --------
    Create an intrinsic object with a given intrinsic matrix:

    .. code-block:: python

        import numpy
        from pycvcam import Cv2Intrinsic

        intrinsic_matrix = numpy.array([[1000, 0, 320],
                                     [0, 1000, 240],
                                     [0, 0, 1]])
        intrinsic = Cv2Intrinsic.from_matrix(intrinsic_matrix)

    Then you can use the intrinsic object to transform ``distorted_points`` to ``image_points``:

    .. code-block:: python

        distorted_points = numpy.array([[100, 200],
                                     [150, 250],
                                     [200, 300]]) # Shape (Npoints, 2)
        result = intrinsic.transform(distorted_points)
        image_points = result.image_points # Shape (Npoints, 2)
        print(image_points)

    You can also access to the jacobian of the intrinsic transformation:

    .. code-block:: python

        result = intrinsic.transform(distorted_points, dx=True, dp=True)
        image_points_dx = result.jacobian_dx  # Jacobian of the image points with respect to the distorted points
        image_points_dp = result.jacobian_dp  # Jacobian of the image points with respect to the intrinsic parameters
        print(image_points_dx)

    The inverse transformation can be computed using the `inverse_transform` method:

    .. code-block:: python

        inverse_result = intrinsic.inverse_transform(image_points, dx=True, dp=True)
        distorted_points = inverse_result.distorted_points  # Shape (Npoints, 2)
        print(distorted_points)
    
    .. seealso::

        For more information about the transformation process, see:

        - :meth:`pycvcam.Cv2Intrinsic._transform` to transform the ``distorted_points`` to ``image_points``.
        - :meth:`pycvcam.Cv2Intrinsic._inverse_transform` to transform the ``image_points`` back to ``distorted_points``.
    
    """
    __slots__ = ["_fx", "_fy", "_cx", "_cy"]

    def __init__(self, parameters: Optional[numpy.ndarray] = None) -> None:
        # Initialize the Transform base class
        super().__init__(parameters=parameters, constants=None)

    # =============================================
    # Overwrite some properties from the base class
    # =============================================
    def _get_jacobian_short_hand(self) -> Dict[str, Tuple[int, int, Optional[str]]]:
        r"""
        Short-hand notation for the Jacobian matrices with respect to the intrinsic parameters.

        - ``df``: The Jacobian of the normalized points with respect to the focal length parameters. It has shape (..., 2, 2), where the last dimension represents (df_x, df_y).
        - ``dc``: The Jacobian of the normalized points with respect to the principal point parameters. It has shape (..., 2, 2), where the last dimension represents (dc_x, dc_y).

        Returns
        -------
        Dict[str, Tuple[int, int, Optional[str]]]
            A dictionary where keys are names of the custom Jacobian views and values are tuples containing:

            - start index (int): The starting index of the parameters to include in the custom Jacobian view.
            - end index (int): The ending index of the parameters to include in the custom Jacobian view.
            - doc (Optional[str]): A documentation string for the custom Jacobian view.
        """
        return {
            "df": (0, 2, "Jacobian of the image points with respect to the focal length parameters (fx, fy)"),
            "dc": (2, 4, "Jacobian of the image points with respect to the principal point parameters (cx, cy)")
        }
    
    @property
    def Nparams(self) -> int:
        r"""
        Get the number of parameters of the intrinsic transformation.

        Returns
        -------
        int
            The number of parameters of the intrinsic transformation. It is always 4 for the Cv2Intrinsic class.
        """
        return 4

    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the parameters of the intrinsic transformation.

        The parameters are a numpy array of shape (4,) representing the focal length and principal point.

        .. seealso::

            - :meth:`pycvcam.Cv2Extrinsic.focal_length_x` or ``fx`` to get the focal length in pixels in x direction.
            - :meth:`pycvcam.Cv2Extrinsic.focal_length_y` or ``fy`` to get the focal length in pixels in y direction.
            - :meth:`pycvcam.Cv2Extrinsic.principal_point_x` or ``cx`` to get the principal point in pixels in x direction.
            - :meth:`pycvcam.Cv2Extrinsic.principal_point_y` or ``cy`` to get the principal point in pixels in y direction.

        Returns
        -------
        Optional[numpy.ndarray]
            The parameters of the intrinsic transformation. Shape (4,) or None if not set.
        """
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            return None
        return numpy.array([self._fx, self._fy, self._cx, self._cy], dtype=Package.get_float_dtype())

    @parameters.setter
    def parameters(self, value: Optional[numpy.ndarray]) -> None:
        if value is None:
            self._fx = None
            self._fy = None
            self._cx = None
            self._cy = None
            return
        value = numpy.asarray(value, dtype=Package.get_float_dtype()).flatten()
        if value.shape != (4,):
            raise ValueError("Parameters must be a 1D array of shape (4,).")
        if not numpy.isfinite(value).all():
            raise ValueError("Parameters must be a finite 1D array of shape (4,).")
        self._fx = value[0]
        self._fy = value[1]
        self._cx = value[2]
        self._cy = value[3]

    @property
    def constants(self) -> Optional[numpy.ndarray]:
        r"""
        Always returns None for the Cv2Intrinsic class, as it does not have any constants.
        """
        return None
    
    @constants.setter
    def constants(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            raise ValueError("Cv2Intrinsic model has no constants, must be set to None.")
        self._constants = None

    @property
    def parameter_names(self) -> List[str]:
        r"""
        Get the names of the parameters of the intrinsic transformation : ["fx", "fy", "cx", "cy"]

        Returns
        -------
        List[str]
            The names of the parameters of the intrinsic transformation.
        """
        return ["fx", "fy", "cx", "cy"]
    
    @property
    def constant_names(self) -> List[str]:
        r"""
        Always returns an empty list for the Cv2Intrinsic class, as it does not have any constants.
        """
        return []

    def is_set(self) -> bool:
        r"""
        Check if the intrinsic parameters are set.

        Returns
        -------
        bool
            True if all intrinsic parameters are set, False otherwise.
        """
        return self._fx is not None and self._fy is not None and self._cx is not None and self._cy is not None

    # =============================================
    # Focal length
    # =============================================
    @property
    def focal_length_x(self) -> Optional[float]:
        r"""
        Get or set the focal length ``fx`` of the intrinsic transformation.

        The focal length is a float representing the focal length of the camera in pixels in x direction.

        This parameter is the component K[0, 0] of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``focal_length_x`` is ``fx``.

        .. seealso::

            - :meth:`pycvcam.Cv2Intrinsic.focal_length_y` or ``fy`` to set the focal length in pixels in y direction.

        Returns
        -------
        Optional[float]
            The focal length of the camera in pixels in x direction. (or None if not set)
        """
        return self._fx

    @focal_length_x.setter
    def focal_length_x(self, fx: Optional[Number]) -> None:
        if fx is None or numpy.isnan(fx):
            self._fx = None
            return
        if not isinstance(fx, Number):
            raise ValueError("Focal length in pixels in x direction must be a number.")
        if not numpy.isfinite(fx):
            raise ValueError("Focal length in pixels in x direction must be a finite number.")
        if fx <= 0:
            raise ValueError("Focal length in pixels in x direction must be greater than 0.")
        self._fx = float(fx)

    @property
    def fx(self) -> float:
        return self.focal_length_x
    
    @fx.setter
    def fx(self, fx: Optional[Number]) -> None:
        self.focal_length_x = fx


    @property
    def focal_length_y(self) -> Optional[float]:
        r"""
        Get or set the focal length ``fy`` of the intrinsic transformation.

        The focal length is a float representing the focal length of the camera in pixels in y direction.

        This parameter is the component K[1, 1] of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``focal_length_y`` is ``fy``.

        .. seealso::

            - :meth:`pycvcam.Cv2Intrinsic.focal_length_x` or ``fx`` to set the focal length in pixels in x direction.

        Returns
        -------
        Optional[float]
            The focal length of the camera in pixels in y direction. (or None if not set)
        """
        return self._fy
    
    @focal_length_y.setter
    def focal_length_y(self, fy: Optional[Number]) -> None:
        if fy is None or numpy.isnan(fy):
            self._fy = None
            return
        if not isinstance(fy, Number):
            raise ValueError("Focal length in pixels in y direction must be a number.")
        if not numpy.isfinite(fy):
            raise ValueError("Focal length in pixels in y direction must be a finite number.")
        if fy <= 0:
            raise ValueError("Focal length in pixels in y direction must be greater than 0.")
        self._fy = float(fy)
    
    @property
    def fy(self) -> float:
        return self.focal_length_y

    @fy.setter
    def fy(self, fy: Optional[Number]) -> None:
        self.focal_length_y = fy

    # =============================================
    # Principal point
    # =============================================
    @property
    def principal_point_x(self) -> Optional[float]:
        r"""
        Get or set the principal point ``cx`` of the intrinsic transformation.

        The principal point is a float representing the principal point of the camera in pixels in x direction.

        This parameter is the component K[0, 2] of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``principal_point_x`` is ``cx``.

        .. seealso::

            - :meth:`pycvcam.Cv2Intrinsic.principal_point_y` or ``cy`` to set the principal point in pixels in y direction.

        Returns
        -------
        Optional[float]
            The principal point of the camera in pixels in x direction. (or None if not set)
        """
        return self._cx
    
    @principal_point_x.setter
    def principal_point_x(self, cx: Optional[Number]) -> None:
        if cx is None or numpy.isnan(cx):
            self._cx = None
            return
        if not isinstance(cx, Number):
            raise ValueError("Principal point in pixels in x direction must be a number.")
        if not numpy.isfinite(cx):
            raise ValueError("Principal point in pixels in x direction must be a finite number.")
        self._cx = float(cx)

    @property
    def cx(self) -> float:
        return self.principal_point_x
    
    @cx.setter
    def cx(self, cx: Optional[Number]) -> None:
        self.principal_point_x = cx

    @property
    def principal_point_y(self) -> Optional[float]:
        r"""
        Get or set the principal point ``cy`` of the intrinsic transformation.

        The principal point is a float representing the principal point of the camera in pixels in y direction.

        This parameter is the component K[1, 2] of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``principal_point_y`` is ``cy``.

        .. seealso::

            - :meth:`pycvcam.Cv2Intrinsic.principal_point_x` or ``cx`` to set the principal point in pixels in x direction.

        Returns
        -------
        Optional[float]
            The principal point of the camera in pixels in y direction. (or None if not set)
        """
        return self._cy
    
    @principal_point_y.setter
    def principal_point_y(self, cy: Optional[Number]) -> None:
        if cy is None or numpy.isnan(cy):
            self._cy = None
            return
        if not isinstance(cy, Number):
            raise ValueError("Principal point in pixels in y direction must be a number.")
        if not numpy.isfinite(cy):
            raise ValueError("Principal point in pixels in y direction must be a finite number.")
        self._cy = float(cy)
    
    @property
    def cy(self) -> float:
        return self.principal_point_y
    
    @cy.setter
    def cy(self, cy: Optional[Number]) -> None:
        self.principal_point_y = cy

    # =============================================
    # Intrinsic matrix
    # =============================================
    @property
    def intrinsic_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the intrinsic matrix of the intrinsic transformation.

        The intrinsic matrix is a 3x3 matrix representing the intrinsic parameters of the camera.

        .. math::

            K = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \end{bmatrix}

        .. note::

            An alias for ``intrinsic_matrix`` is ``K``.

        .. seealso::

            - :meth:`pycvcam.Cv2Intrinsic.intrinsic_vector` or ``k`` to get the intrinsic vector of the camera.

        Returns
        -------
        Optional[numpy.ndarray]
            The intrinsic matrix of the camera. (or None if one of the parameters is not set)
        """
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            return None
        return numpy.array([
            [self._fx, 0, self._cx],
            [0, self._fy, self._cy],
            [0, 0, 1]
        ], dtype=Package.get_float_dtype())
    
    @intrinsic_matrix.setter
    def intrinsic_matrix(self, intrinsic_matrix: Optional[numpy.ndarray]) -> None:
        if intrinsic_matrix is None:
            self._fx = None
            self._fy = None
            self._cx = None
            self._cy = None
            return
        intrinsic_matrix = numpy.asarray(intrinsic_matrix, dtype=Package.get_float_dtype())
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be a 3x3 matrix.")
        # Check if a skew value is given
        if abs(intrinsic_matrix[0, 1]) > 1e-6:
            raise ValueError("Skew value is not supported by Cv2Intrinsic. Use SkewIntrinsic instead.")
        if abs(intrinsic_matrix[1, 0]) > 1e-6 or abs(intrinsic_matrix[2, 0]) > 1e-6 or abs(intrinsic_matrix[2, 1]) > 1e-6:
            raise ValueError("Some coefficients of the intrinsic matrix are unexpected.")
        # Set the intrinsic parameters
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

    @property
    def K(self) -> Optional[numpy.ndarray]:
        return self.intrinsic_matrix
    
    @K.setter
    def K(self, intrinsic_matrix: Optional[numpy.ndarray]) -> None:
        self.intrinsic_matrix = intrinsic_matrix

    # =============================================
    # Intrinsic vector
    # =============================================
    @property
    def intrinsic_vector(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the intrinsic vector of the intrinsic transformation.

        The intrinsic vector is a 4x1 vector representing the intrinsic parameters of the camera.

        .. math::

            \begin{bmatrix}
            f_x \\
            f_y \\
            c_x \\
            c_y
            \end{bmatrix}

        .. note::

            An alias for ``intrinsic_vector`` is ``k``.

        .. seealso::

            - :meth:`pycvcam.Cv2Intrinsic.intrinsic_matrix` or ``K`` to set the intrinsic matrix of the camera.

        Returns
        -------
        Optional[numpy.ndarray]
            The intrinsic vector of the camera. (or None if one of the parameters is not set)
        """
        return self.parameters
    
    @intrinsic_vector.setter
    def intrinsic_vector(self, intrinsic_vector: Optional[numpy.ndarray]) -> None:
        self.parameters = intrinsic_vector

    @property
    def k(self) -> Optional[numpy.ndarray]:
        return self.intrinsic_vector

    @k.setter
    def k(self, intrinsic_vector: Optional[numpy.ndarray]) -> None:
        self.intrinsic_vector = intrinsic_vector

    # =============================================
    # Instantiation methods
    # =============================================
    @classmethod
    def from_matrix(cls, intrinsic_matrix: numpy.ndarray) -> Cv2Intrinsic:
        r"""
        Class method to create a Cv2Intrinsic object from an intrinsic matrix.

        Parameters
        ----------
        intrinsic_matrix : numpy.ndarray
            The intrinsic matrix of the camera. It should be a numpy array of shape (3, 3).

        Returns
        -------
        Cv2Intrinsic
            A new instance of the Cv2Intrinsic class with the specified intrinsic matrix.

        Examples
        --------
        Create an intrinsic object with a given intrinsic matrix:

        .. code-block:: python

            import numpy as np
            from pycvcam import Cv2Intrinsic

            intrinsic_matrix = numpy.array([[1000, 0, 320],
                                         [0, 1000, 240],
                                         [0, 0, 1]])
            intrinsic = Cv2Intrinsic.from_matrix(intrinsic_matrix)
        
        """
        intrinsic = cls()
        intrinsic.intrinsic_matrix = intrinsic_matrix
        return intrinsic

    # =============================================
    # Transform methods
    # =============================================
    def _transform(self, distorted_points: numpy.ndarray, *, dx: bool = False, dp: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the ``distorted_points`` to the ``image_points``.

        Lets consider ``distorted_points`` in the camera normalized coordinate system :math:`\vec{x}_d = (x_d, y_d)`, the corresponding ``image_points`` in the image coordinate system are given by :

        The equation used for the intrinsic transformation is:    

        .. math::

            x_i = f_x \cdot x_d + c_x
    
        .. math::

            y_i = f_y \cdot y_d + c_y

        The jacobians with respect to the intrinsic parameters is an array with shape (Npoints, 2, 4), where the last dimension represents the parameters (fx, fy, cx, cy).
        The jacobian with respect to the distorted points is an array with shape (Npoints, 2, 2).

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
            The jacobian of the image points with respect to the intrinsic parameters. Shape (Npoints, 2, 4) if dp is True, otherwise None.
        """
        # Extract the useful coordinates
        x_D = distorted_points[:, 0] # shape (Npoints,)
        y_D = distorted_points[:, 1] # shape (Npoints,)

        # Compute the image points
        x_I = self._fx * x_D + self._cx # shape (Npoints,)
        y_I = self._fy * y_D + self._cy # shape (Npoints,)

        image_points_flat = numpy.empty(distorted_points.shape) # shape (Npoints, 2)
        image_points_flat[:, 0] = x_I # shape (Npoints,)
        image_points_flat[:, 1] = y_I # shape (Npoints,)
 
        # Compute the jacobian with respect to the distorted points
        if dx:
            jacobian_flat_dx = numpy.empty((*distorted_points.shape, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 2)
            jacobian_flat_dx[:, 0, 0] = self._fx # shape (Npoints,)
            jacobian_flat_dx[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dx[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dx[:, 1, 1] = self._fy # shape (Npoints,)
        else:
            jacobian_flat_dx = None

        # Compute the jacobian with respect to the intrinsic parameters
        if dp:
            jacobian_flat_dp = numpy.empty((*distorted_points.shape, 4), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 4)
            jacobian_flat_dp[:, 0, 0] = x_D # shape (Npoints,)
            jacobian_flat_dp[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 0, 2] = 1.0 # shape (Npoints,)
            jacobian_flat_dp[:, 0, 3] = 0.0 # shape (Npoints,)

            jacobian_flat_dp[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 1] = y_D # shape (Npoints,)
            jacobian_flat_dp[:, 1, 2] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 3] = 1.0 # shape (Npoints,)
        else:
            jacobian_flat_dp = None

        return image_points_flat, jacobian_flat_dx, jacobian_flat_dp
    

    def _inverse_transform(self, image_points: numpy.ndarray, *, dx: bool = False, dp: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from the ``image_points`` to the ``distorted_points``.

        Lets consider ``image_points`` in the image coordinate system :math:`\vec{x}_i = (x_i, y_i)`, the corresponding ``distorted_points`` in the camera normalized coordinate system are given by :

        .. math::

            x_d = \frac{x_i - c_x}{f_x}
        
        .. math::
        
            y_d = \frac{y_i - c_y}{f_y}

        The jacobians with respect to the intrinsic parameters is an array with shape (Npoints, 2, 4), where the last dimension represents the parameters (fx, fy, cx, cy).
        The jacobian with respect to the image points is an array with shape (Npoints, 2, 2).

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
            The jacobian of the distorted points with respect to the intrinsic parameters. Shape (Npoints, 2, 4) if dp is True, otherwise None.
        """
        # Extract the useful coordinates
        x_I = image_points[:, 0] # shape (Npoints,)
        y_I = image_points[:, 1] # shape (Npoints,)

        # Compute the distorted points
        x_D = (x_I - self._cx) / self._fx # shape (Npoints,)
        y_D = (y_I - self._cy) / self._fy # shape (Npoints,)

        distorted_points_flat = numpy.empty(image_points.shape) # shape (Npoints, 2)
        distorted_points_flat[:, 0] = x_D # shape (Npoints,)
        distorted_points_flat[:, 1] = y_D # shape (Npoints,)

        # Compute the jacobian with respect to the image points
        if dx:
            jacobian_flat_dx = numpy.empty((*image_points.shape, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 2)
            jacobian_flat_dx[:, 0, 0] = 1.0 / self._fx # shape (Npoints,)
            jacobian_flat_dx[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dx[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dx[:, 1, 1] = 1.0 / self._fy # shape (Npoints,)
        else:
            jacobian_flat_dx = None

        # Compute the jacobian with respect to the intrinsic parameters
        if dp:
            jacobian_flat_dp = numpy.empty((*image_points.shape, 4), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 4)
            jacobian_flat_dp[:, 0, 0] = - x_D / self._fx # shape (Npoints,) because x_D = (x_I - c_x) / f_x
            jacobian_flat_dp[:, 0, 1] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 0, 2] = - 1.0 / self._fx # shape (Npoints,)
            jacobian_flat_dp[:, 0, 3] = 0.0 # shape (Npoints,)

            jacobian_flat_dp[:, 1, 0] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 1] = - y_D / self._fy # shape (Npoints,) because y_D = (y_I - c_y) / f_y
            jacobian_flat_dp[:, 1, 2] = 0.0 # shape (Npoints,)
            jacobian_flat_dp[:, 1, 3] = -1.0 / self._fy # shape (Npoints,)
        else:
            jacobian_flat_dp = None

        return distorted_points_flat, jacobian_flat_dx, jacobian_flat_dp



        