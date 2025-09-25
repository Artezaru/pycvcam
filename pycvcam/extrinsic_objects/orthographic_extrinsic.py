
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
import numpy
from py3dframe import Frame
import cv2

from ..core import Extrinsic
from ..core.package import Package


class OrthographicExtrinsic(Extrinsic):
    r"""

    Subclass of the :class:`pycvcam.core.Extrinsic` class that represents the orthographic extrinsic transformation.

    .. note::

        This class represents the extrinsic transformation, which is the first step of the process from the ``world_points`` to the ``image_points``.

    The ``OrthographicExtrinsic`` model in the composition of a changement in reference frame and an orthographic projection.

    Lets consider ``world_points`` in the global coordinate system :math:`\vec{X}_w = (X_w, Y_w, Z_w)`, the corresponding ``normalized_points`` in the camera normalized coordinate system are given :math:`\vec{x}_n` can be optained by :

    .. math::

        \vec{X_c} = R \cdot \vec{X_w} + T

    .. math::

        \vec{x}_n = (X_c, Y_c)

    where :math:`R` is the rotation matrix, :math:`T` is the translation vector, and :math:`Z_c` the depth of the point in the camera coordinate system is ignored.

    .. note::

        To compute the translation vector and the rotation vector, you can use ``cv2.Rodrigues()`` or ``py3dframe.Frame`` with convention 4.

    .. seealso::

        Package ``py3dframe`` (https://github.com/Artezaru/py3dframe) for the implementation of the 3D frame and the rotation vector.

    This transformation is caracterized by 6 parameters and 0 constants:

    - 3 parameters as rotation vector :math:`\vec{rvec} = (r_x, r_y, r_z)`.
    - 3 parameters as translation vector :math:`\vec{tvec} = (t_x, t_y, t_z)`.

    Two short-hand notations are provided to access the jacobian with respect to the rotation vector and translation vector in the results class:

    - ``jacobian_dr``: The Jacobian of the normalized points with respect to the rotation vector. It has shape (..., 2, 3).
    - ``jacobian_dt``: The Jacobian of the normalized points with respect to the translation vector. It has shape (..., 2, 3).

    .. note::

        The ``OrthographicExtrinsic`` class can be instantiated with 3 different ways:

        - Setting directly the parameters as a numpy array of shape (6,) (__init__ method) containing the rotation vector and translation vector concatenated.
        - Using the classmethod ``from_rt`` to set the rotation vector and translation vector.
        - Using the classmethod ``from_frame`` to set the 3D frame of the camera in the world coordinate system.

    Parameters
    ----------
    parameters : Optional[numpy.ndarray]
        The parameters of the extrinsic transformation. It should be a numpy array of shape (6,) containing the rotation vector and translation vector concatenated.

    Examples
    --------
    Create an extrinsic object with a rotation vector and a translation vector:

    .. code-block:: python

        import numpy
        from pycvcam import OrthographicExtrinsic

        rvec = numpy.array([0.1, 0.2, 0.3])
        tvec = numpy.array([0.5, 0.5, 0.5])

        extrinsic = OrthographicExtrinsic.from_rt(rvec, tvec)

    Then you can use the extrinsic object to transform ``world_points`` to ``normalized_points``:

    .. code-block:: python

        world_points = numpy.array([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9],
                                   [10, 11, 12]]) # shape (Npoints, 3)

        result = extrinsic.transform(world_points)
        normalized_points = result.normalized_points # shape (Npoints, 2)
        print(normalized_points)

    You can also access to the jacobian of the extrinsic transformation:

    .. code-block:: python

        result = extrinsic.transform(world_points, dx=True, dp=True)
        normalized_points_dx = result.jacobian_dx  # Shape (Npoints, 2, 3)
        normalized_points_dp = result.jacobian_dp  # Shape (Npoints, 2, 6)
        print(normalized_points_dx) 
        print(normalized_points_dp)

    The inverse transformation can be computed using the `inverse_transform` method:
    By default, the depth is assumed to be 1.0 for all points, but you can provide a specific depth for each point with shape (...,).

    .. code-block:: python

        depth = numpy.array([1.0, 2.0, 3.0, 4.0])  # Example depth values for each point

        inverse_result = extrinsic.inverse_transform(normalized_points, dx=True, dp=True, depth=depth)
        world_points = inverse_result.world_points  # Shape (Npoints, 3)
        print(world_points)

    .. note::

        The jacobian with respect to the depth is not computed.
    
    .. seealso::

        For more information about the transformation process, see:

        - :meth:`pycvcam.OrthographicExtrinsic._transform` to transform the ``world_points`` to ``normalized_points``.
        - :meth:`pycvcam.OrthographicExtrinsic._inverse_transform` to transform the ``normalized_points`` back to ``world_points``.

    """
    __slots__ = ["_rvec", "_tvec"]

    def __init__(self, parameters: Optional[numpy.ndarray] = None) -> None:
        # Initialize the Transform base class
        super().__init__(parameters=parameters, constants=None)

    # =============================================
    # Overwrite some properties from the base class
    # =============================================
    def _get_jacobian_short_hand(self) -> Dict[str, Tuple[int, int, Optional[str]]]:
        r"""
        Short-hand notation for the Jacobian matrices with respect to the extrinsic parameters.

        - ``dr``: The Jacobian of the normalized points with respect to the rotation vector. It has shape (..., 2, 3).
        - ``dt``: The Jacobian of the normalized points with respect to the translation vector. It has shape (..., 2, 3).

        Returns
        -------
        Dict[str, Tuple[int, int, Optional[str]]]
            A dictionary where keys are names of the custom Jacobian views and values are tuples containing:

            - start index (int): The starting index of the parameters to include in the custom Jacobian view.
            - end index (int): The ending index of the parameters to include in the custom Jacobian view.
            - doc (Optional[str]): A documentation string for the custom Jacobian view.
        """
        return {
            "dr": (0, 3, "Jacobian with respect to the rotation vector (rvec)"),
            "dt": (3, 6, "Jacobian with respect to the translation vector (tvec)"),
        }
    
    @property
    def Nparams(self) -> int:
        r"""
        Get the number of parameters of the extrinsic transformation.

        Returns
        -------
        int
            The number of parameters of the extrinsic transformation. It is always 6 for the OrthographicExtrinsic class.
        """
        return 6

    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the parameters of the extrinsic transformation.

        The parameters are a numpy array of shape (6,) representing the rotation vector and translation vector concatenated.

        .. seealso::

            - :meth:`pycvcam.OrthographicExtrinsic.rotation_vector` or ``rvec`` to set the rotation vector of the extrinsic transformation.
            - :meth:`pycvcam.OrthographicExtrinsic.translation_vector` or ``tvec`` to set the translation vector of the extrinsic transformation.

        Returns
        -------
        Optional[numpy.ndarray]
            The parameters of the extrinsic transformation. Shape (6,) or None if not set.
        """
        if self._rvec is None or self._tvec is None:
            return None
        return numpy.concatenate((self._rvec, self._tvec), axis=0)
    
    @parameters.setter
    def parameters(self, value: Optional[numpy.ndarray]) -> None:
        if value is None:
            self._rvec = None
            self._tvec = None
            return
        value = numpy.asarray(value, dtype=Package.get_float_dtype()).flatten()
        if value.shape != (6,):
            raise ValueError("Parameters must be a 1D array of shape (6,).")
        if not numpy.isfinite(value).all():
            raise ValueError("Parameters must be a finite 1D array of shape (6,).")
        self._rvec = value[:3]
        self._tvec = value[3:]

    @property
    def constants(self) -> Optional[numpy.ndarray]:
        r"""
        Always returns None for the OrthographicExtrinsic class, as it does not have any constants.
        """
        return None
    
    @constants.setter
    def constants(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            raise ValueError("OrthographicExtrinsic model has no constants, must be set to None.")
        self._constants = None

    @property
    def parameter_names(self) -> List[str]:
        r"""
        Get the names of the parameters of the extrinsic transformation : ["r_x", "r_y", "r_z", "t_x", "t_y", "t_z"]

        Returns
        -------
        List[str]
            The names of the parameters of the extrinsic transformation.
        """
        return ["r_x", "r_y", "r_z", "t_x", "t_y", "t_z"]

    @property
    def constant_names(self) -> List[str]:
        r"""
        Always returns an empty list for the OrthographicExtrinsic class, as it does not have any constants.
        """
        return []

    def is_set(self) -> bool:
        r"""
        Check if the extrinsic parameters are set.

        Returns
        -------
        bool
            True if both rotation vector and translation vector are set, False otherwise.
        """
        return self._rvec is not None and self._tvec is not None

    # =============================================
    # translation vector
    # =============================================
    @property
    def translation_vector(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the translation vector ``tvec`` of the extrinsic transformation.

        The translation vector is a numpy array of shape (3,) representing the translation of the camera in the world coordinate system.

        .. note::

            An alias for ``translation_vector`` is ``tvec``.

        .. seealso::

            - :meth:`pycvcam.OrthographicExtrinsic.rotation_vector` or ``rvec`` to set the rotation vector of the extrinsic transformation.

        Returns
        -------
        Optional[numpy.ndarray]
            The translation vector of the camera in the world coordinate system. (or None if not set)
        """
        return self._tvec
    
    @translation_vector.setter
    def translation_vector(self, tvec: numpy.ndarray) -> None:
        if tvec is None:
            self._tvec = None
            return
        tvec = numpy.asarray(tvec, dtype=Package.get_float_dtype()).flatten()
        if tvec.shape != (3,):
            raise ValueError("Translation vector must be a 3D vector.")
        if not numpy.isfinite(tvec).all():
            raise ValueError("Translation vector must be a finite 3D vector.")
        self._tvec = tvec

    @property
    def tvec(self) -> Optional[numpy.ndarray]:
        return self.translation_vector

    @tvec.setter
    def tvec(self, tvec: Optional[numpy.ndarray]) -> None:
        self.translation_vector = tvec

    # =============================================
    # rotation vector
    # =============================================
    @property
    def rotation_vector(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the rotation vector ``rvec`` of the extrinsic transformation.

        The rotation vector is a numpy array of shape (3,) representing the rotation of the camera in the world coordinate system.

        .. note::

            An alias for ``rotation_vector`` is ``rvec``.

        .. seealso::

            - :meth:`pycvcam.OrthographicExtrinsic.translation_vector` or ``tvec`` to set the translation vector of the extrinsic transformation.

        Returns
        -------
        Optional[numpy.ndarray]
            The rotation vector of the camera in the world coordinate system. (or None if not set)
        """
        return self._rvec
    
    @rotation_vector.setter
    def rotation_vector(self, rvec: Optional[numpy.ndarray]) -> None:
        if rvec is None:
            self._rvec = None
            return
        rvec = numpy.asarray(rvec, dtype=Package.get_float_dtype()).flatten()
        if rvec.shape != (3,):
            raise ValueError("Rotation vector must be a 3D vector.")
        if not numpy.isfinite(rvec).all():
            raise ValueError("Rotation vector must be a finite 3D vector.")
        self._rvec = rvec

    @property
    def rvec(self) -> Optional[numpy.ndarray]:
        return self.rotation_vector
    
    @rvec.setter
    def rvec(self, rvec: Optional[numpy.ndarray]) -> None:
        self.rotation_vector = rvec

    # =============================================
    # Rotation matrix
    # =============================================
    @property
    def rotation_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the rotation matrix of the extrinsic transformation.

        The rotation matrix is a numpy array of shape (3, 3) representing the rotation of the camera in the world coordinate system.

        .. note::

            The rotation matrix is computed using the Rodrigues formula.
            An alias for ``rotation_matrix`` is ``rmat``.

        Returns
        -------
        Optional[numpy.ndarray]
            The rotation matrix of the camera in the world coordinate system. (or None if not set)
        """
        if self._rvec is None:
            return None
        return cv2.Rodrigues(self._rvec)[0]
    
    @rotation_matrix.setter
    def rotation_matrix(self, rmat: Optional[numpy.ndarray]) -> None:
        if rmat is None:
            self._rvec = None
            return
        rmat = numpy.asarray(rmat, dtype=Package.get_float_dtype())
        if rmat.shape != (3, 3):
            raise ValueError("Rotation matrix must be a 3x3 matrix.")
        if not numpy.isfinite(rmat).all():
            raise ValueError("Rotation matrix must be a finite 3x3 matrix.")
        self._rvec = cv2.Rodrigues(rmat)[0].flatten()

    @property
    def rmat(self) -> Optional[numpy.ndarray]:
        return self.rotation_matrix
    
    @rmat.setter
    def rmat(self, rmat: Optional[numpy.ndarray]) -> None:
        self.rotation_matrix = rmat

    # =============================================
    # Frame (from py3dframe)
    # =============================================
    @property
    def frame(self) -> Optional[Frame]:
        r"""
        Get or set the 3D frame of the extrinsic transformation.

        The frame is a py3dframe.Frame object representing the 3D frame of the camera in the world coordinate system.

        .. seealso::

            https://github.com/Artezaru/py3dframe for more information about the Frame class.

        Returns
        -------
        Optional[Frame]
            The 3D frame of the camera in the world coordinate system. (or None if not set)
        """
        if self._rvec is None or self._tvec is None:
            return None
        return Frame(translation=self._tvec, rotation_vector=self._rvec, convention=4)
    
    @frame.setter
    def frame(self, frame: Optional[Frame]) -> None:
        if frame is None:
            self._rvec = None
            self._tvec = None
            return
        if not isinstance(frame, Frame):
            raise ValueError("Frame must be a py3dframe.Frame object.")
        self._rvec = frame.get_global_rotation_vector(convention=4).flatten()
        self._tvec = frame.get_global_translation(convention=4).flatten()

    
    # =============================================
    # Instantiation methods
    # =============================================
    @classmethod
    def from_rt(cls, rvec: numpy.ndarray, tvec: numpy.ndarray) -> OrthographicExtrinsic:
        r"""
        Class method to create a OrthographicExtrinsic object from a rotation vector and a translation vector.

        Parameters
        ----------
        rvec : numpy.ndarray
            The rotation vector of the camera in the world coordinate system. It should be a numpy array of shape (3,).

        tvec : numpy.ndarray
            The translation vector of the camera in the world coordinate system. It should be a numpy array of shape (3,).

        Returns
        -------
        OrthographicExtrinsic
            A new instance of the OrthographicExtrinsic class with the specified rotation and translation vectors.

        Examples
        --------
        Create an extrinsic object with a rotation vector and a translation vector:

        .. code-block:: python

            import numpy as np

            from pycvcam import OrthographicExtrinsic

            rvec = numpy.array([0.1, 0.2, 0.3])
            tvec = numpy.array([0.5, 0.5, 0.5])

            extrinsic = OrthographicExtrinsic.from_rt(rvec, tvec)
        
        """
        extrinsic = cls()
        extrinsic.rotation_vector = rvec
        extrinsic.translation_vector = tvec
        return extrinsic
    
    @classmethod
    def from_frame(cls, frame: Frame) -> OrthographicExtrinsic:
        r"""
        Class method to create a OrthographicExtrinsic object from a 3D frame.

        Parameters
        ----------
        frame : Frame
            The 3D frame of the camera in the world coordinate system. It should be a py3dframe.Frame object.

        Returns
        -------
        OrthographicExtrinsic
            A new instance of the OrthographicExtrinsic class with the specified 3D frame.

        Examples
        --------
        Create an extrinsic object with a 3D frame:

        .. code-block:: python

            from py3dframe import Frame
            from pycvcam import OrthographicExtrinsic

            frame = Frame(translation=[0.5, 0.5, 0.5], rotation_vector=[0.1, 0.2, 0.3], convention=4)
            extrinsic = OrthographicExtrinsic.from_frame(frame)
        
        """
        extrinsic = cls()
        extrinsic.frame = frame
        return extrinsic
    
    # =============================================
    # Transform methods
    # =============================================
    def _transform(self, world_points: numpy.ndarray, *, dx: bool = False, dp: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the ``world_points`` to the ``normalized_points``.

        Lets consider ``world_points`` in the global coordinate system :math:`\vec{X}_w = (X_w, Y_w, Z_w)`, the corresponding ``normalized_points`` in the camera normalized coordinate system are given :math:`\vec{x}_n` can be optained by :

        .. math::

            \vec{X}_c = R \cdot \vec{X}_w + T

        .. math::

            \vec{x}_n = (X_c, Y_c)

        where :math:`R` is the rotation matrix, :math:`T` is the translation vector, and :math:`Z_c` the depth of the point in the camera coordinate system is ignored.

        The jacobians with respect to the extrinsic parameters is an array with shape (Npoints, 2, 6), where the last dimension contains the jacobian with respect to the rotation vector and translation vector.
        The jacobian with respect to the world points is an array with shape (Npoints, 2, 3).

        .. warning::

            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``world_points`` is (Npoints, 3) before calling this method.

        Parameters
        ----------
        world_points : numpy.ndarray
            Array of world 3dpoints to be transformed with shape (Npoints, 3).

        dx : bool, optional
            If True, the Jacobian of the normalized points with respect to the input 3D world points is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 3).

        dp : bool, optional
            If True, the Jacobian of the normalized points with respect to the pose parameters is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 6).

        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points in camera normalized coordinates. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the normalized points with respect to the world points. Shape (Npoints, 2, 3) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the normalized points with respect to the extrinsic parameters. Shape (Npoints, 2, 6) if dp is True, otherwise None.
        """
        # Get the number of points
        Npoints = world_points.shape[0]

        # Get the rotation matrix and translation vector
        rmat, jacobian = cv2.Rodrigues(self._rvec)
        rmat = numpy.asarray(rmat, dtype=Package.get_float_dtype()) # shape (3, 3)
        jacobian = numpy.asarray(jacobian, dtype=Package.get_float_dtype()) # shape (3, 9) [R11,R12,R13,R21,R22,R23,R31,R32,R33]
        rmat_dr = jacobian.reshape(3, 3, 3).transpose(1, 2, 0) # shape (3, 3, 3) # [i, j, k] = dR[i,j]/drvec[k]

        # ==================
        # Camera points
        # ==================
        # Compute the camera points
        points_camera_flat = world_points @ rmat.T + self._tvec[numpy.newaxis, :] # shape (Npoints, 3)

        # Compute the jacobian with respect to the world points
        if dx:
            points_camera_flat_dx = numpy.broadcast_to(rmat, (Npoints, 3, 3))

        # Compute the jacobian with respect to the extrinsic parameters
        if dp:
            points_camera_flat_dp = numpy.empty((Npoints, 3, 6), dtype=Package.get_float_dtype()) # shape (Npoints, 3, 6)
            for k in range(3):
                points_camera_flat_dp[:, :, k] = world_points @ rmat_dr[:, :, k].T # shape (Npoints, 3)
            points_camera_flat_dp[:, :, 3] = numpy.array([1.0, 0.0, 0.0], dtype=Package.get_float_dtype())[numpy.newaxis, :] # shape (Npoints, 3)
            points_camera_flat_dp[:, :, 4] = numpy.array([0.0, 1.0, 0.0], dtype=Package.get_float_dtype())[numpy.newaxis, :] # shape (Npoints, 3)
            points_camera_flat_dp[:, :, 5] = numpy.array([0.0, 0.0, 1.0], dtype=Package.get_float_dtype())[numpy.newaxis, :] # shape (Npoints, 3)

        # ==================
        # Normalized points
        # ==================
        normalized_points_flat = points_camera_flat[:, :2] # shape (Npoints, 2)

        # Compute the jacobian with respect to the camera points
        if dx:
            jacobian_flat_dx = points_camera_flat_dx[:, :2, :] # shape (Npoints, 2, 3)

        # Compute the jacobian with respect to the extrinsic parameters
        if dp:
            jacobian_flat_dp = points_camera_flat_dp[:, :2, :] # shape (Npoints, 2, 6)
        if not dx:
            jacobian_flat_dx = None
        if not dp:
            jacobian_flat_dp = None

        return normalized_points_flat, jacobian_flat_dx, jacobian_flat_dp
    

    def _inverse_transform(self, normalized_points: numpy.ndarray, *, dx: bool = False, dp: bool = False, depth: Optional[numpy.ndarray] = None) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the ``normalized_points`` to the ``world_points``.

        Lets consider ``normalized_points`` in the camera normalized coordinate system :math:`\vec{x}_n = (x_n, y_n)`, the corresponding ``world_points`` in the global coordinate system are given :math:`\vec{X}_w` can be optained by :

        .. math::

            \vec{X}_c = (x_n, y_n, Z_c)

        .. math::

            \vec{X}_w = R^{-1} \cdot (\vec{X}_c - T)

        where :math:`R` is the rotation matrix, :math:`T` is the translation vector, and :math:`Z_c` is the depth of the point in the camera coordinate system.

        The jacobians with respect to the extrinsic parameters is an array with shape (Npoints, 3, 6), where the last dimension contains the jacobian with respect to the rotation vector and translation vector.
        The jacobian with respect to the normalized points is an array with shape (Npoints, 3, 2).

        .. warning::

            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.inverse_transform` method.
            Please ensure, the shape of the input ``normalized_points`` is (Npoints, 2) before calling this method, and same for the depth parameter if provided.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            Array of normalized points in camera normalized coordinates to be transformed with shape (Npoints, 2).

        dx : bool, optional
            If True, the Jacobian of the normalized points with respect to the input 3D world points is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 3).

        dp : bool, optional
            If True, the Jacobian of the normalized points with respect to the pose parameters is computed. Default is False.
            The output will be a 2D array of shape (Npoints, 2, 6).

        depth : Optional[numpy.ndarray], optional
            The depth of the points in the world coordinate system. If None, the depth is assumed to be 1.0 for all points.
            The shape should be (...,) if provided, and it should match the number of points in the normalized_points array.

        Returns
        -------
        world_points : numpy.ndarray
            The transformed world 3D points. It will be a 2D array of shape (Npoints, 3).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the world points with respect to the normalized points. Shape (Npoints, 3, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the world points with respect to the extrinsic parameters. Shape (Npoints, 3, 6) if dp is True, otherwise None.
        """
        # Get the number of points
        Npoints = normalized_points.shape[0]

        # Get the rotation matrix and translation vector
        rmat, jacobian = cv2.Rodrigues(self._rvec)
        rmat = numpy.asarray(rmat, dtype=Package.get_float_dtype()) # shape (3, 3)
        rmat_inv = rmat.T # Inverse of the rotation matrix (R^{-1} = R^{T})
        jacobian = numpy.asarray(jacobian, dtype=Package.get_float_dtype()) # shape (3, 9) [R11,R12,R13,R21,R22,R23,R31,R32,R33]
        rmat_dr = jacobian.reshape(3, 3, 3).transpose(1, 2, 0) # shape (3, 3, 3) # [i, j, k] = dR[i,j]/drvec[k]
        rmat_inv_dr = rmat_dr.transpose(1, 0, 2) # shape (3, 3, 3) # [i, j, k] = dR^{-1}[i,j]/drvec[k] = dR^{T}[i,j]/drvec[k] = dR[j,i]/drvec[k]

        # ==================
        # Check depth
        # ==================
        if depth is None:
            depth = numpy.ones((Npoints,), dtype=Package.get_float_dtype())
        else:
            depth = numpy.asarray(depth, dtype=Package.get_float_dtype()).flatten()
            if depth.shape != (Npoints,):
                raise ValueError("Depth must be a 1D array with the same number of points as normalized_points.")

        # ==================
        # Camera points
        # ==================
        # Compute the camera points
        X_C = normalized_points[:, 0] # shape (Npoints,)
        Y_C = normalized_points[:, 1] # shape (Npoints,)
        Z_C = depth # shape (Npoints,)

        points_camera_flat = numpy.empty((Npoints, 3), dtype=Package.get_float_dtype()) # shape (Npoints, 3)
        points_camera_flat[:, 0] = X_C
        points_camera_flat[:, 1] = Y_C
        points_camera_flat[:, 2] = Z_C

        # Compute the jacobian with respect to the normalized points
        if dx:
            points_camera_flat_dx = numpy.empty((Npoints, 3, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 3, 2)
            points_camera_flat_dx[:, 0, 0] = 1.0 # shape (Npoints, 2)
            points_camera_flat_dx[:, 0, 1] = 0.0
            points_camera_flat_dx[:, 1, 0] = 0.0
            points_camera_flat_dx[:, 1, 1] = 1.0 # shape (Npoints, 2)
            points_camera_flat_dx[:, 2, 0] = 0.0
            points_camera_flat_dx[:, 2, 1] = 0.0

        # ===================
        # World points
        # ===================
        # Compute the world points
        world_points_flat = (points_camera_flat - self._tvec[numpy.newaxis, :]) @ rmat_inv.T # shape (Npoints, 3)

        # Compute the jacobian with respect to the camera points
        if dx:
            world_points_flat_dx = numpy.empty((Npoints, 3, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 3, 2)
            world_points_flat_dx[:, :, 0] = points_camera_flat_dx[:, :, 0] @ rmat_inv.T # shape (Npoints, 3)
            world_points_flat_dx[:, :, 1] = points_camera_flat_dx[:, :, 1] @ rmat_inv.T # shape (Npoints, 3)

        # Compute the jacobian with respect to the extrinsic parameters
        if dp:
            world_points_flat_dp = numpy.empty((Npoints, 3, 6), dtype=Package.get_float_dtype()) # shape (Npoints, 3, 6)
            for k in range(3):
                world_points_flat_dp[:, :, k] = (points_camera_flat - self._tvec[numpy.newaxis, :]) @ rmat_inv_dr[:, :, k].T
            world_points_flat_dp[:, :, 3] = - numpy.array([1.0, 0.0, 0.0], dtype=Package.get_float_dtype())[numpy.newaxis, :] @ rmat_inv.T
            world_points_flat_dp[:, :, 4] = - numpy.array([0.0, 1.0, 0.0], dtype=Package.get_float_dtype())[numpy.newaxis, :] @ rmat_inv.T
            world_points_flat_dp[:, :, 5] = - numpy.array([0.0, 0.0, 1.0], dtype=Package.get_float_dtype())[numpy.newaxis, :] @ rmat_inv.T

        if not dx:
            world_points_flat_dx = None
        if not dp:
            world_points_flat_dp = None

        return world_points_flat, world_points_flat_dx, world_points_flat_dp


    def _compute_rays(self, normalized_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Computes the rays from the camera to the scene for the the extrinsic model in the world coordinate system.

        The ray structure is as follows:

        - The first 3 elements are the origin of the ray in the world coordinate system (the normalized points with z=1 and a change of coordinate system).
        - The last 3 elements are the direction of the ray in the world coordinate system, which is always (0, 0, 1) in the camera coordinate system.

        Lets :math:`\vec{X}_n` the 3D normalized points, with coordinates :math:`(x_n, y_n, 1.0)` in the camera coordinate system.

        The points in the world coordinate system are computed as follows:

        .. math::
    
            \begin{align*}
            \vec{X}_w &= R^{-1} \cdot (\vec{X}_n - T) \\
            \vec{O}_w &= - R^{-1} \cdot T 
            \end{align*}

        The origin of the ray in the world coordinate system is the normalized points :math:`\vec{X}_w` in world coordinates and the direction of the ray is the normalized vector (0, 0, 1) in which the transformation is applied.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in the camera coordinate system. Shape (Npoints, 2).

        Returns
        -------
        numpy.ndarray
            The rays in the world coordinate system. Shape (Npoints, 6).
        """
        # Get the number of points
        Npoints = normalized_points.shape[0]

        # Get the rotation matrix and translation vector
        rmat, _ = cv2.Rodrigues(self._rvec)
        rmat_inv = rmat.T
        tvec = self._tvec

        # Compute the origin of the ray in the world coordinate system
        vector_camera = numpy.array([0.0, 0.0, 1.0], dtype=Package.get_float_dtype()) # shape (3,)
        vector_world = vector_camera @ rmat_inv.T  # shape (3,)
        direction_world = vector_world / numpy.linalg.norm(vector_world)  # shape (3,)

        # Compute the normalized points in the world coordinate system
        normalized_points_world = (numpy.concatenate((normalized_points, numpy.ones((Npoints, 1), dtype=Package.get_float_dtype())), axis=1) - tvec[numpy.newaxis, :]) @ rmat_inv.T # shape (Npoints, 3)

        # Compute the direction of the ray in the world coordinate system
        direction_world = direction_world[numpy.newaxis, :].repeat(Npoints, axis=0) # shape (Npoints, 3)

        # Create the rays in the world coordinate system
        rays = numpy.empty((Npoints, 6), dtype=Package.get_float_dtype()) # shape (Npoints, 6)
        rays[:, :3] = normalized_points_world # The first 3 elements are the origin of the ray in the world coordinate system
        rays[:, 3:] = direction_world # The last 3 elements are the direction of the ray in the world coordinate system

        return rays