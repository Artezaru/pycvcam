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

from typing import Optional
import numpy


from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic
from .core.package import Package

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic



def undistort_points(
        image_points: numpy.ndarray,
        intrinsic: Optional[Intrinsic],
        distortion: Optional[Distortion],
        R: Optional[Extrinsic] = None,
        P: Optional[Intrinsic] = None,
        *,
        transpose: bool = False,
        _skip: bool = False,
        **kwargs
    ) -> numpy.ndarray:
    r"""
    Undistort 2D ``image_points`` using the camera intrinsic, distortion transformations to obtain the ``normalized_points`` in the camera coordinate system or ``undistorted_points`` if ``R`` or ``P`` are provided.

    The process to undistort a 2D-image point is as follows:

    1. The ``image_points`` (:math:`\vec{x}_i`) are normalized by applying the inverse intrinsic application to obtain the ``distorted_points`` (:math:`\vec{x}_d`).
    2. The ``distorted_points`` (:math:`\vec{x}_d`) are undistorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``normalized_points`` (:math:`\vec{x}_n`).
    3. A rectification extrinsic operation R and a new projection intrinsic projection P can be applied to the ``normalized_points`` to return the ``undistorted_points`` in the space required by the user.

    .. note::

        The ``P = intrinsic`` to return the undistorted points in the image coordinate system.
        This allow to create the same camera model but with no distortion.

    .. warning::

        Iterative non-linear optimization is used to find the undistorted points.

    The given points ``image_points`` are assumed to be in the sensor coordinate system and expressed in 2D coordinates with shape (..., 2).
    If the user gives not give a intrinsic transformation, it equivalent to give directly the normalized points.

    .. note::

        The expected ``image_points`` can be extracted from the ``pixel_points`` by swaping the axes.

    .. warning::

        The points are converting to float before applying the inverse transformation.
        See :class:`pycvcam.core.Package` for more details on the default data types used in the package.
        
    Parameters
    ----------
    image_points : numpy.ndarray
        The 2D image points in the camera normalized coordinate system. Shape (..., 2)
        
    intrinsic : Optional[Intrinsic]
        The intrinsic transformation to be applied to the image points.
        If None, a no intrinsic transformation is applied (i.e., identity transformation).

    distortion : Optional[Distortion]
        The distortion model to be applied to the normalized points.
        If None, a no distortion transformation is applied (i.e., identity transformation).

    R : Optional[Extrinsic], optional
        The rectification extrinsic transformation (rotation and translation) to be applied to the normalized points.
        If None, a no extrinsic transformation is applied (i.e., identity transformation). Default is None.

    P : Optional[Intrinsic], optional
        The projection intrinsic transformation to be applied to the normalized points.
        If None, a no intrinsic transformation is applied (i.e., identity transformation).
        This is useful to return the undistorted points in the image coordinate system.

    transpose : bool, optional
        If True, the input points are assumed to be in the shape (2, ...) instead of (..., 2). Default is False.
        The output points will be in the same shape as the input points.

    _skip : bool, optional
            [INTERNAL USE], If True, skip the checks for the transformation parameters and assume the points are given in the (Npoints, input_dim) float format.
            `transpose` is ignored if this parameter is set to True.
    
    **kwargs : optional
        Additional keyword arguments to be passed. [warning]

    Returns
    -------
    numpy.ndarray
        The undistorted 2D image points in the camera coordinate system. Shape (..., 2). If no ``P`` is given, the ``normalized_points`` are returned instead of the ``undistorted_points``.

    Example
    --------
    The following example shows how to undistort 2D image points using the intrinsic camera matrix and a distortion model.

    .. code-block:: python

        import numpy
        from pycvcam import undistort_points, Cv2Distortion, Cv2Intrinsic

        # Define the 2D image points in the camera coordinate system
        image_points = numpy.array([[320.0, 240.0],
                                    [420.0, 440.0],
                                    [520.0, 540.0],
                                    [620.0, 640.0],
                                    [720.0, 740.0]]) # shape (5, 2)

        # Define the intrinsic camera matrix
        K = numpy.array([[1000.0, 0.0, 320.0],
                        [0.0, 1000.0, 240.0],
                        [0.0, 0.0, 1.0]])
    
        # Create the intrinsic object
        intrinsic = Cv2Intrinsic.from_matrix(K)

        # Define the distortion model (optional)
        distortion = Cv2Distortion([0.1, 0.2, 0.3, 0.4, 0.5])

        # Undistort the 2D image points
        normalized_points = undistort_points(image_points, intrinsic=intrinsic, distortion=distortion)

    To return the undistorted points in the image coordinate system, you can provide a projection P equal to the intrinsic K:

    .. code-block:: python

        undistorted_points = undistort_points(image_points, intrinsic=intrinsic, distortion=distortion, P=K)
    
    """
    # Set the default values if None
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if distortion is None:
        distortion = NoDistortion()
    if R is None:
        R = NoExtrinsic()
    if P is None:
        P = NoIntrinsic()

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError("The distortion object must be ready to transform the points, check is_set() method.")
    if not isinstance(R, Extrinsic):
        raise ValueError("R must be an instance of the Extrinsic class")
    if not R.is_set():
        raise ValueError("The rectification extrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(P, Intrinsic):
        raise ValueError("P must be an instance of the Intrinsic class")
    if not P.is_set():
        raise ValueError("The projection intrinsic object must be ready to transform the points, check is_set() method.")

    if not _skip:
        if not isinstance(transpose, bool):
            raise ValueError("transpose must be a boolean value")
        
        # Create the array of points
        image_points = numpy.asarray(image_points, dtype=Package.get_float_dtype())

        # Transpose the points if needed
        if transpose:
            image_points = numpy.moveaxis(image_points, 0, -1) # (2, ...) -> (..., 2)

        # Extract the original shape
        shape = image_points.shape # (..., 2)

        # Flatten the points along the last axis
        image_points = image_points.reshape(-1, shape[-1]) # shape (..., 2) -> shape (Npoints, 2)
        
        # Check the shape of the points
        if image_points.ndim !=2 or image_points.shape[1] != 2:
            raise ValueError(f"The points must be in the shape (..., 2) or (2, ...) if ``transpose`` is True. Got {image_points.shape} instead and transpose is {transpose}.")

    Npoints = image_points.shape[0] # Npoints
    output_points = image_points.copy() # shape (Npoints, 2)
    print(intrinsic, distortion)
    # Realize the transformation:
    if not isinstance(intrinsic, NoIntrinsic):
        output_points, _, _ = intrinsic._inverse_transform(output_points, dx=False, dp=False) # shape (Npoints, 2) -> shape (Npoints, 2)
    if not isinstance(distortion, NoDistortion):
        output_points, _, _ = distortion._inverse_transform(output_points, dx=False, dp=False, **kwargs) # shape (Npoints, 2) -> shape (Npoints, 2)
    if not isinstance(R, NoExtrinsic):
        output_points, _, _ = R._transform(numpy.concatenate((output_points, numpy.ones((Npoints, 1))), axis=1), dx=False, dp=False) # shape (Npoints, 2) -> shape (Npoints, 3)
        output_points = output_points[:, :2] # shape (Npoints, 3) -> shape (Npoints, 2)

    if not isinstance(P, NoIntrinsic):
        output_points, _, _ = P._transform(output_points, dx=False, dp=False) # shape (Npoints, 2) -> shape (Npoints, 2)

    if not _skip:
        # Reshape the normalized points back to the original shape
        output_points = output_points.reshape(shape) # shape (Npoints, 2) -> (..., 2)

        # Transpose the points back to the original shape if needed
        if transpose:
            output_points = numpy.moveaxis(output_points, -1, 0) # (..., 2) -> (2, ...)

    return output_points

