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
import cv2
import scipy

from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic
from .core.rays import Rays
from .core.package import Package

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic

def compute_rays(
    image_points: numpy.ndarray,
    intrinsic: Optional[Intrinsic],
    distortion: Optional[Distortion],
    extrinsic: Optional[Extrinsic],
    *,
    transpose: bool = False,
    _skip: bool = False,
    **kwargs
    ) -> Rays:
    r"""

    Compute the rays emitted from the camera to the scene based on the given image points, intrinsic parameters, distortion model, and extrinsic parameters.

    The process to compute the rays is as follows:

    1. The ``image_points`` (:math:`\vec{x}_i`) are normalized by the inverse intrinsic matrix transformation to obtain the ``distorted_points`` (:math:`\vec{x}_d`).
    2. The ``distorted_points`` (:math:`\vec{x}_d`) are undistorted by the distortion model using the coefficients :math:`\{\lambda_1, \lambda_2, \lambda_3, \ldots\}` to obtain the ``normalized_points`` (:math:`\vec{x}_n`).
    3. The ``normalized_points`` (:math:`\vec{x}_n`) are used to compute the rays in the world coordinate system using the extrinsic parameters to obtain the ``rays``.

    The ray structure is as follows:

    - The first 3 elements are the origin of the ray in the world coordinate system.
    - The last 3 elements are the direction of the ray in the world coordinate system. The direction vector is normalized.

    .. seealso::

        - :class:`pycvcam.core.Rays` for the rays structure.

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

    extrinsic : Optional[Extrinsic]
        The extrinsic transformation (rotation and translation) to be applied to the normalized points.
        If None, a no extrinsic transformation is applied (i.e., identity transformation).

    transpose : bool, optional
        If True, the input image points are transposed before processing, the input shape is expected to be (2, ...) instead of (..., 2) and the output shape will be (6, ...).
        Default is False.

    _skip : bool, optional
            [INTERNAL USE], If True, skip the checks for the transformation parameters and assume the points are given in the (Npoints, input_dim) float format.
            `transpose` is ignored if this parameter is set to True.

    **kwargs : dict
        Additional keyword arguments for distortion models ``undistort`` method.

    Returns
    -------
    Rays
        The rays in the world coordinate system.

    Example
    -------

    Create a simple example to construct the rays from an image to the scene:

    .. code-block:: python

        import numpy
        import cv2
        from pycvcam import compute_rays, Cv2Extrinsic, Cv2Intrinsic, Cv2Distortion

        # Read the image : 
        image = cv2.imread('image.jpg')
        image_height, image_width = image.shape[:2]

        # Construct the intrinsic transformation :
        intrinsic = Cv2Intrinsic.from_matrix(numpy.array([[1000, 0, image_width / 2],
                                                          [0, 1000, image_height / 2],
                                                          [0, 0, 1]]))

        # Construct the distortion transformation:
        distortion = Cv2Distortion(parameters=numpy.array([0.1, -0.05, 0, 0, 0]))

        # Construct the extrinsic transformation:
        extrinsic = Cv2Extrinsic.from_rt(rvec=[0.1, 0.2, 0.3], tvec=[0, 0, 5])

        # Define the image points (e.g., pixels in the image):
        pixel_points = numpy.indices((image_height, image_width)) # shape (2, H, W)
        pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
        image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

        # Throw rays from the image points to the scene:
        rays = compute_rays(image_points, intrinsic, distortion, extrinsic, transpose=False).rays

        # Here `rays` will contain the origin and direction of the rays in the world coordinate system with shape (..., 6).
        # rays[i, :] = [origin_x, origin_y, origin_z, direction_x, direction_y, direction_z]

    """
    # Set the default values if None
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if distortion is None:
        distortion = NoDistortion()
    if extrinsic is None:
        extrinsic = NoExtrinsic()

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError("The distortion object must be ready to transform the points, check is_set() method.")
    if not isinstance(extrinsic, Extrinsic):
        raise ValueError("extrinsic must be an instance of the Extrinsic class")
    if not extrinsic.is_set():
        raise ValueError("The extrinsic object must be ready to transform the points, check is_set() method.")
    
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
    output_points = image_points
    
    # Realize the transformation:
    if not isinstance(intrinsic, NoIntrinsic):
        output_points, _, _ = intrinsic._inverse_transform(output_points, dx=False, dp=False) # shape (Npoints, 2) -> shape (Npoints, 2)
    if not isinstance(distortion, NoDistortion):
        output_points, _, _ = distortion._inverse_transform(output_points, dx=False, dp=False, **kwargs)

    # Always use the extrinsic transformation to compute the rays:
    rays = extrinsic._compute_rays(output_points) # shape (Npoints, 2) -> shape (Npoints, 6)
    
    if not _skip:
        # Reshape the rays  back to the original shape
        rays = rays.reshape((*shape[:-1], 6)) # shape (Npoints, 6) -> (..., 6)

        # Transpose the rays back to the original shape if needed
        if transpose:
            rays = numpy.moveaxis(rays, -1, 0) # (..., 6) -> (6, ...)

    return Rays(rays=rays, transpose=transpose)