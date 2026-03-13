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

from typing import Optional, Dict
from numpy.typing import ArrayLike
import numpy


from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic


def undistort_points(
    image_points: ArrayLike,
    intrinsic: Optional[Intrinsic],
    distortion: Optional[Distortion],
    R: Optional[Extrinsic] = None,
    P: Optional[Intrinsic] = None,
    *,
    transpose: bool = False,
    inverse_intrinsic_kwargs: Optional[Dict] = None,
    inverse_distortion_kwargs: Optional[Dict] = None,
    R_kwargs: Optional[Dict] = None,
    P_kwargs: Optional[Dict] = None,
) -> numpy.ndarray:
    r"""
    Undistort 2D ``image_points`` using the camera intrinsic, distortion
    transformations to obtain the ``normalized_points`` in the normalized coordinate
    system or ``undistorted_points`` if ``R`` or ``P`` are provided.

    As a reminder,

    .. math::

        \vec{x}_d = \text{Intrinsic}^{-1}(\vec{x}_i) \\
        \vec{x}_n = \text{Distortion}^{-1}(\vec{x}_d)

    Then optionally,

    .. math::

        \vec{x}_u = P(R(\vec{x}_n)))

    Where:

    - :math:`\vec{x}_i` are the 2D ``image_points`` in the image coordinate system :math:`(\vec{e}_x, \vec{e}_y)`.
    - :math:`\vec{x}_d` are the 2D ``distorted_points`` in the normalized coordinate system :math:`(\vec{I}, \vec{J})`.
    - :math:`\vec{x}_n` are the 2D ``normalized_points`` in the normalized coordinate system :math:`(\vec{I}, \vec{J})`.
    - :math:`\vec{x}_u` are the 2D ``undistorted_points`` in the space required by the user.

    .. note::

        The behavior of the function can be adapted to the user's needs by providing
        the appropriate transformations.

        - Use ``intrinsic = None`` to give directly the distorted points in the normalized coordinate system.
        - Use ``P = intrinsic`` to return the undistorted points in the image coordinate system.
        - Use ``intrinsic = None`` and ``P = None`` if the distortion model is defined in the image coordinate system.

    .. warning::

        Iterative non-linear optimization is used to find the undistorted points.

    The given points ``image_points`` are assumed to be in the sensor coordinate system
    and expressed in 2D coordinates with shape (..., 2).

    .. note::

        The expected ``image_points`` can be extracted from the ``pixel_points``
        by swapping the axes.


    Parameters
    ----------
    image_points : ArrayLike
        The 2D image points in the image coordinate system. Shape (..., 2)

    intrinsic : Optional[:class:`Intrinsic`]
        The intrinsic transformation to be applied to the image points.
        If None, a no intrinsic transformation is applied (i.e., identity
        transformation).

    distortion : Optional[:class:`Distortion`]
        The distortion model to be applied to the normalized points.
        If None, a no distortion transformation is applied (i.e., identity
        transformation).

    R : Optional[:class:`Extrinsic`], optional
        The rectification extrinsic transformation (rotation and translation) to be
        applied to the normalized points.
        If None, a no extrinsic transformation is applied (i.e., identity
        transformation). Default is None.

    P : Optional[:class:`Intrinsic`], optional
        The projection intrinsic transformation to be applied to the normalized points.
        If None, a no intrinsic transformation is applied (i.e., identity
        transformation). This is useful to return the undistorted points in the image
        coordinate system.

    transpose : bool, optional
        If True, the input points are assumed to be in the shape (2, ...) instead of
        (..., 2). Default is False.
        The output points will be in the same shape as the input points.

    inverse_intrinsic_kwargs : Optional[Dict], optional
        Additional keyword arguments to be passed to the intrinsic inverse
        transformation (``intrinsic._inverse_transform``).
        Default is None.

    inverse_distortion_kwargs : Optional[Dict], optional
        Additional keyword arguments to be passed to the distortion inverse
        transformation (``distortion._inverse_transform``).
        Default is None.

    R_kwargs : Optional[Dict], optional
        Additional keyword arguments to be passed to the rectification extrinsic
        transformation (``R._transform``). Default is None.

    P_kwargs : Optional[Dict], optional
        Additional keyword arguments to be passed to the projection intrinsic
        transformation (``P._transform``). Default is None.


    Returns
    -------
    numpy.ndarray
        The 2D normalized points in the normalized coordinate system with shape
        (..., 2) or the 2D undistorted points in the user coordinate system
        if ``P`` or ``R`` are given.


    See Also
    --------
    pycvcam.distort_points
        Similar to this function but applies the transformations in the opposite
        direction to distort the points instead of undistorting them.

    pycvcam.undistort_image
        Undistort an image using the camera intrinsic and distortion transformations.

    pycvcam.project_points
        Project 3D points to 2D image points using the camera intrinsic, distortion,
        and extrinsic transformations.



    Example
    --------
    The following example shows how to undistort 2D image points using the intrinsic
    camera matrix and a distortion model.

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
        normalized_points = undistort_points(
            image_points,
            intrinsic=intrinsic,
            distortion=distortion
        )

    To return the undistorted points in the image coordinate system, you can provide
    a projection P equal to the intrinsic K:

    .. code-block:: python

        undistorted_points = undistort_points(
            image_points,
            intrinsic=intrinsic,
            distortion=distortion,
            P=intrinsic
        )

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
    if inverse_intrinsic_kwargs is None:
        inverse_intrinsic_kwargs = {}
    if inverse_distortion_kwargs is None:
        inverse_distortion_kwargs = {}
    if R_kwargs is None:
        R_kwargs = {}
    if P_kwargs is None:
        P_kwargs = {}

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError(
            "The intrinsic object must be ready to transform the points, check is_set() method."
        )
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError(
            "The distortion object must be ready to transform the points, check is_set() method."
        )
    if not isinstance(R, Extrinsic):
        raise ValueError("R must be an instance of the Extrinsic class")
    if not R.is_set():
        raise ValueError(
            "The rectification extrinsic object must be ready to transform the points, check is_set() method."
        )
    if not isinstance(P, Intrinsic):
        raise ValueError("P must be an instance of the Intrinsic class")
    if not P.is_set():
        raise ValueError(
            "The projection intrinsic object must be ready to transform the points, check is_set() method."
        )

    if not isinstance(inverse_intrinsic_kwargs, dict):
        raise ValueError("inverse_intrinsic_kwargs must be a dictionary")
    if not isinstance(inverse_distortion_kwargs, dict):
        raise ValueError("inverse_distortion_kwargs must be a dictionary")
    if not isinstance(R_kwargs, dict):
        raise ValueError("R_kwargs must be a dictionary")
    if not isinstance(P_kwargs, dict):
        raise ValueError("P_kwargs must be a dictionary")

    if not isinstance(transpose, bool):
        raise ValueError("transpose must be a boolean value")

    # Create the array of points
    image_points = numpy.asarray(image_points, dtype=numpy.float64)

    # Transpose the points if needed
    if transpose:
        image_points = numpy.moveaxis(image_points, 0, -1)  # (2, ...) -> (..., 2)

    # Extract the original shape
    shape = image_points.shape  # (..., 2)

    # Flatten the points along the last axis
    image_points = image_points.reshape(
        -1, shape[-1]
    )  # shape (..., 2) -> shape (n_points, 2)

    # Check the shape of the points
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError(
            f"The points must be in the shape (..., 2) or (2, ...) if ``transpose`` is True. Got {image_points.shape} instead and transpose is {transpose}."
        )

    n_points = image_points.shape[0]  # n_points
    output_points = image_points.copy()  # shape (n_points, 2)

    # Realize the transformation:
    if not isinstance(intrinsic, NoIntrinsic):
        output_points, _, _ = intrinsic._inverse_transform(
            output_points, dx=False, dp=False, **inverse_intrinsic_kwargs
        )  # shape (n_points, 2) -> shape (n_points, 2)
    if not isinstance(distortion, NoDistortion):
        output_points, _, _ = distortion._inverse_transform(
            output_points, dx=False, dp=False, **inverse_distortion_kwargs
        )  # shape (n_points, 2) -> shape (n_points, 2)
    if not isinstance(R, NoExtrinsic):
        output_points, _, _ = R._transform(
            numpy.concatenate((output_points, numpy.ones((n_points, 1))), axis=1),
            dx=False,
            dp=False,
            **R_kwargs,
        )  # shape (n_points, 2) -> shape (n_points, 3)
        output_points = output_points[
            :, :2
        ]  # shape (n_points, 3) -> shape (n_points, 2)

    if not isinstance(P, NoIntrinsic):
        output_points, _, _ = P._transform(
            output_points, dx=False, dp=False, **P_kwargs
        )  # shape (n_points, 2) -> shape (n_points, 2)

    # Reshape the normalized points back to the original shape
    output_points = output_points.reshape(shape)  # shape (n_points, 2) -> (..., 2)

    # Transpose the points back to the original shape if needed
    if transpose:
        output_points = numpy.moveaxis(output_points, -1, 0)  # (..., 2) -> (2, ...)

    return output_points


def distort_points(
    image_points: ArrayLike,
    intrinsic: Optional[Intrinsic],
    distortion: Optional[Distortion],
    R: Optional[Extrinsic] = None,
    P: Optional[Intrinsic] = None,
    *,
    transpose: bool = False,
    inverse_intrinsic_kwargs: Optional[Dict] = None,
    distortion_kwargs: Optional[Dict] = None,
    R_kwargs: Optional[Dict] = None,
    P_kwargs: Optional[Dict] = None,
) -> numpy.ndarray:
    r"""
    Distort 2D ``image_points`` using the camera intrinsic, distortion
    transformations to obtain the ``distorted_points`` in the normalized coordinate
    system or ``transformed_points`` if ``R`` or ``P`` are provided.

    As a reminder,

    .. math::

        \vec{x}_n = \text{Intrinsic}^{-1}(\vec{x}_i) \\
        \vec{x}_d = \text{Distortion}(\vec{x}_n)

    Then optionally,

    .. math::

        \vec{x}_u = P(R(\vec{x}_d)))

    Where:

    - :math:`\vec{x}_i` are the 2D ``image_points`` in the image coordinate system :math:`(\vec{e}_x, \vec{e}_y)`.
    - :math:`\vec{x}_d` are the 2D ``distorted_points`` in the normalized coordinate system :math:`(\vec{I}, \vec{J})`.
    - :math:`\vec{x}_n` are the 2D ``normalized_points`` in the normalized coordinate system :math:`(\vec{I}, \vec{J})`.
    - :math:`\vec{x}_u` are the 2D ``transformed_points`` in the space required by the user.

    .. note::

        The behavior of the function can be adapted to the user's needs by providing
        the appropriate transformations.

        - Use ``intrinsic = None`` to give directly the normalized points in the normalized coordinate system.
        - Use ``P = intrinsic`` to return the distorted points in the image coordinate system.
        - Use ``intrinsic = None`` and ``P = None`` if the distortion model is defined in the image coordinate system.


    The given points ``image_points`` are assumed to be in the sensor coordinate system
    and expressed in 2D coordinates with shape (..., 2).

    .. note::

        The expected ``image_points`` can be extracted from the ``pixel_points``
        by swapping the axes.


    Parameters
    ----------
    image_points : ArrayLike
        The 2D image points in the image coordinate system. Shape (..., 2)

    intrinsic : Optional[:class:`Intrinsic`]
        The intrinsic transformation to be applied to the image points.
        If None, a no intrinsic transformation is applied (i.e., identity
        transformation).

    distortion : Optional[:class:`Distortion`]
        The distortion model to be applied to the normalized points.
        If None, a no distortion transformation is applied (i.e., identity
        transformation).

    R : Optional[:class:`Extrinsic`], optional
        The rectification extrinsic transformation (rotation and translation) to be
        applied to the distorted points.
        If None, a no extrinsic transformation is applied (i.e., identity
        transformation). Default is None.

    P : Optional[:class:`Intrinsic`], optional
        The projection intrinsic transformation to be applied to the distorted points.
        If None, a no intrinsic transformation is applied (i.e., identity
        transformation). This is useful to return the distorted points in the image
        coordinate system.

    transpose : bool, optional
        If True, the input points are assumed to be in the shape (2, ...) instead of
        (..., 2). Default is False.
        The output points will be in the same shape as the input points.

    inverse_intrinsic_kwargs : Optional[Dict], optional
        Additional keyword arguments to be passed to the intrinsic inverse
        transformation (``intrinsic._inverse_transform``).
        Default is None.

    distortion_kwargs : Optional[Dict], optional
        Additional keyword arguments to be passed to the distortion transformation
        (``distortion._transform``).
        Default is None.

    R_kwargs : Optional[Dict], optional
        Additional keyword arguments to be passed to the rectification extrinsic
        transformation (``R._transform``). Default is None.

    P_kwargs : Optional[Dict], optional
        Additional keyword arguments to be passed to the projection intrinsic
        transformation (``P._transform``). Default is None.


    Returns
    -------
    numpy.ndarray
        The 2D  distorted points in the normalized coordinate system with shape
        (..., 2) or the 2D transformed points in the user coordinate system
        if ``P`` or ``R`` are given.


    See Also
    --------
    pycvcam.undistort_points
        Similar to this function but applies the transformations in the opposite
        direction to undistort the points instead of distorting them.

    pycvcam.distort_image
        Distort an image using the camera intrinsic and distortion transformations.

    pycvcam.project_points
        Project 3D points to 2D image points using the camera intrinsic, distortion,
        and extrinsic transformations.



    Example
    --------
    The following example shows how to distort 2D image points using the intrinsic
    camera matrix and a distortion model.

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
        normalized_points = undistort_points(
            image_points,
            intrinsic=intrinsic,
            distortion=distortion
        )

    To return the undistorted points in the image coordinate system, you can provide
    a projection P equal to the intrinsic K:

    .. code-block:: python

        undistorted_points = undistort_points(
            image_points,
            intrinsic=intrinsic,
            distortion=distortion,
            P=intrinsic
        )

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
    if inverse_intrinsic_kwargs is None:
        inverse_intrinsic_kwargs = {}
    if distortion_kwargs is None:
        distortion_kwargs = {}
    if R_kwargs is None:
        R_kwargs = {}
    if P_kwargs is None:
        P_kwargs = {}

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError(
            "The intrinsic object must be ready to transform the points, check is_set() method."
        )
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError(
            "The distortion object must be ready to transform the points, check is_set() method."
        )
    if not isinstance(R, Extrinsic):
        raise ValueError("R must be an instance of the Extrinsic class")
    if not R.is_set():
        raise ValueError(
            "The rectification extrinsic object must be ready to transform the points, check is_set() method."
        )
    if not isinstance(P, Intrinsic):
        raise ValueError("P must be an instance of the Intrinsic class")
    if not P.is_set():
        raise ValueError(
            "The projection intrinsic object must be ready to transform the points, check is_set() method."
        )

    if not isinstance(inverse_intrinsic_kwargs, dict):
        raise ValueError("inverse_intrinsic_kwargs must be a dictionary")
    if not isinstance(distortion_kwargs, dict):
        raise ValueError("distortion_kwargs must be a dictionary")
    if not isinstance(R_kwargs, dict):
        raise ValueError("R_kwargs must be a dictionary")
    if not isinstance(P_kwargs, dict):
        raise ValueError("P_kwargs must be a dictionary")

    if not isinstance(transpose, bool):
        raise ValueError("transpose must be a boolean value")

    # Create the array of points
    image_points = numpy.asarray(image_points, dtype=numpy.float64)

    # Transpose the points if needed
    if transpose:
        image_points = numpy.moveaxis(image_points, 0, -1)  # (2, ...) -> (..., 2)

    # Extract the original shape
    shape = image_points.shape  # (..., 2)

    # Flatten the points along the last axis
    image_points = image_points.reshape(
        -1, shape[-1]
    )  # shape (..., 2) -> shape (n_points, 2)

    # Check the shape of the points
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError(
            f"The points must be in the shape (..., 2) or (2, ...) if ``transpose`` is True. Got {image_points.shape} instead and transpose is {transpose}."
        )

    n_points = image_points.shape[0]  # n_points
    output_points = image_points.copy()  # shape (n_points, 2)

    # Realize the transformation:
    if not isinstance(intrinsic, NoIntrinsic):
        output_points, _, _ = intrinsic._inverse_transform(
            output_points, dx=False, dp=False, **inverse_intrinsic_kwargs
        )  # shape (n_points, 2) -> shape (n_points, 2)
    if not isinstance(distortion, NoDistortion):
        output_points, _, _ = distortion._transform(
            output_points, dx=False, dp=False, **distortion_kwargs
        )  # shape (n_points, 2) -> shape (n_points, 2)
    if not isinstance(R, NoExtrinsic):
        output_points, _, _ = R._transform(
            numpy.concatenate((output_points, numpy.ones((n_points, 1))), axis=1),
            dx=False,
            dp=False,
            **R_kwargs,
        )  # shape (n_points, 2) -> shape (n_points, 3)
        output_points = output_points[
            :, :2
        ]  # shape (n_points, 3) -> shape (n_points, 2)

    if not isinstance(P, NoIntrinsic):
        output_points, _, _ = P._transform(
            output_points, dx=False, dp=False, **P_kwargs
        )  # shape (n_points, 2) -> shape (n_points, 2)

    # Reshape the normalized points back to the original shape
    output_points = output_points.reshape(shape)  # shape (n_points, 2) -> (..., 2)

    # Transpose the points back to the original shape if needed
    if transpose:
        output_points = numpy.moveaxis(output_points, -1, 0)  # (..., 2) -> (2, ...)

    return output_points
