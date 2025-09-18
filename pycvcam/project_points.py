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

from .core.transform import TransformResult
from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic
from .core.package import Package

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic



def project_points(
        world_points: numpy.ndarray, 
        intrinsic: Optional[Intrinsic],
        distortion: Optional[Distortion],
        extrinsic: Optional[Extrinsic],
        *,
        transpose: bool = False,
        dx: bool = False, 
        dp: bool = False,
        _skip: bool = False,
        **kwargs
    ) -> TransformResult:
    r"""
    Project 3D ``world_points`` to 2D ``image_points`` using the camera intrinsic, distortion and extrinsic transformations.

    As a reminder,

    .. math::

        \begin{align*}
        \vec{x}_n = \text{Extrinsic}(\vec{X}_w) \\
        \vec{x}_d = \text{Distortion}(\vec{x}_n) \\
        \vec{x}_i = \text{Intrinsic}(\vec{x}_d) \\
        \end{align*}

    Where:

    - :math:`\vec{X}_w` are the 3D ``world_points`` in the world coordinate system.
    - :math:`\vec{x}_n` are the 2D ``normalized_points`` in the normalized coordinate system.
    - :math:`\vec{x}_d` are the 2D ``distorted_points`` in the normalized coordinate system.
    - :math:`\vec{x}_i` are the 2D ``image_points`` in the image coordinate system.

    The ``image_points`` can be then converted to pixel coordinates by applying a swap of the axes.

    To compute the Jacobians of the image points with respect to the input 3D world points and the projection parameters, set the ``dx`` and ``dp`` parameters to True.
    The Jacobians are computed using the chain rule of differentiation and are returned in the result object.

    To access the Jacobians, you can use the following properties of the result object:

    - ``jacobian_dx``: The Jacobian of the image points with respect to the input 3D world points. Shape (..., 2, 3).
    - ``jacobian_dp``: The Jacobian of the image points with respect to the projection parameters (extrinsic, distortion, intrinsic). Shape (..., 2, Nextrinsic + Ndistortion + Nintrinsic).
    - ``jacobian_dintrinsic``: Alias for ``jacobian_dp[..., :Nintrinsic]`` to represent the Jacobian with respect to the intrinsic parameters. Shape (..., 2, Nintrinsic).
    - ``jacobian_ddistortion``: Alias for ``jacobian_dp[..., Nintrinsic:Nintrinsic + Ndistortion]`` to represent the Jacobian with respect to the distortion parameters. Shape (..., 2, Ndistortion).
    - ``jacobian_dextrinsic``: Alias for ``jacobian_dp[..., Nintrinsic + Ndistortion:]`` to represent the Jacobian with respect to the extrinsic parameters. Shape (..., 2, Nextrinsic).

    .. warning::

        The points are converting to float before applying the inverse transformation.
        See :class:`pycvcam.core.Package` for more details on the default data types used in the package.

    Parameters
    ----------
    world_points : numpy.ndarray
        The 3D points in the world coordinate system. Shape (..., 3).
    
    intrinsic : Optional[Intrinsic]
        The intrinsic transformation to be applied to the distorted points.
        If None, a no intrinsic transformation is applied (identity intrinsic).

    distortion : Optional[Distortion]
        The distortion model to be applied to the normalized points.
        If None, a no distortion transformation is applied (identity distortion).

    extrinsic : Optional[Extrinsic]
        The extrinsic transformation to be applied to the 3D world points.
        If None, a no extrinsic transformation is applied (identity transformation).

    transpose : bool, optional
        If True, the input points are assumed to be in the shape (3, ...) instead of (..., 3). Default is False.
        In this case, the output points will be in the shape (2, ...) and the jacobians will be in the shape (2, ..., 3) and (2, ..., Nparams) respectively.
        
    dx : bool, optional
        If True, compute the Jacobian of the image points with respect to the input 3D world points with shape (..., 2, 3).
        If False, the Jacobian is not computed. default is False.

    dp : bool, optional
        If True, compute the Jacobian of the image points with respect to the projection parameters with shape (..., 2, Nparams).
        If False, the Jacobian is not computed. Default is False.

    _skip : bool, optional
        [INTERNAL USE], If True, skip the checks for the transformation parameters and assume the points are given in the (Npoints, input_dim) float format.
        `transpose` is ignored if this parameter is set to True.

    **kwargs : dict
        Additional keyword arguments to be passed. [warning]
        
    Returns
    -------
    TransformResult
        The result of the projection transformation.
        
    Examples
    --------
    Create a simple example to project 3D points to 2D image points using the intrinsic and extrinsic parameters of the camera.

    .. code-block:: python

        import numpy
        from pycvcam import project_points, Cv2Distortion, Cv2Extrinsic, Cv2Intrinsic

        # Define the 3D points in the world coordinate system
        world_points = numpy.array([[0.0, 0.0, 5.0],
                                    [0.1, -0.1, 5.0],
                                    [-0.1, 0.2, 5.0],
                                    [0.2, 0.1, 5.0],
                                    [-0.2, -0.2, 5.0]]) # shape (5, 3)

        # Define the rotation vector and translation vector
        rvec = numpy.array([0.01, 0.02, 0.03])  # small rotation
        tvec = numpy.array([0.1, -0.1, 0.2])    # small translation
        extrinsic = Cv2Extrinsic.from_rt(rvec, tvec)

        # Define the intrinsic camera matrix
        K = numpy.array([[1000.0, 0.0, 320.0],
                        [0.0, 1000.0, 240.0],
                        [0.0, 0.0, 1.0]])

        intrinsic = Cv2Intrinsic.from_matrix(K)

        # Define the distortion model (optional)
        distortion = Cv2Distortion(parameters = [0.1, 0.2, 0.3, 0.4, 0.5])

        # Project the 3D points to 2D image points
        result = project_points(world_points, intrinsic=intrinsic, distortion=distortion, extrinsic=extrinsic, transpose=False)
        print("Projected image points:")
        print(result.image_points) # shape (5, 2)

    You can also compute the Jacobians of the image points with respect to the input 3D world points and the projection parameters by setting the ``dx`` and ``dp`` parameters to True.

    .. code-block:: python

        # Project the 3D points to 2D image points with Jacobians
        result = project_points(world_points, intrinsic=intrinsic, distortion=distortion, extrinsic=extrinsic, transpose=False, dx=True, dp=True)

        print("Jacobian with respect to 3D points:")
        print(result.jacobian_dx) # shape (5, 2, 3)
        print("Jacobian with respect to projection parameters:")
        print(result.jacobian_dp) # shape (5, 2, Nparams)
        print("Jacobian with respect to extrinsic parameters:")
        print(result.jacobian_dextrinsic) # shape (5, 2, Nextrinsic) -> ordered as given by the selected extrinsic object
        print("Jacobian with respect to distortion parameters:")
        print(result.jacobian_ddistortion) # shape (5, 2, Ndistortion) -> ordered as given by the selected distortion object
        print("Jacobian with respect to intrinsic parameters:")
        print(result.jacobian_dintrinsic) # shape (5, 2, Nintrinsic) -> ordered as given by the selected intrinsic object

    This method can also be used without any extrinsic, distortion or intrinsic parameters by passing None.
    """
    # Set the default values if None
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if extrinsic is None:
        extrinsic = NoExtrinsic()
    if distortion is None:
        distortion = NoDistortion()

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError("The intrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(extrinsic, Extrinsic):
        raise ValueError("extrinsic must be an instance of the Extrinsic class")
    if not extrinsic.is_set():
        raise ValueError("The extrinsic object must be ready to transform the points, check is_set() method.")
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError("The distortion object must be ready to transform the points, check is_set() method.")

    # Initialize the jacobians
    jacobian_dx = None
    jacobian_dp = None

    if not _skip:
        if not isinstance(transpose, bool):
            raise ValueError("transpose must be a boolean value")
        if not isinstance(dx, bool):
            raise ValueError("dx must be a boolean value")
        if not isinstance(dp, bool):
            raise ValueError("dp must be a boolean value")        

        # Create the array of points
        world_points = numpy.asarray(world_points, dtype=Package.get_float_dtype())

        # Transpose the points if needed
        if transpose:
            world_points = numpy.moveaxis(world_points, 0, -1) # (3, ...) -> (..., 3)

        # Extract the original shape
        shape = world_points.shape # (..., 3)

        # Flatten the points along the last axis
        world_points = world_points.reshape(-1, shape[-1]) # shape (..., 3) -> shape (Npoints, 3)

        # Check the shape of the points
        if world_points.ndim !=2 or world_points.shape[1] != 3:
            raise ValueError(f"The points must be in the shape (..., 3) or (3, ...) if ``transpose`` is True. Got {shape} instead and transpose is {transpose}.")
        
    # Extract the useful constants
    Npoints = world_points.shape[0] # Npoints
    Nparams = intrinsic.Nparams + distortion.Nparams + extrinsic.Nparams # Total number of parameters

    # Realize the transformation:
    normalized_points, extrinsic_jacobian_dx, extrinsic_jacobian_dp = extrinsic._transform(world_points, dx=dx, dp=dp)
    distorted_points, distortion_jacobian_dx, distortion_jacobian_dp = distortion._transform(normalized_points, dx=dx or dp, dp=dp, **kwargs) # (dx is requiered for propagation of dp)
    image_points, intrinsic_jacobian_dx, intrinsic_jacobian_dp = intrinsic._transform(distorted_points, dx=dx or dp, dp=dp) # (dx is requiered for propagation of dp)

    # Apply the chain rules to compute the Jacobians with respect to the projection parameters
    if dp:
        jacobian_flat_dp = numpy.empty((Npoints, 2, Nparams), dtype=Package.get_float_dtype())
        # wrt the extrinsic parameters
        if isinstance(intrinsic, NoIntrinsic) and isinstance(distortion, NoDistortion):
            jacobian_flat_dp[..., intrinsic.Nparams + distortion.Nparams:] = extrinsic_jacobian_dp
        elif isinstance(intrinsic, NoIntrinsic):
            jacobian_flat_dp[..., intrinsic.Nparams + distortion.Nparams:] = numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dp)
        elif isinstance(distortion, NoDistortion):
            jacobian_flat_dp[..., intrinsic.Nparams + distortion.Nparams:] = numpy.matmul(intrinsic_jacobian_dx, extrinsic_jacobian_dp)
        else:
            jacobian_flat_dp[..., intrinsic.Nparams + distortion.Nparams:] = numpy.matmul(intrinsic_jacobian_dx, numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dp))

        # wrt the distortion parameters
        if intrinsic is None:
            jacobian_flat_dp[..., intrinsic.Nparams:intrinsic.Nparams + distortion.Nparams] = distortion_jacobian_dp
        else:
            jacobian_flat_dp[..., intrinsic.Nparams:intrinsic.Nparams + distortion.Nparams] = numpy.matmul(intrinsic_jacobian_dx, distortion_jacobian_dp)

        # wrt the intrinsic parameters
        jacobian_flat_dp[..., :intrinsic.Nparams] = intrinsic_jacobian_dp # (intrinsic parameters)

    # Apply the chain rules to compute the Jacobians with respect to the input 3D world points
    if dx:
        if isinstance(intrinsic, NoIntrinsic) and isinstance(distortion, NoDistortion):
            jacobian_flat_dx = extrinsic_jacobian_dx
        elif isinstance(intrinsic, NoIntrinsic):
            jacobian_flat_dx = numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dx)
        elif isinstance(distortion, NoDistortion):
            jacobian_flat_dx = numpy.matmul(intrinsic_jacobian_dx, extrinsic_jacobian_dx)
        else:
            jacobian_flat_dx = numpy.matmul(intrinsic_jacobian_dx, numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dx)) # shape (Npoints, 2, 3)

    if not _skip:
        # Reshape the normalized points back to the original shape (Warning shape is (..., 3) and not (..., 2))
        image_points = image_points.reshape((*shape[:-1], 2)) # shape (Npoints, 2) -> (..., 2)
        jacobian_dx = jacobian_flat_dx.reshape((*shape[:-1], 2, 3)) if dx else None # shape (Npoints, 2, 3) -> (..., 2, 3)
        jacobian_dp = jacobian_flat_dp.reshape((*shape[:-1], 2, Nparams)) if dp else None # shape (Npoints, 2, Nparams) -> (..., 2, Nparams)

        # Transpose the points back to the original shape if needed
        if transpose:
            image_points = numpy.moveaxis(image_points, -1, 0) # (..., 2) -> (2, ...)
            jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if dx else None # (..., 2, 2) -> (2, ..., 2)
            jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if dp else None # (..., 2, Nparams) -> (2, ..., Nparams)

    # Return the result
    result = TransformResult(
        transformed_points=image_points,
        jacobian_dx=jacobian_dx,
        jacobian_dp=jacobian_dp,
        transpose=transpose
    )

    # Add the short-hand properties for the jacobians
    result.add_jacobian("dintrinsic", 0, intrinsic.Nparams, f"Jacobian of the image points with respect to the intrinsic parameters (see {intrinsic.__class__.__name__}) for more details on their order")
    result.add_jacobian("ddistortion", intrinsic.Nparams, intrinsic.Nparams + distortion.Nparams, f"Jacobian of the image points with respect to the distortion parameters (see {distortion.__class__.__name__}) for more details on their order")
    result.add_jacobian("dextrinsic", intrinsic.Nparams + distortion.Nparams, Nparams, f"Jacobian of the image points with respect to the extrinsic parameters (see {extrinsic.__class__.__name__}) for more details on their order")

    # Add the alias for the transformed points
    result.add_alias("image_points")
    return result


