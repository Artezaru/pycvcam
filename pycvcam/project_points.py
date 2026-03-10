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
import numpy
from numpy.typing import ArrayLike

from .core.transform import TransformResult
from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic


def project_points(
    world_points: ArrayLike,
    intrinsic: Optional[Intrinsic],
    distortion: Optional[Distortion],
    extrinsic: Optional[Extrinsic],
    *,
    transpose: bool = False,
    dx: bool = False,
    dp: bool = False,
    dintrinsic: bool = False,
    ddistortion: bool = False,
    dextrinsic: bool = False,
    intrinsic_kwargs: Optional[Dict] = None,
    distortion_kwargs: Optional[Dict] = None,
    extrinsic_kwargs: Optional[Dict] = None,
) -> TransformResult:
    r"""
    Project 3D ``world_points`` :math:`\vec{X}_w` to 2D ``image_points`` 
    :math:`\vec{x}_i` using the camera intrinsic, distortion and extrinsic 
    transformations.

    As a reminder,

    .. math::

        \vec{x}_n &= \text{Extrinsic}(\vec{X}_w) \\
        \vec{x}_d &= \text{Distortion}(\vec{x}_n) \\
        \vec{x}_i &= \text{Intrinsic}(\vec{x}_d) \\

    Where:

    - :math:`\vec{X}_w` are the 3D ``world_points`` in the world coordinate system :math:`(\vec{E}_x, \vec{E}_y, \vec{E}_z)`.
    - :math:`\vec{x}_n` are the 2D ``normalized_points`` in the normalized coordinate system :math:`(\vec{I}, \vec{J})`.
    - :math:`\vec{x}_d` are the 2D ``distorted_points`` in the normalized coordinate system :math:`(\vec{I}, \vec{J})`.
    - :math:`\vec{x}_i` are the 2D ``image_points`` in the image coordinate system :math:`(\vec{e}_x, \vec{e}_y)`.

    .. note::
    
        The ``image_points`` can be then converted to pixel coordinates 
        :math:`(\vec{u}, \vec{v})` by applying a swap of the axes.

    To compute the Jacobians of the image points with respect to the input 3D world 
    points and the projection parameters, set the ``dx`` and ``dp`` parameters to True.
    The Jacobians are computed using the chain rule of differentiation and are returned 
    in the result object.

    To access the Jacobians, you can use the following properties of the result object:

    - ``jacobian_dx``: The Jacobian of the image points with respect to the input 3D world points. Shape (..., 2, 3).
    - ``jacobian_dp``: The Jacobian of the image points with respect to the projection parameters (extrinsic, distortion, intrinsic). Shape (..., 2, Nextrinsic + Ndistortion + Nintrinsic).
    - ``jacobian_dintrinsic``: Alias for ``jacobian_dp[..., :Nintrinsic]`` to represent the Jacobian with respect to the intrinsic parameters. Shape (..., 2, Nintrinsic).
    - ``jacobian_ddistortion``: Alias for ``jacobian_dp[..., Nintrinsic:Nintrinsic + Ndistortion]`` to represent the Jacobian with respect to the distortion parameters. Shape (..., 2, Ndistortion).
    - ``jacobian_dextrinsic``: Alias for ``jacobian_dp[..., Nintrinsic + Ndistortion:]`` to represent the Jacobian with respect to the extrinsic parameters. Shape (..., 2, Nextrinsic).


    Parameters
    ----------
    world_points : ArrayLike
        The 3D points in the world coordinate system. Shape (..., 3).
    
    intrinsic : Optional[:class:`Intrinsic`]
        The intrinsic transformation to be applied to the distorted points.
        If None, a no intrinsic transformation is applied (identity intrinsic).

    distortion : Optional[:class:`Distortion`]
        The distortion model to be applied to the normalized points.
        If None, a no distortion transformation is applied (identity distortion).

    extrinsic : Optional[:class:`Extrinsic`]
        The extrinsic transformation to be applied to the 3D world points.
        If None, a no extrinsic transformation is applied (identity transformation).

    transpose : bool, optional
        If True, the input points are assumed to be in the shape (3, ...) instead of 
        (..., 3). Default is False.
        In this case, the output points will be in the shape (2, ...) and the 
        jacobians will be in the shape (2, ..., 3) and (2, ..., n_params) respectively.
        
    dx : bool, optional
        If True, compute the Jacobian of the image points with respect to the input 
        3D world points with shape (..., 2, 3).
        If False, the Jacobian is not computed. default is False.

    dp : bool, optional
        If True, compute the Jacobian of the image points with respect to the projection 
        parameters with shape (..., 2, n_params).
        If True (dintrinsic, ddistortion, dextrinsic are ignored and computed
        automatically.
        If False, the Jacobian is not computed. Default is False.
        
    dintrinsic : bool, optional
        If True, compute the Jacobian of the image points with respect to the intrinsic 
        parameters only (other dp components are ignored for efficiency and set to
        nan in the result).
    
    ddistortion : bool, optional
        If True, compute the Jacobian of the image points with respect to the distortion 
        parameters only (other dp components are ignored for efficiency and set to nan 
        in the result).
        
    dextrinsic : bool, optional
        If True, compute the Jacobian of the image points with respect to the extrinsic
        parameters only (other dp components are ignored for efficiency and set to nan 
        in the result).
        
    intrinsic_kwargs : Optional[dict], optional
        Additional keyword arguments to be passed to the intrinsic transformation.
        
    distortion_kwargs : Optional[dict], optional
        Additional keyword arguments to be passed to the distortion transformation.
        
    extrinsic_kwargs : Optional[dict], optional
        Additional keyword arguments to be passed to the extrinsic transformation.
        
        
    Returns
    -------
    :class:`TransformResult`
        The result of the projection transformation containing the projected image 
        points and the Jacobians if requested in the image coordinate system.
        
        
    Examples
    --------
    
    See a complete example in the gallery: :ref:`sphx_glr__gallery_project_points.py`. 
    
    """
    # Set the default values if None
    if intrinsic is None:
        intrinsic = NoIntrinsic()
    if extrinsic is None:
        extrinsic = NoExtrinsic()
    if distortion is None:
        distortion = NoDistortion()
    if intrinsic_kwargs is None:
        intrinsic_kwargs = {}
    if distortion_kwargs is None:
        distortion_kwargs = {}
    if extrinsic_kwargs is None:
        extrinsic_kwargs = {}

    # Check the types of the parameters
    if not isinstance(intrinsic, Intrinsic):
        raise ValueError("intrinsic must be an instance of the Intrinsic class")
    if not intrinsic.is_set():
        raise ValueError(
            "The intrinsic object must be ready to transform the points, check is_set() method."
        )
    if not isinstance(extrinsic, Extrinsic):
        raise ValueError("extrinsic must be an instance of the Extrinsic class")
    if not extrinsic.is_set():
        raise ValueError(
            "The extrinsic object must be ready to transform the points, check is_set() method."
        )
    if not isinstance(distortion, Distortion):
        raise ValueError("distortion must be an instance of the Distortion class.")
    if not distortion.is_set():
        raise ValueError(
            "The distortion object must be ready to transform the points, check is_set() method."
        )

    # Initialize the jacobians
    jacobian_dx = None
    jacobian_dp = None

    if not isinstance(transpose, bool):
        raise ValueError("transpose must be a boolean value")
    if not isinstance(dx, bool):
        raise ValueError("dx must be a boolean value")
    if not isinstance(dp, bool):
        raise ValueError("dp must be a boolean value")
    if not isinstance(dintrinsic, bool):
        raise ValueError("dintrinsic must be a boolean value")
    if not isinstance(ddistortion, bool):
        raise ValueError("ddistortion must be a boolean value")
    if not isinstance(dextrinsic, bool):
        raise ValueError("dextrinsic must be a boolean value")
    if not isinstance(intrinsic_kwargs, dict):
        raise ValueError("intrinsic_kwargs must be a dictionary")
    if not isinstance(distortion_kwargs, dict):
        raise ValueError("distortion_kwargs must be a dictionary")
    if not isinstance(extrinsic_kwargs, dict):
        raise ValueError("extrinsic_kwargs must be a dictionary")

    # Create the array of points
    world_points = numpy.asarray(world_points, dtype=numpy.float64)

    # Transpose the points if needed
    if transpose:
        world_points = numpy.moveaxis(world_points, 0, -1)  # (3, ...) -> (..., 3)

    # Extract the original shape
    shape = world_points.shape  # (..., 3)

    # Flatten the points along the last axis
    world_points = world_points.reshape(
        -1, shape[-1]
    )  # shape (..., 3) -> shape (n_points, 3)

    # Check the shape of the points
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(
            f"The points must be in the shape (..., 3) or (3, ...) if ``transpose`` is True. Got {shape} instead and transpose is {transpose}."
        )

    # Select the parameters to compute the Jacobian with respect to
    if dp:
        # If dp is True, compute all the jacobians
        dintrinsic = True
        ddistortion = True
        dextrinsic = True
    atleast1dp = dintrinsic or ddistortion or dextrinsic

    # Check if some transformations can be skipped for the Jacobian computation for efficiency
    skip_intrinsic = isinstance(intrinsic, NoIntrinsic)
    skip_distortion = isinstance(distortion, NoDistortion)
    skip_extrinsic = isinstance(extrinsic, NoExtrinsic)

    # Extract the useful constants
    n_points = world_points.shape[0]  # n_points
    n_params = (
        intrinsic.n_params + distortion.n_params + extrinsic.n_params
    )  # Total number of parameters

    # Realize the transformation:
    normalized_points, extrinsic_jacobian_dx, extrinsic_jacobian_dp = (
        extrinsic._transform(world_points, dx=dx, dp=dextrinsic, **extrinsic_kwargs)
    )
    distorted_points, distortion_jacobian_dx, distortion_jacobian_dp = (
        distortion._transform(
            normalized_points,
            dx=(dx or dextrinsic) and not skip_distortion,
            dp=ddistortion,
            **distortion_kwargs,
        )
    )  # (dx is requiered for propagation of dp)
    image_points, intrinsic_jacobian_dx, intrinsic_jacobian_dp = intrinsic._transform(
        distorted_points,
        dx=(dx or dextrinsic or ddistortion) and not skip_intrinsic,
        dp=dintrinsic,
        **intrinsic_kwargs,
    )  # (dx is requiered for propagation of dp)

    # Apply the chain rules to compute the Jacobians with respect to the projection parameters
    if atleast1dp:
        jacobian_flat_dp = numpy.full(
            (n_points, 2, n_params), numpy.nan, dtype=numpy.float64
        )
        # wrt the extrinsic parameters
        if dextrinsic:
            if skip_intrinsic and skip_distortion:
                jacobian_flat_dp[..., intrinsic.n_params + distortion.n_params :] = (
                    extrinsic_jacobian_dp
                )
            elif skip_intrinsic:
                jacobian_flat_dp[..., intrinsic.n_params + distortion.n_params :] = (
                    numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dp)
                )
            elif skip_distortion:
                jacobian_flat_dp[..., intrinsic.n_params + distortion.n_params :] = (
                    numpy.matmul(intrinsic_jacobian_dx, extrinsic_jacobian_dp)
                )
            else:
                jacobian_flat_dp[..., intrinsic.n_params + distortion.n_params :] = (
                    numpy.matmul(
                        intrinsic_jacobian_dx,
                        numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dp),
                    )
                )

        # wrt the distortion parameters
        if ddistortion:
            if skip_intrinsic:
                jacobian_flat_dp[
                    ..., intrinsic.n_params : intrinsic.n_params + distortion.n_params
                ] = distortion_jacobian_dp
            else:
                jacobian_flat_dp[
                    ..., intrinsic.n_params : intrinsic.n_params + distortion.n_params
                ] = numpy.matmul(intrinsic_jacobian_dx, distortion_jacobian_dp)

        # wrt the intrinsic parameters
        if dintrinsic:
            jacobian_flat_dp[..., : intrinsic.n_params] = (
                intrinsic_jacobian_dp  # (intrinsic parameters)
            )

    # Apply the chain rules to compute the Jacobians with respect to the input 3D world points
    if dx:
        if skip_intrinsic and skip_distortion:
            jacobian_flat_dx = extrinsic_jacobian_dx
        elif skip_intrinsic:
            jacobian_flat_dx = numpy.matmul(
                distortion_jacobian_dx, extrinsic_jacobian_dx
            )
        elif skip_distortion:
            jacobian_flat_dx = numpy.matmul(
                intrinsic_jacobian_dx, extrinsic_jacobian_dx
            )
        else:
            jacobian_flat_dx = numpy.matmul(
                intrinsic_jacobian_dx,
                numpy.matmul(distortion_jacobian_dx, extrinsic_jacobian_dx),
            )  # shape (n_points, 2, 3)

    # Reshape the normalized points back to the original shape (Warning shape is (..., 3) and not (..., 2))
    image_points = image_points.reshape(
        (*shape[:-1], 2)
    )  # shape (n_points, 2) -> (..., 2)
    jacobian_dx = (
        jacobian_flat_dx.reshape((*shape[:-1], 2, 3)) if dx else None
    )  # shape (n_points, 2, 3) -> (..., 2, 3)
    jacobian_dp = (
        jacobian_flat_dp.reshape((*shape[:-1], 2, n_params)) if atleast1dp else None
    )  # shape (n_points, 2, n_params) -> (..., 2, n_params)

    # Transpose the points back to the original shape if needed
    if transpose:
        image_points = numpy.moveaxis(image_points, -1, 0)  # (..., 2) -> (2, ...)
        jacobian_dx = (
            numpy.moveaxis(jacobian_dx, -2, 0) if dx else None
        )  # (..., 2, 2) -> (2, ..., 2)
        jacobian_dp = (
            numpy.moveaxis(jacobian_dp, -2, 0) if atleast1dp else None
        )  # (..., 2, n_params) -> (2, ..., n_params)

    # Return the result
    result = TransformResult(
        transformed_points=image_points,
        jacobian_dx=jacobian_dx,
        jacobian_dp=jacobian_dp,
        transpose=transpose,
    )

    # Add the short-hand properties for the jacobians
    result.add_jacobian(
        "dintrinsic",
        0,
        intrinsic.n_params,
        f"Jacobian of the image points with respect to the intrinsic parameters (see {intrinsic.__class__.__name__}) for more details on their order",
    )
    result.add_jacobian(
        "ddistortion",
        intrinsic.n_params,
        intrinsic.n_params + distortion.n_params,
        f"Jacobian of the image points with respect to the distortion parameters (see {distortion.__class__.__name__}) for more details on their order",
    )
    result.add_jacobian(
        "dextrinsic",
        intrinsic.n_params + distortion.n_params,
        n_params,
        f"Jacobian of the image points with respect to the extrinsic parameters (see {extrinsic.__class__.__name__}) for more details on their order",
    )

    # Add the alias for the transformed points
    result.add_alias("image_points")
    return result
