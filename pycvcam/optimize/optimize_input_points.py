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

import numpy
from typing import Optional
import scipy

from ..core.transform import Transform
from ..core.package import Package

def optimize_input_points(
    transform: Transform,
    output_points: numpy.ndarray,
    guess: Optional[numpy.ndarray] = None,
    *,
    transpose: bool = False,
    max_iter: int = 10,
    eps: float = 1e-8,
    verbose: bool = False,
    _skip: bool = False
) -> numpy.ndarray:
    r"""
    Optimize the input points of the transformation using the given output points.

    Estimate the optimized input points of the transformation such that the transformed input points match the output points.

    .. warning::

        This method can only be used if the dimensions are the same, i.e. input_dim == output_dim.

    Lets consider a set of output points :math:`X_O` with shape (..., dim) and a set of input points :math:`\vec{X}_I` with shape (..., input_dim).
    We search :math:`\vec{X}_I = \vec{X}_{I_0} + \delta \vec{X}_I` such that:

    .. math::

        \vec{X}_O = \text{Transform}(\vec{X}_I, \lambda) = T(\vec{X}_{I_0} + \delta \vec{X}_I, \lambda)

    We have:

    .. math::

        \nabla_{X} T (\vec{X}_{I_0}, \lambda) \delta \vec{X}_I = \vec{X}_O - T(\vec{X}_{I_0}, \lambda)

    The corrections are computed using the following equations:

    .. math::

        J \delta \vec{X}_I = R

    Where :math:`J = \nabla_{X} T (\vec{X}_{I_0}, \lambda)` is the Jacobian matrix of the transformation with respect to the input points, and :math:`R = \vec{X}_O - T(\vec{X}_{I_0}, \lambda)` is the residual vector.

    :math:`\vec{X}_{I_0}` is the initial guess for the input points, if None, it use the output points as the initial guess.

    .. note::

        The ``_skip`` parameter is used to skip the checks for the transformation parameters and assume the output points are given in the (Npoints, dim) float format.
        Please use this parameter with caution, as it may lead to unexpected results if the transformation parameters are not set correctly.

    .. warning::

            The points are converting to float before applying the inverse transformation.
            See :class:`pycvcam.core.Package` for more details on the default data types used in the package.

    Parameters
    ----------
    transform : Transform
        The transformation object to optimize.

    output_points : numpy.ndarray
        The output points to be matched. Shape (..., dim) (or (dim, ...) if `transpose` is True).

    guess : Optional[numpy.ndarray], optional
        The initial guess for the input points of the transformation with shape (..., dim). If None, the output points are used as the initial guess. Default is None.

    transpose : bool, optional
        If True, the output points are transposed to shape (dim, ...). Default is False.

    max_iter : int, optional
        The maximum number of iterations for the optimization. Default is 10.

    eps : float, optional
        The convergence threshold for the optimization. Default is 1e-8.

    verbose : bool, optional
        If True, print the optimization progress and diagnostics. Default is False.

    _skip : bool, optional
        If True, skip the checks for the transformation parameters and assume the output points are given in the (Npoints, dim) float format.
        The guess must be given in the (Npoints, dim) float format.
        `transpose` is ignored if this parameter is set to True.

    Returns
    -------
    numpy.ndarray
        The optimized input points of the transformation with shape (..., dim).

    Raises
    ------
    ValueError
        If the output points do not have the expected shape, or if the input and output dimensions do not match the transformation's input and output dimensions.

    TypeError
        If the output points or guess are not numpy arrays, or if the guess is not a numpy array.

    Examples
    --------

    Lets assume, we want to optimize the input points of a Cv2Distortion object to match a set of distorted points:

    .. code-block:: python

        import numpy
        from pycvcam import Cv2Distortion
        from pycvcam.optimize import optimize_input_points

        # Create a Cv2Distortion object with known parameters
        distortion = Cv2Distortion(parameters=numpy.array([1e-3, 2e-3, 1e-3, 1e-4, 2e-3]), Nparams=5)

        # Generate some random distorted points
        distorted_points = numpy.random.rand(10, 2)  # Random 2D points

        # Optimize the input points to match the distorted points
        optimized_input_points = optimize_input_points(distortion, distorted_points) # shape (10, 2)
        print("Optimized Input Points:", optimized_input_points)

    """
    if not isinstance(transform, Transform):
        raise TypeError(f"transform must be an instance of Transform, got {type(transform)}")
    
    if transform.input_dim != transform.output_dim:
        raise ValueError(f"Input dimension ({transform.input_dim}) must be equal to output dimension ({transform.output_dim}) for this method to work.")
    dim = transform.input_dim  # Since input_dim == output_dim

    if not _skip:
        # Check the boolean flags
        if not isinstance(transpose, bool):
            raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise TypeError(f"max_iter must be an integer greater than 0, got {max_iter}")
        if not isinstance(eps, float) or eps <= 0:
            raise TypeError(f"eps must be a positive float, got {eps}")
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose must be a boolean, got {type(verbose)}")

        # Check if the transformation is set
        if not transform.is_set():
            raise ValueError("Transformation parameters are not set. Please set the parameters before optimizing.")

        # Convert output points to float
        output_points = numpy.asarray(output_points, dtype=Package.get_float_dtype())

        # Check the guess
        if guess is not None:
            guess = numpy.asarray(guess, dtype=Package.get_float_dtype())
        else:
            # Use the output points as the initial guess
            guess = numpy.zeros((output_points.shape[0], dim), dtype=Package.get_float_dtype())

        # Check the shape of the output points
        if output_points.ndim < 2:
            raise ValueError(f"Output points must have at least 2 dimensions, got {output_points.ndim} dimensions.")
        if guess.ndim < 2:
                raise ValueError(f"Guess must have at least 2 dimensions, got {guess.ndim} dimensions.")
        
        # Transpose the output points if requested
        if transpose:
            output_points = numpy.moveaxis(output_points, 0, -1) # (dim, ...) -> (..., dim)
            guess = numpy.moveaxis(guess, 0, -1) # (dim, ...) -> (..., dim)

        # Flatten the output points to 2D for processing
        shape = output_points.shape  # (..., dim)
        output_points = output_points.reshape(-1, dim)  # (..., dim) -> (Npoints, dim)
        guess = guess.reshape(-1, dim)  # (..., dim) -> (Npoints, dim)
        
        # Check the number of points
        if output_points.shape[0] != guess.shape[0]:
            raise ValueError(f"Output points and guess must have the same number of points, got {output_points.shape[0]} and {guess.shape[0]} points respectively.")
        if output_points.shape[0] == 0:
            raise ValueError("Output points and guess must have at least one point.")

        if output_points.shape[-1] != dim:
            raise ValueError(f"Output points must have {dim} dimensions, got {output_points.shape[-1]} dimensions.")
        if guess.shape[-1] != dim:
            raise ValueError(f"Guess must have {dim} dimensions, got {guess.shape[-1]} dimensions.")
        
    # Initialize the guess for the input points
    Npoints = output_points.shape[0]
    delta_itk = numpy.zeros_like(guess, dtype=Package.get_float_dtype()) # shape (Npoints, dim) (Delta for the next iteration)
    Nopt = Npoints # Number of points in computation

    # Prepare the output array:
    input_points = guess

    # Create the mask for the points in computation
    mask = numpy.logical_and(numpy.isfinite(output_points).all(axis=1), numpy.isfinite(input_points).all(axis=1))  # shape (Npoints,)

    # Run the iterative algorithm
    for it in range(max_iter):
        # Compute the transformation of the input points and the Jacobian with respect to the input points
        output_points_itk, jacobian_dx, _ = transform._transform(input_points[mask, :], dx=True, dp=False) # shape (Nopt, dim), (Nopt, dim, dim), None

        # Check if the jacobian_dx is None
        if jacobian_dx is None:
            raise ValueError("Jacobian with respect to the input points is not available. Please implement the _transform method to return the Jacobian with respect to the input points.")
        
        # Check the convergence of the optimization
        diff = numpy.linalg.norm(output_points_itk - output_points[mask, :], axis=1)  # shape (Nopt,)
        eps_mask = diff > eps # shape (Nopt,)
        mask[mask] = numpy.logical_and(mask[mask], eps_mask)

        if numpy.sum(mask) == 0:
            if verbose:
                print(f"Optimization converged in {it} iterations.")
            break

        Nopt = numpy.sum(mask)  # Update the number of points in computation

        output_points_itk = output_points_itk[eps_mask, :]  # shape (NewNopt, dim)
        jacobian_dx = jacobian_dx[eps_mask, :, :]  # shape (NewNopt, dim, dim)

        # Construct the residual vector R and the Jacobian J
        R = output_points[mask, :] - output_points_itk # shape (Nopt, dim)
        J = jacobian_dx # shape (Nopt, dim, dim)

        # Solve the linear system to find the delta
        delta_itk = numpy.array([scipy.linalg.solve(J[i], R[i]) for i in range(Nopt)], dtype=Package.get_float_dtype()) # shape (Nopt, dim)

        # Update the input points
        input_points[mask, :] = input_points[mask, :] + delta_itk
        if verbose:
            print(f"Iteration {it+1}: {numpy.sum(mask)} valid points out of {Npoints}. Max delta: {numpy.max(numpy.abs(delta_itk))}")

    # Return the optimized input points
    if not _skip:
        input_points = input_points.reshape(*shape[:-1], dim)  # (Npoints, dim) -> (..., dim)

        if transpose:
            input_points = numpy.moveaxis(input_points, -1, 0) # (..., dim) -> (dim, ...)

    return input_points

