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
import copy
from typing import Optional

from ..core.transform import Transform
from ..core.package import Package

def optimize_parameters(
    transform: Transform,
    input_points: numpy.ndarray,
    output_points: numpy.ndarray,
    guess: Optional[numpy.ndarray] = None,
    *,
    transpose: bool = False,
    max_iter: int = 10,
    eps: float = 1e-8,
    cond_cutoff: float = 1e5,
    reg_factor: float = 0.0,
    precond_jacobi: bool = False,
    verbose: bool = False,
    _skip: bool = False,
    ) -> numpy.ndarray:
    r"""
    Optimize the parameters of the transformation using the given input and output points.

    Estimate the optimized parameters of the transformation such that the transformed input points match the output points.

    Lets consider a set of input points :math:`\vec{X}_I` with shape (..., input_dim) and a set of output points :math:`\vec{X}_O` with shape (..., output_dim).
    We search :math:`\lambda = \lambda_0 + \delta \lambda` such that:

    .. math::

        \vec{X}_O = \text{Transform}(\vec{X}_I, \lambda) = T(\vec{X}_I, \lambda_0 + \delta \lambda)

    .. note::

        The current parameters of the transformation are not directly modified.
    
    We have:

    .. math::

        \nabla_{\lambda} T (\vec{X}_I, \lambda_0) \delta \lambda = \vec{X}_O - T(\vec{X}_I, \lambda_0)

    The corrections are computed using the following equations:

    .. math::

        J^{T} J \delta \lambda = J^{T} R

    Where :math:`J = \nabla_{\lambda} T (\vec{X}_I, \lambda_0)` is the Jacobian matrix of the transformation with respect to the parameters, and :math:`R = \vec{X}_O - T(\vec{X}_I, \lambda_0)` is the residual vector.

    :math:`\lambda_0` is the initial guess for the parameters, if None, the current parameters of the transformation are used. (or a zero vector if the parameters are not set).

    .. note::

        This method can be used to optimize the parameters of any transformation that implements the `_transform` method with ``jacobian_dp`` computation.

    .. note::

        The ``_skip`` parameter is used to skip the checks for the transformation parameters and assume the input and output points are given in the (Npoints, input_dim) and (Npoints, output_dim) float format, respectively.
        Please use this parameter with caution, as it may lead to unexpected results if the transformation parameters are not set correctly.

    For conditioning, the following steps are applied:

    - First, a regularization term is added to the Jacobian matrix to improve stability: :math:`J^{T} J + \text{regfactor} I`.
    - Second, a preconditioner is applied to the Jacobian matrix to improve the conditioning of the problem.
    
    The `cond_cutoff` parameter is used to detect ill-conditioned problems. If the condition number of the Jacobian matrix is greater than this value, a warning is raised and the optimization returns NaN array.

    .. warning::

            The points are converting to float before applying the inverse transformation.
            See :class:`pycvcam.core.Package` for more details on the default data types used in the package.

    Parameters
    ----------
    transform : Transform
        The transformation object to optimize.

    input_points : numpy.ndarray
        The input points to be transformed. Shape (..., input_dim) (or (input_dim, ...) if `transpose` is True).
    
    output_points : numpy.ndarray
        The output points to be matched. Shape (..., output_dim) (or (output_dim, ...) if `transpose` is True).

    guess : Optional[numpy.ndarray], optional
        The initial guess for the parameters of the transformation with shape (Nparams,). If None, the current parameters of the transformation are used (or a zero vector if the parameters are not set). Default is None.

    transpose : bool, optional
        If True, the input and output points are transposed to shape (input_dim, ...) and (output_dim, ...), respectively. Default is False.

    max_iter : int, optional
        The maximum number of iterations for the optimization. Default is 10.

    eps : float, optional
        The convergence threshold for the optimization. Default is 1e-8.

    verbose : bool, optional
        If True, print the optimization progress and diagnostics. Default is False.

    cond_cutoff : float, optional
        The cutoff value for the condition number of the Jacobian matrix. If the condition number is greater than this value, the optimization will be considered unstable and will raise a warning and return NaN array. This is used to detect ill-conditioned problems. Default is 1e5.

    reg_factor : float, optional
        The regularization factor for the optimization. If greater than 0, it adds a tikhonov regularization term to the optimization problem to improve stability :math:`J^{T} J + \text{regfactor} I`. Default is 0.0.

    precond_jacobi : bool, optional
        If True, apply a preconditioner to the Jacobian matrix to improve the conditioning of the problem. This is done by applying the Jacobi preconditioner to the Jacobian matrix before solving the optimization problem. Default is False.

    _skip : bool, optional
        If True, skip the checks for the transformation parameters and assume the input and output points are given in the (Npoints, input_dim) and (Npoints, output_dim) float format, respectively.
        The guess must be given in the (Nparams,) float format.
        `transpose` is ignored if this parameter is set to True.

    Returns
    -------
    numpy.ndarray
        The optimized parameters of the transformation with shape (Nparams,).

    Raises
    ------
    ValueError
        If the input and output points do not have the same number of points, or if the input and output dimensions do not match the transformation's input and output dimensions.

    TypeError
        If the input and output points are not numpy arrays, or if the guess is not a numpy array.

    Examples
    --------

    Lets assume, we want to optimize the parameters or a ZernikeDistortion object to match a set of normalized points to a set of distorted points:

    .. code-block:: python

        import numpy
        from pycvcam import ZernikeDistortion
        from pycvcam.optimize import optimize_parameters

        # Create a ZernikeDistortion object with initial parameters at zero (Nzer = 3 model)
        zernike_distortion = ZernikeDistortion(parameters=numpy.random.rand(20))

        # Generate some random normalized points
        normalized_points = numpy.random.rand(100, 2)  # 100 points in 2D
        distorted_points = zernike_distortion.apply(normalized_points)

        # Optimize the parameters to match the distorted points
        optimized_parameters = optimize_parameters(
            transform=zernike_distortion,
            input_points=normalized_points,
            output_points=distorted_points,
            guess=numpy.zeros_like(zernike_distortion.parameters),  # Initial guess for the parameters
        )

        print("Optimized parameters:", optimized_parameters) # Shape (Nparams,)

    """
    if not isinstance(transform, Transform):
        raise TypeError(f"transform must be an instance of Transform, got {type(transform)}")

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
        if not isinstance(cond_cutoff, float) or cond_cutoff <= 0:
            raise TypeError(f"cond_cutoff must be a positive float, got {cond_cutoff}")
        if not isinstance(reg_factor, float) or reg_factor < 0:
            raise TypeError(f"reg_factor must be a non-negative float, got {reg_factor}")
        if not isinstance(precond_jacobi, bool):
            raise TypeError(f"precond_jacobi must be a boolean, got {type(precond_jacobi)}")

        # Convert input and output points to float
        input_points = numpy.asarray(input_points, dtype=Package.get_float_dtype())
        output_points = numpy.asarray(output_points, dtype=Package.get_float_dtype())

        # Check the shape of the input and output points
        if input_points.ndim < 2 or output_points.ndim < 2:
            raise ValueError(f"Input and output points must have at least 2 dimensions, got {input_points.ndim} and {output_points.ndim} dimensions respectively.")
        
        # Transpose the input and output points if requested
        if transpose:
            input_points = numpy.moveaxis(input_points, 0, -1) # (input_dim, ...) -> (..., input_dim)
            output_points = numpy.moveaxis(output_points, 0, -1) # (output_dim, ...) -> (..., output_dim)

        # Flatten the input and output points to 2D for processing
        input_points = input_points.reshape(-1, transform.input_dim)  # (..., input_dim) -> (Npoints, input_dim)
        output_points = output_points.reshape(-1, transform.output_dim)  # (..., output_dim) -> (Npoints, output_dim)

        # Check the number of points
        if input_points.shape[0] != output_points.shape[0]:
            raise ValueError(f"Input and output points must have the same number of points, got {input_points.shape[0]} and {output_points.shape[0]} points respectively.")
        
        if input_points.shape[0] == 0:
            raise ValueError("Input and output points must have at least one point.")
        
        # Check the last dimension of the input and output points
        if input_points.shape[-1] != transform.input_dim:
            raise ValueError(f"Input points must have {transform.input_dim} dimensions, got {input_points.shape[-1]} dimensions.")
        if output_points.shape[-1] != transform.output_dim:
            raise ValueError(f"Output points must have {transform.output_dim} dimensions, got {output_points.shape[-1]} dimensions.")

        # Check the guess
        if guess is not None:
            guess = numpy.asarray(guess, dtype=Package.get_float_dtype())
            if guess.ndim != 1:
                raise ValueError(f"Guess must be a 1D array, got {guess.ndim} dimensions.")
            if guess.shape[0] != transform.Nparams:
                raise ValueError(f"Guess must have {transform.Nparams} parameters, got {guess.shape[0]} parameters.")
        
        else:
            # Use the current parameters as the guess
            guess = transform.parameters if transform.is_set() else numpy.zeros(transform.Nparams, dtype=Package.get_float_dtype())

    # Return empty arrays if Nparams is 0
    if transform.Nparams == 0:
        return numpy.zeros(0, dtype=Package.get_float_dtype())

    # Create a perfect copy of the current class to avoid modifying the original one
    object_class = copy.deepcopy(transform)
    Npoints = input_points.shape[0]  # Number of points in computation

    # Set the parameters of the object class to the guess
    object_class.parameters = guess
    delta_itk = numpy.zeros_like(object_class.parameters, dtype=Package.get_float_dtype())

    # Run the iterative algorithm
    for it in range(max_iter):

        #============================================
        # Transformation and Jacobian computation
        #============================================

        # Compute the transformed points and the Jacobian with respect to the parameters
        transformed_points_itk, _, jacobian_dp = object_class._transform(input_points, dx=False, dp=True)  # shape (Npoints, output_dim), None, (Npoints, output_dim, Nparams)

        # Check if the jacobian_dp is None$
        if jacobian_dp is None:
            raise ValueError("Jacobian with respect to the parameters is not available. Please implement the _transform method to return the Jacobian with respect to the parameters.")

        # Check the convergence of the optimization
        diff = numpy.linalg.norm(transformed_points_itk - output_points, axis=1)  # shape (Npoints,)
        if verbose:
            print(f"Iteration {it}: |X_O - X_I| - Max difference: {numpy.nanmax(diff)}, Mean difference: {numpy.nanmean(diff)}")

        if numpy.all(diff[~numpy.isnan(diff)] < eps):
            if verbose:
                print(f"Optimization converged in {it} iterations.")
            break
        
        #===================================================
        # Create the residual vector and Jacobian matrix
        #===================================================

        if verbose:
            print("\n#=====================================================")
            print(f"STARTING ITERATION {it+1} OF THE OPTIMIZATION PROCESS")
            print("#=====================================================")

        # Construct the residual vector R and the Jacobian J
        R = output_points - transformed_points_itk  # shape (Npoints, output_dim)
        J = jacobian_dp  # shape (Npoints, output_dim, Nparams)

        # Create masks to filter out invalid points
        mask_R = numpy.isfinite(R).all(axis=1)  # Create a mask for finite values in R
        mask_J = numpy.isfinite(J).all(axis=(1, 2))  # Create a mask for finite values in each row of J
        mask = mask_R & mask_J  # Combine the masks to filter out invalid points

        if verbose:
            print(f"Iteration {it+1}: {numpy.sum(mask)} valid points out of {Npoints}.")

        # Apply the masks to R_flat and J_flat
        R = R[mask, :]  # shape (Nvalid_points, output_dim)
        J = J[mask, :, :]  # shape (Nvalid_points, output_dim, Nparams)

        # Flatten the residual vector and Jacobian matrix
        R_flat = R.flatten()  # Flatten the residual vector to shape (Npoints * output_dim,)
        J_flat = J.reshape(Npoints * transform.output_dim, -1)  # Flatten the Jacobian to shape (Npoints * output_dim, Nparams)

        # Compute the delta using the normal equations: J^T J delta = J^T R
        JTJ = numpy.dot(J_flat.T, J_flat)  # shape (Nparams, Nparams)
        JTR = numpy.dot(J_flat.T, R_flat)  # shape (Nparams,)


        #===================================================
        # Regularization and conditioning part
        #===================================================

        # Display the condition number of the Jacobian matrix without regularization
        if verbose:
            print(f"Iteration {it+1}: Condition number of JTJ before preconditionning and regularization: {numpy.linalg.cond(JTJ)}")

        # Add regularization if requested
        if reg_factor > 0.0:
            JTJ += reg_factor * numpy.eye(transform.Nparams, dtype=Package.get_float_dtype())

            if verbose:
                print(f"Iteration {it+1}: Condition number of JTJ after regularization: {numpy.linalg.cond(JTJ)}")

        # Apply preconditioning if requested
        if precond_jacobi:
            # Compute the diagonal of JTJ for Jacobi preconditioning
            diag_JTJ = numpy.diag(JTJ)
            
            if numpy.any(diag_JTJ == 0):
                raise ValueError("Jacobi preconditioner cannot be applied because the diagonal of JTJ contains zeros.")

            # Apply the Jacobi preconditioner
            JTJ = JTJ / diag_JTJ[:, numpy.newaxis]  # Normalize each row by the diagonal element
            JTR = JTR / diag_JTJ  # Normalize the residual vector by the diagonal elements

            if verbose:
                print(f"Iteration {it+1}: Condition number of JTJ after Jacobi preconditioning: {numpy.linalg.cond(JTJ)}")

        # Display more information if verbose is True.
        if verbose:
            eigvals, eigvecs = numpy.linalg.eig(JTJ)
            print(f"Iteration {it+1}: Eigenvalues of JTJ:\n{eigvals}")

        # ===================================================
        # Condition number check
        # ===================================================

        # Condition number check
        cond_number = numpy.linalg.cond(JTJ)

        if verbose:
            print(f"Iteration {it+1}: Condition number of JTJ: {cond_number}")

        if cond_number > cond_cutoff:
            print(f"Warning: Condition number {cond_number} exceeds cutoff {cond_cutoff}. Optimization may be unstable. skipping iteration {it+1} and returning NaN array.")
            return numpy.full(transform.Nparams, numpy.nan, dtype=Package.get_float_dtype())

        # ====================================================
        # Solve the linear system to find the delta
        # ====================================================

        # Solve the linear system to find the delta
        delta_itk = numpy.linalg.solve(JTJ, JTR) # shape (Nparams,)

        if verbose:
            print(f"Iteration {it+1}: Delta parameters:\n{delta_itk}")
            
        # Update the parameters of the object class
        object_class.parameters = object_class.parameters + delta_itk  # shape (Nparams,)

        if verbose:
            print(f"Iteration {it+1}: Updated parameters:\n{object_class.parameters}")
    
    return object_class.parameters  # shape (Nparams,)