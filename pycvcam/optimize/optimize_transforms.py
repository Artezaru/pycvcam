from typing import Optional, Tuple, Callable, Dict, Sequence
from numpy.typing import ArrayLike
from numbers import Real, Integral

import numpy
import time
import copy
import scipy

from ..core.transform import Transform
from ..core.transform_composition import TransformComposition
from ..core.intrinsic import Intrinsic
from ..core.distortion import Distortion
from ..core.extrinsic import Extrinsic
from ..intrinsic_objects.no_intrinsic import NoIntrinsic
from ..distortion_objects.no_distortion import NoDistortion
from ..extrinsic_objects.no_extrinsic import NoExtrinsic


def _study_jacobian_least_squares(
    residual: numpy.ndarray,
    jacobian: numpy.ndarray,
    parameters: numpy.ndarray,
    _pretext: Optional[str] = None,
    _start: bool = True,
    _sparse: bool = False,
) -> None:
    """
    Study the Jacobian matrix of the least squares problem to analyze the observability

    Print:
    - The shape and density of the Jacobian matrix
    - The singular values and the condition number of the Jacobian matrix
    - The directions of the parameters
    - The estimated variances of the parameters based on the covariance matrix approximation

    Parameters
    ----------
    residual : numpy.ndarray
        The residual vector of the least squares problem with shape (n_points*output_dim,).

    jacobian : numpy.ndarray
        The Jacobian matrix of the least squares problem with shape (n_points*output_dim, n_params).

    parameters : numpy.ndarray
        The current parameters of the transformation with shape (n_params,).

    _pretext : Optional[str], optional
        A pretext to display before the analysis of the Jacobian matrix.
        Default is None, which means no pretext is displayed.

    _start : bool, optional
        True = Iter 0 before the optimization, False = Iter N after the optimization.

    _sparse : bool, optional
        If True, the Jacobian is treated as a sparse matrix for the analysis.
        Default is False.

    """
    if _start:
        print("\n" + "=" * 50)
        print("\n" + "-" * 50)
        print(f"Initial Jacobian (Iter 0) analysis of the least squares problem")
        print("-" * 50 + "\n")
    else:
        print("\n" + "-" * 50)
        print(f"Jacobian analysis of the least squares problem (End of optimization)")
        print("-" * 50 + "\n")

    if _start and _pretext is not None:
        print(_pretext + "\n")

    m, n = jacobian.shape
    if _start:
        print(f"Jacobian shape: {m} x {n} (equations x parameters)")
        if m < n:
            print(
                f"Warning: Underdetermined system (more parameters than residuals), "
                f"the optimization may not converge to a unique solution."
            )
        if m == n:
            print(
                f"Warning: Square system (same number of parameters and residuals), "
                f"the optimization may be sensitive to noise and may not converge."
            )
        if m > n:
            print(
                f"Overdetermined system (more residuals than parameters), "
                f"the optimization is likely to converge to a unique solution."
            )
        if n == 0:
            print("No parameters to optimize.")
            return

    if _start and not _sparse:
        density = numpy.count_nonzero(jacobian) / (m * n)
        print(f"Density: {density*100:.2f}%")
    if _start and _sparse:
        density = scipy.sparse.csr_matrix(jacobian).count_nonzero() / (m * n)
        print(f"Density (sparse): {density*100:.2f}%")

    # SVD
    if not _sparse:
        U, S, Vt = numpy.linalg.svd(jacobian, full_matrices=False)
        sigma_max = S[0]
        sigma_min = S[-1]
        cond_number = sigma_max / sigma_min
        print(f"Singular values (max/min): {sigma_max:.3e} / {sigma_min:.3e}")
        print(f"Condition number: {cond_number:.3e}")
    else:
        U, S, Vt = scipy.sparse.linalg.svds(jacobian, k=min(m, n) - 1)
        sigma_max = S[-1]
        sigma_min = S[0]
        cond_number = sigma_max / sigma_min
        print(f"Singular values (max/min): {sigma_max:.3e} / {sigma_min:.3e}")
        print(f"Condition number: {cond_number:.3e}")

    # Variance contribution of each singular value (1/sigma^2)
    print("\nSingular values and their contribution to the variance:")
    header = f"| {'Index':^10} | {'Singular Value λ':^18} | {'Var = 1/λ^2':^20} |"
    print(header)
    for i, sigma in enumerate(S):
        var_contribution = 1 / (sigma**2) if sigma > 1e-12 else numpy.inf
        print(f"| {i:^10} | {sigma:^18.3e} | {var_contribution:^20.3e} |")

    # Display the directions of the 3 more quasi linearly dependent directions (smallest singular values)
    if n >= 3:
        null_indices = [0, n - 3, n - 2, n - 1]
    elif n == 3:
        null_indices = [n - 3, n - 2, n - 1]
    elif n == 2:
        null_indices = [n - 2, n - 1]
    else:
        null_indices = [n - 1]

    print(f"\nHigher and Smallers singular values directions vectors (V^T rows):")
    header = f"| {'Parameter':^10} | "
    for i in null_indices:
        header += f"{'Vt['+str(i)+']':^15} | "
    print(header)
    for j in range(n):
        row = f"| {j:^10} | "
        for i in null_indices:
            row += f"{Vt[i, j]:^15.3e} | "
        print(row)

    # Parameters Covariances
    cost = 0.5 * numpy.sum(residual**2)
    sigma2 = 2 * cost / (m - n) if m > n else numpy.inf
    if not _sparse:
        cov = sigma2 * numpy.linalg.inv(jacobian.T @ jacobian)
    else:
        cov = sigma2 * scipy.sparse.linalg.inv(jacobian.T @ jacobian).toarray()
    print("\nEstimated variances of the parameters:")
    header = f"| {'Parameter':^10} | {'Value P':^15} | {'Var = σ^2 (J.T J)^-1':^20} | {'Ratio √V/|P|':^15} |"
    print(header)
    for i in range(n):
        var = cov[i, i]
        rel_sqrt = (
            (numpy.sqrt(var) / abs(parameters[i])) * 100
            if abs(parameters[i]) > 1e-12
            else numpy.inf
        )
        if rel_sqrt > 1000:
            rel_sqrt_str = f"> 1000 %"
        else:
            rel_sqrt_str = f"{rel_sqrt:.3f} %"

        row = (
            f"| {i:^10} | {parameters[i]:^15.3e} | {var:^20.3e} | {rel_sqrt_str:^15} |"
        )
        print(row)
    if _start:
        print("\n" + "-" * 50)
        print("Optimization in progress...")
        print("-" * 50 + "\n")
    if not _start:
        print("\n" + "=" * 50 + "\n")


def _build_callback_least_squares(
    seq_transforms: Sequence[Transform],
    seq_masks: Sequence[numpy.ndarray],
    seq_guesses: Sequence[numpy.ndarray],
    max_iterations: int,
    max_time: int,
    return_history: bool,
) -> Tuple[Callable, Callable]:
    r"""
    Build a callback function for the least squares optimization to track the
    optimization process and implement stopping criteria for scipy.optimize.least_squares.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of Transform objects to be optimized. The parameters of these
        transformations will be updated during the optimization process.

    seq_masks : Sequence[numpy.ndarray]
        A sequence of mask arrays for each chain, where each element is a numpy array of
        shape (n_params,) containing boolean values indicating which parameters of the
        transformations in that chain are being optimized (True) and which are kept
        fixed (False).

    seq_guesses : Sequence[numpy.ndarray]
        A sequence of initial guess arrays for each transformation, where each element is
        a numpy array of shape (n_params,) containing the initial guess for the
        parameters of that transformation.

    max_iterations : int
        Stop criterion by the number of iterations. The optimization process is stopped
        when the number of iterations exceeds ``max_iterations``.

    max_time : int
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds.

    return_history : bool
        If True, the callback function will store the history of the optimization process
        in a list. Each entry in the list is a tuple containing the current parameters and
        the intermediate result from scipy.optimize.least_squares. The history can be
        retrieved using the get_history function returned by this function.

    Returns
    -------
    callback : Callable
        The callback function to be passed to scipy.optimize.least_squares. This function
        will be called at each iteration of the optimization process with the current
        intermediate result from scipy.optimize.least_squares.

    get_history : Callable
        A function that returns the history of the optimization process if return_history
        is True, or None if return_history is False.

    """
    n_transforms = len(seq_transforms)

    history = []
    count_call = 0
    start_time = time.time()

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        nonlocal history, start_time, count_call

        if return_history:
            parameters = []
            index = 0
            for i in range(n_transforms):
                p = seq_guesses[i].copy()
                p[seq_masks[i]] = intermediate_result.x[
                    index : index + numpy.sum(seq_masks[i])
                ]
                parameters.append(p)
                index += numpy.sum(seq_masks[i])

            parameters = tuple(parameters)  # len n_transforms, each shape (n_params,)
            history.append(parameters)

        if max_iterations is not None and count_call > max_iterations:
            raise StopIteration(
                f"Maximum number of iterations {max_iterations} exceeded"
            )
        if max_time is not None and (time.time() - start_time) > max_time:
            raise StopIteration(f"Maximum time of {max_time} seconds exceeded")

        count_call += 1

    def get_history() -> Optional[list]:
        nonlocal history
        return history if return_history else None

    return callback, get_history


def _build_residual_jacobian_functions(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[int]],
    seq_inputs: Sequence[numpy.ndarray],
    seq_outputs: Sequence[numpy.ndarray],
    seq_masks: Sequence[numpy.ndarray],
    seq_guesses: Sequence[numpy.ndarray],
    seq_transform_kwargs: Sequence[Dict],
    filter_nans: bool,
    _sparse: bool = False,
    _save_length: int = 1,
) -> Tuple[Callable, Callable]:
    r"""
    Build the residual and Jacobian functions for the least squares optimization.
    The residual function computes the residuals between the transformed input points
    and the output points for each chain, while respecting the structure of the chains
    and the masks for the parameters.

    The Jacobian function computes the Jacobian matrix of the residuals with respect to the
    parameters, while respecting the structure of the chains and the masks for the parameters.
    The Jacobian matrix is assembled according to the chain structure and the mask,
    and can be treated as a sparse matrix if _sparse is True.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of Transform objects to be optimized. The parameters of these
        transformations will be updated during the optimization process.

    seq_chains : Sequence[Sequence[int]]
        A sequence of chains, where each chain is a sequence of indices corresponding to
        the transformations in seq_transforms that are part of that chain. The order of
        the indices in each chain defines the order of the transformations in the chain.

    seq_inputs : Sequence[numpy.ndarray]
        A sequence of input points for each chain, where each element is a numpy array
        of shape (n_points, input_dim) containing the input points for that chain.

    seq_outputs : Sequence[numpy.ndarray]
        A sequence of output points for each chain, where each element is a numpy array
        of shape (n_points, output_dim) containing the output points for that chain.

    seq_masks : Sequence[numpy.ndarray]
        A sequence of mask arrays for each chain, where each element is a numpy array of
        shape (n_params,) containing boolean values indicating which parameters of the
        transformations in that chain are being optimized (True) and which are kept
        fixed (False).

    seq_guesses : Sequence[numpy.ndarray]
        A sequence of initial guess arrays for each transformation, where each element is
        a numpy array of shape (n_params,) containing the initial guess for the
        parameters of that transformation.

    seq_transform_kwargs : Sequence[Dict]
        A sequence of dictionaries containing additional keyword arguments to be passed to
        the _transform method of each transformation during the computation of the
        residuals and Jacobian. Each element in the sequence corresponds to a transformation
        in seq_transforms, and the dictionary contains the keyword arguments for that transformation.

    filter_nans : bool
        If True, the function will filter out any NaN or infinite values in the residuals and
        Jacobian by setting them to zero. This can help to prevent the optimization from
        diverging due to invalid values, but it may also affect the convergence of the
        optimization if there are many NaN or infinite values.

    _sparse : bool, optional
        If True, the Jacobian is treated as a sparse matrix for the computation and assembly.
        Default is False.

    _save_length : int, optional
        The number of previous parameter sets, residuals, and Jacobians to save for quick retrieval
        if the same parameters are encountered again. This can help to speed up the optimization
        if the optimization process encounters the same parameters multiple times, but it also
        increases the memory usage. Default is 1, which means only the last set of parameters, residuals, and Jacobian is saved.

    Returns
    -------
    residuals_func : Callable
        A function that takes a parameter vector as input and returns the residuals vector.

    jacobian_func : Callable
        A function that takes a parameter vector as input and returns the Jacobian matrix.

    """
    seq_last_params = [None] * _save_length
    seq_last_res = [None] * _save_length
    seq_last_jac = [None] * _save_length

    n_transforms = len(seq_transforms)
    n_chains = len(seq_chains)

    T_n_reduced_parameters = [numpy.sum(m) for m in seq_masks]  # Nt
    T_start_rp = numpy.cumsum([0] + T_n_reduced_parameters[:-1])  # Nt
    T_end_rp = numpy.cumsum(T_n_reduced_parameters)  # Nt

    C_n_points = [p.shape[0] for p in seq_inputs]  # Nc
    C_output_dim = [p.shape[1] for p in seq_outputs]  # Nc
    C_n_equations = [n * o for n, o in zip(C_n_points, C_output_dim)]  # Nc
    C_mask = [numpy.concatenate([seq_masks[i] for i in c]) for c in seq_chains]  # Nc
    C_start_eq = numpy.cumsum([0] + C_n_equations[:-1])  # Nc
    C_end_eq = numpy.cumsum(C_n_equations)  # Nc
    C_local_offsets = [
        numpy.cumsum([0] + [T_n_reduced_parameters[it] for it in c[:-1]])
        for c in seq_chains
    ]

    C_tc = [TransformComposition([seq_transforms[it] for it in c]) for c in seq_chains]
    C_kwargs = [[seq_transform_kwargs[it] for it in c] for c in seq_chains]

    def compute_func(params: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        nonlocal seq_last_params, seq_last_res, seq_last_jac

        # Return the last computed residuals and Jacobian if already precomputed
        for i in range(_save_length):
            if seq_last_params[i] is not None and numpy.array_equal(
                params, seq_last_params[i]
            ):
                return seq_last_res[i], seq_last_jac[i]

        # Update the parameters of the transformations
        for it, transform in enumerate(seq_transforms):
            p = seq_guesses[it].copy()
            p[seq_masks[it]] = params[T_start_rp[it] : T_end_rp[it]]
            if transform.parameters is not None:
                transform.parameters = p

        # Compute the residuals and Jacobians for each chain
        R_full = numpy.empty((sum(C_n_equations),), dtype=numpy.float64)

        if _sparse:
            J_full = scipy.sparse.lil_matrix(
                (sum(C_n_equations), sum(T_n_reduced_parameters))
            )
        else:
            J_full = numpy.zeros((sum(C_n_equations), sum(T_n_reduced_parameters)))

        for ic, chain in enumerate(seq_chains):
            tc = C_tc[ic]
            list_kwargs = C_kwargs[ic]
            transformed_points, _, jacobian_dp = tc._transform(
                seq_inputs[ic], dx=False, dp=True, list_kwargs=list_kwargs
            )  # shape (n_points, output_dim) and (n_points, output_dim, n_parameters)

            R = seq_outputs[ic] - transformed_points  # shape (n_points, output_dim)
            R_full[C_start_eq[ic] : C_end_eq[ic]] = (
                R.ravel()
            )  # shape (n_points * output_dim,)

            jacobian_dp = -jacobian_dp[
                :, :, C_mask[ic]
            ]  # shape (..., n_reduced_parameters)
            jacobian_dp = jacobian_dp.reshape(
                C_n_points[ic] * C_output_dim[ic], -1
            )  # shape (n_points * output_dim, n_reduced_parameters)

            local_offsets = C_local_offsets[ic]
            for index, it in enumerate(chain):
                local_start = local_offsets[index]
                local_end = local_start + T_n_reduced_parameters[it]

                J_full[
                    C_start_eq[ic] : C_end_eq[ic], T_start_rp[it] : T_end_rp[it]
                ] += jacobian_dp[:, local_start:local_end]

        # Filter NaN or infinite values in the residuals and Jacobian by setting them to zero

        if filter_nans:
            filter_mask_R = ~numpy.isfinite(R_full)
            if not _sparse:
                filter_mask_J = ~numpy.isfinite(J_full).any(axis=1)
            if _sparse:
                filter_mask_J = ~numpy.isfinite(J_full.toarray()).any(axis=1)

            filter_mask = filter_mask_R | filter_mask_J
            if numpy.all(filter_mask):
                raise ValueError(
                    "All residuals are NaN or infinite, filtering set empty matrix, optimization cannot proceed."
                )

            R_full[filter_mask] = 0.0
            J_full[filter_mask, :] = 0.0
            print(
                f"Warning: Filtered {numpy.sum(filter_mask)} out of {R_full.size} equations due to NaN or infinite values in the residuals or Jacobian."
            )

        R = R_full
        if _sparse:
            J = J_full.tocsr()  # Convert to CSR for efficient arithmetic operations
        else:
            J = J_full  # Already a dense array

        # Save the computed residuals and Jacobian for future calls
        seq_last_params = [params.copy()] + seq_last_params[:-1]
        seq_last_res = [R.copy()] + seq_last_res[:-1]
        seq_last_jac = [J.copy()] + seq_last_jac[:-1]
        return R, J

    def residuals_func(params: numpy.ndarray) -> numpy.ndarray:
        res, _ = compute_func(params)
        return res

    def jacobian_func(params: numpy.ndarray) -> numpy.ndarray:
        _, jac = compute_func(params)
        return jac

    return residuals_func, jacobian_func


def _solve_optimize_chains_trf_scipy(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[int]],
    seq_inputs: Sequence[numpy.ndarray],
    seq_outputs: Sequence[numpy.ndarray],
    seq_masks: Sequence[numpy.ndarray],
    seq_guesses: Sequence[numpy.ndarray],
    seq_bounds: Sequence[Tuple[numpy.ndarray, numpy.ndarray]],
    seq_scales: Sequence[numpy.ndarray],
    seq_transform_kwargs: Sequence[Dict],
    max_iterations: Optional[int],
    max_time: Optional[int],
    ftol: Optional[Real],
    xtol: Optional[Real],
    gtol: Optional[Real],
    loss: Optional[str],
    filter_nans: bool,
    verbose_level: int,
    return_result: bool,
    return_history: bool,
    _pretext: Optional[str] = None,
    _sparse: bool = False,
) -> Tuple[numpy.ndarray]:
    r"""
    Optimize the parameters of a set of transformations organized in chains using
    the least squares optimization method from Scipy.

    Trust Region Reflective algorithm is used, which allows to handle bounds and
    scaling of the parameters, and provides a robust optimization process for
    non-linear problems.
    The optimization is performed by minimizing the residuals between the
    transformed input points and the output points for each chain,
    while respecting the structure of the chains and the masks for the parameters.

    This function is used internally by the `optimize_parameters_least_squares` and
    `optimize_chain_parameters_least_squares` functions to perform the optimization of
    the parameters of the transformations in a single step, allowing to optimize
    multiple transformations and chains simultaneously.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of Transform objects to be optimized. The parameters of these
        transformations will be updated during the optimization process.

    seq_chains : Sequence[Sequence[int]]
        A sequence of chains, where each chain is a sequence of indices corresponding to
        the transformations in seq_transforms that are part of that chain. The order of
        the indices in each chain defines the order of the transformations in the chain.

    seq_inputs : Sequence[numpy.ndarray]
        A sequence of input points for each chain, where each element is a numpy array
        of shape (n_points, input_dim) containing the input points for that chain.

    seq_outputs : Sequence[numpy.ndarray]
        A sequence of output points for each chain, where each element is a numpy array
        of shape (n_points, output_dim) containing the output points for that chain.

    seq_masks : Sequence[numpy.ndarray]
        A sequence of mask arrays for each chain, where each element is a numpy array of
        shape (n_params,) containing boolean values indicating which parameters of the
        transformations in that chain are being optimized (True) and which are kept
        fixed (False).

    seq_guesses : Sequence[numpy.ndarray]
        A sequence of initial guess arrays for each transformation, where each element is
        a numpy array of shape (n_params,) containing the initial guess for the
        parameters of that transformation.

    seq_bounds : Sequence[Tuple[numpy.ndarray, numpy.ndarray]]
        A sequence of tuples containing the lower and upper bounds for the parameters of
        each transformation, where each tuple contains two numpy arrays of shape (n_params,)
        representing the lower and upper bounds for the parameters of that transformation.

    seq_scales : Sequence[numpy.ndarray]
        A sequence of scale arrays for each transformation, where each element is a numpy
        array of shape (n_params,) containing the scale for the parameters of that
        transformation. The scale is used to normalize the parameters during the optimization
        process, which can help to improve the convergence of the optimization.

    seq_transform_kwargs : Sequence[Dict]
        A sequence of dictionaries containing additional keyword arguments to be passed to
        the _transform method of each transformation during the computation of the
        residuals and Jacobian. Each element in the sequence corresponds to a transformation
        in seq_transforms, and the dictionary contains the keyword arguments for that transformation.

    max_iterations : Optional[int]
        The maximum number of iterations for the optimization process. If None, there is
        no limit on the number of iterations.

    max_time : Optional[int]
        The maximum time in seconds for the optimization process. If None, there is no
        time limit for the optimization.

    ftol : Optional[Real]
        The tolerance for the cost function value to declare convergence. The optimization
        will stop when the cost function value changes less than ftol between iterations.

    xtol : Optional[Real]
        The tolerance for the parameters to declare convergence. The optimization will stop
        when the change in the parameters is less than xtol between iterations.

    gtol : Optional[Real]
        The tolerance for the gradient to declare convergence. The optimization will stop
        when the norm of the gradient is less than gtol.

    loss : Optional[str]
        The loss function to be used for robust optimization. If None, the standard least
        squares loss is used. Possible values include 'linear', 'soft_l1', 'huber', 'cauchy',
        and 'arctan'.

    filter_nans : bool
        If True, the function will filter out any NaN or infinite values in the residuals and
        Jacobian by setting them to zero. This can help to prevent the optimization from
        diverging due to invalid values, but it may also affect the convergence of the
        optimization if there are many NaN or infinite values.

    verbose_level : int
        The level of verbosity for the optimization process. Higher values will print more
        detailed information about the optimization process, including the cost function value,
        parameter values, and convergence status at each iteration.

    return_result : bool
        If True, the function will return the full result object from scipy.optimize.least_squares,
        which includes information about the optimization process, such as the optimized parameters,
        cost function value, number of iterations, and convergence status. If False, only the
        optimized parameters will be returned.

    return_history : bool
        If True, the function will return a history of the optimization process, which
        is a list of tuples containing the parameters and intermediate results at each
        iteration. If False, no history will be returned.

    _pretext : Optional[str], optional
        A pretext to display before the optimization process starts. This can be used to provide
        context or information about the optimization process that is about to begin.
        Default is None, which means no pretext is displayed.

    _sparse : bool, optional
        If True, the Jacobian matrix will be treated as a sparse matrix for the optimization process.
        This can be beneficial for large problems where the Jacobian is sparse, as it can reduce
        memory usage and improve computational efficiency. Default is False, which means the Jacobian
        will be treated as a dense matrix.

    Returns
    -------
    Tuple[numpy.ndarray]
        The optimized parameters of the transformations, where each element in the tuple
        corresponds to a transformation in seq_transforms, and is a numpy array of
        shape (n_params,) containing the optimized parameters for that transformation.

    scipy.optimize.OptimizeResult (optional)
        If return_result is True, the full result object from scipy.optimize.least_squares is also returned,
        which includes information about the optimization process, such as the optimized parameters,
        cost function value, number of iterations, and convergence status.

    Sequence[Tuple[numpy.ndarray]] (optional)
        If return_history is True, a history of the optimization process is also returned, which is
        a list of tuples containing the parameters and intermediate results at each iteration.

    """
    n_transforms = len(seq_transforms)
    n_chains = len(seq_chains)

    if n_transforms == 0:
        raise ValueError("No transformations to optimize.")
    if n_chains == 0:
        raise ValueError("No chains to optimize.")

    params_initial = numpy.concatenate(
        [seq_guesses[i][seq_masks[i]] for i in range(n_transforms)]
    )  # shape (sum(n_reduced_parameters),)
    params_bounds = numpy.concatenate(
        [seq_bounds[i][:, seq_masks[i]] for i in range(n_transforms)], axis=1
    )  # shape (2, sum(n_reduced_parameters))
    params_scales = numpy.concatenate(
        [seq_scales[i][seq_masks[i]] for i in range(n_transforms)]
    )  # shape (sum(n_reduced_parameters),)

    f, jac = _build_residual_jacobian_functions(
        seq_transforms,
        seq_chains,
        seq_inputs,
        seq_outputs,
        seq_masks,
        seq_guesses,
        seq_transform_kwargs,
        filter_nans,
        _sparse=_sparse,
    )

    callback, get_history = _build_callback_least_squares(
        seq_transforms=seq_transforms,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        return_history=return_history,
        max_iterations=max_iterations,
        max_time=max_time,
    )

    if verbose_level >= 3:
        _study_jacobian_least_squares(
            f(params_initial),
            jac(params_initial),
            params_initial,
            _pretext,
            _start=True,
            _sparse=_sparse,
        )

    # Run the least squares optimization
    result = scipy.optimize.least_squares(
        fun=f,
        x0=params_initial,
        jac=jac,
        bounds=params_bounds,
        x_scale=params_scales,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        verbose=min(verbose_level, 2),
        method="trf",  # Trust Region Reflective algorithm
        loss=loss,
        callback=callback,
    )

    if verbose_level >= 3:
        _study_jacobian_least_squares(
            f(result.x),
            jac(result.x),
            result.x,
            _pretext,
            _start=False,
            _sparse=_sparse,
        )

    parameters = []
    index = 0
    for i in range(n_transforms):
        p = seq_guesses[i].copy()
        p[seq_masks[i]] = result.x[index : index + numpy.sum(seq_masks[i])]
        parameters.append(p)
        index += numpy.sum(seq_masks[i])
    parameters = tuple(parameters)  # len n_transforms, each shape (n_params,)

    if return_result and return_history:
        return parameters, result, get_history()
    elif return_result:
        return parameters, result
    elif return_history:
        return parameters, get_history()
    else:
        return parameters


def _solve_optimize_chains_lm_scipy(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[int]],
    seq_inputs: Sequence[numpy.ndarray],
    seq_outputs: Sequence[numpy.ndarray],
    seq_masks: Sequence[numpy.ndarray],
    seq_guesses: Sequence[numpy.ndarray],
    seq_transform_kwargs: Sequence[Dict],
    max_iterations: Optional[int],
    max_time: Optional[int],
    ftol: Optional[Real],
    xtol: Optional[Real],
    gtol: Optional[Real],
    loss: Optional[str],
    filter_nans: bool,
    verbose_level: int,
    return_result: bool,
    return_history: bool,
    _pretext: Optional[str] = None,
    _sparse: bool = False,
) -> Tuple[numpy.ndarray]:
    r"""
    Optimize the parameters of a set of transformations organized in chains using
    the least squares optimization method from Scipy.

    Levenberg-Marquardt algorithm is used, which is a popular optimization algorithm
    for non-linear least squares problems without bounds.

    The optimization is performed by minimizing the residuals between the
    transformed input points and the output points for each chain,
    while respecting the structure of the chains and the masks for the parameters.

    This function is used internally by the `optimize_parameters_least_squares` and
    `optimize_chain_parameters_least_squares` functions to perform the optimization of
    the parameters of the transformations in a single step, allowing to optimize
    multiple transformations and chains simultaneously.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of Transform objects to be optimized. The parameters of these
        transformations will be updated during the optimization process.

    seq_chains : Sequence[Sequence[int]]
        A sequence of chains, where each chain is a sequence of indices corresponding to
        the transformations in seq_transforms that are part of that chain. The order of
        the indices in each chain defines the order of the transformations in the chain.

    seq_inputs : Sequence[numpy.ndarray]
        A sequence of input points for each chain, where each element is a numpy array
        of shape (n_points, input_dim) containing the input points for that chain.

    seq_outputs : Sequence[numpy.ndarray]
        A sequence of output points for each chain, where each element is a numpy array
        of shape (n_points, output_dim) containing the output points for that chain.

    seq_masks : Sequence[numpy.ndarray]
        A sequence of mask arrays for each chain, where each element is a numpy array of
        shape (n_params,) containing boolean values indicating which parameters of the
        transformations in that chain are being optimized (True) and which are kept
        fixed (False).

    seq_guesses : Sequence[numpy.ndarray]
        A sequence of initial guess arrays for each transformation, where each element is
        a numpy array of shape (n_params,) containing the initial guess for the
        parameters of that transformation.

    seq_transform_kwargs : Sequence[Dict]
        A sequence of dictionaries containing additional keyword arguments to be passed to
        the _transform method of each transformation during the computation of the
        residuals and Jacobian. Each element in the sequence corresponds to a transformation
        in seq_transforms, and the dictionary contains the keyword arguments for that transformation.

    max_iterations : Optional[int]
        The maximum number of iterations for the optimization process. If None, there is
        no limit on the number of iterations.

    max_time : Optional[int]
        The maximum time in seconds for the optimization process. If None, there is no
        time limit for the optimization.

    ftol : Optional[Real]
        The tolerance for the cost function value to declare convergence. The optimization
        will stop when the cost function value changes less than ftol between iterations.

    xtol : Optional[Real]
        The tolerance for the parameters to declare convergence. The optimization will stop
        when the change in the parameters is less than xtol between iterations.

    gtol : Optional[Real]
        The tolerance for the gradient to declare convergence. The optimization will stop
        when the norm of the gradient is less than gtol.

    loss : Optional[str]
        The loss function to be used for robust optimization. If None, the standard least
        squares loss is used. Possible values include 'linear', 'soft_l1', 'huber', 'cauchy',
        and 'arctan'.

    filter_nans : bool
        If True, the function will filter out any NaN or infinite values in the residuals and
        Jacobian by setting them to zero. This can help to prevent the optimization from
        diverging due to invalid values, but it may also affect the convergence of the
        optimization if there are many NaN or infinite values.

    verbose_level : int
        The level of verbosity for the optimization process. Higher values will print more
        detailed information about the optimization process, including the cost function value,
        parameter values, and convergence status at each iteration.

    return_result : bool
        If True, the function will return the full result object from scipy.optimize.least_squares,
        which includes information about the optimization process, such as the optimized parameters,
        cost function value, number of iterations, and convergence status. If False, only the
        optimized parameters will be returned.

    return_history : bool
        If True, the function will return a history of the optimization process, which
        is a list of tuples containing the parameters and intermediate results at each
        iteration. If False, no history will be returned.

    _pretext : Optional[str], optional
        A pretext to display before the optimization process starts. This can be used to provide
        context or information about the optimization process that is about to begin.
        Default is None, which means no pretext is displayed.

    _sparse : bool, optional
        If True, the Jacobian matrix will be treated as a sparse matrix for the optimization process.
        This can be beneficial for large problems where the Jacobian is sparse, as it can reduce
        memory usage and improve computational efficiency. Default is False, which means the Jacobian
        will be treated as a dense matrix.

    Returns
    -------
    Tuple[numpy.ndarray]
        The optimized parameters of the transformations, where each element in the tuple
        corresponds to a transformation in seq_transforms, and is a numpy array of
        shape (n_params,) containing the optimized parameters for that transformation.

    scipy.optimize.OptimizeResult (optional)
        If return_result is True, the full result object from scipy.optimize.least_squares is also returned,
        which includes information about the optimization process, such as the optimized parameters,
        cost function value, number of iterations, and convergence status.

    Sequence[Tuple[numpy.ndarray]] (optional)
        If return_history is True, a history of the optimization process is also returned, which is
        a list of tuples containing the parameters and intermediate results at each iteration.

    """
    n_transforms = len(seq_transforms)
    n_chains = len(seq_chains)

    if n_transforms == 0:
        raise ValueError("No transformations to optimize.")
    if n_chains == 0:
        raise ValueError("No chains to optimize.")

    params_initial = numpy.concatenate(
        [seq_guesses[i][seq_masks[i]] for i in range(n_transforms)]
    )  # shape (sum(n_reduced_parameters),)

    f, jac = _build_residual_jacobian_functions(
        seq_transforms,
        seq_chains,
        seq_inputs,
        seq_outputs,
        seq_masks,
        seq_guesses,
        seq_transform_kwargs,
        filter_nans,
        _sparse=_sparse,
    )

    if verbose_level >= 3:
        _study_jacobian_least_squares(
            f(params_initial),
            jac(params_initial),
            params_initial,
            _pretext,
            _start=True,
            _sparse=_sparse,
        )

    # Run the least squares optimization
    result = scipy.optimize.least_squares(
        fun=f,
        x0=params_initial,
        jac=jac,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        verbose=min(verbose_level, 2),
        method="lm",  # Levenberg-Marquardt algorithm
        loss=loss,
    )

    if verbose_level >= 3:
        _study_jacobian_least_squares(
            f(result.x),
            jac(result.x),
            result.x,
            _pretext,
            _start=False,
            _sparse=_sparse,
        )

    parameters = []
    index = 0
    for i in range(n_transforms):
        p = seq_guesses[i].copy()
        p[seq_masks[i]] = result.x[index : index + numpy.sum(seq_masks[i])]
        parameters.append(p)
        index += numpy.sum(seq_masks[i])
    parameters = tuple(parameters)  # len n_transforms, each shape (n_params,)

    if return_result and return_history:
        return (
            parameters,
            result,
            [(numpy.full(parameters[i].shape, numpy.nan) for i in range(n_transforms))],
        )
    elif return_result:
        return parameters, result
    elif return_history:
        return parameters, [
            (numpy.full(parameters[i].shape, numpy.nan) for i in range(n_transforms))
        ]
    else:
        return parameters


def _solve_optimize_chains_gauss_newton(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[int]],
    seq_inputs: Sequence[numpy.ndarray],
    seq_outputs: Sequence[numpy.ndarray],
    seq_masks: Sequence[numpy.ndarray],
    seq_guesses: Sequence[numpy.ndarray],
    seq_transform_kwargs: Sequence[Dict],
    max_iterations: Optional[int],
    max_time: Optional[int],
    ftol: Optional[Real],
    xtol: Optional[Real],
    gtol: Optional[Real],
    filter_nans: bool,
    verbose_level: int,
    return_history: bool,
    _pretext: Optional[str] = None,
    _sparse: bool = False,
) -> Tuple[numpy.ndarray]:
    r"""
    Optimize the parameters of a set of transformations organized in chains using
    the Gauss-Newton optimization method.

    .. note::

        This method does not implement bounds or scaling of the parameters,
        and does not provide a robust optimization process for handling
        non-linear problems, which can lead to divergence or convergence to local minima.

    The optimization is performed by minimizing the residuals between the transformed
    input points and the output points for each chain solving iteratively a linearized
    version of the problem at each iteration.

    Lets consider a set of input points :math:`\vec{X}_I` with shape (..., input_dim)
    and a set of output points :math:`\vec{X}_O` with shape (..., output_dim).
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

    Where :math:`J = \nabla_{\lambda} T (\vec{X}_I, \lambda_0)` is the Jacobian matrix
    of the transformation with respect to the parameters, and
    :math:`R = \vec{X}_O - T(\vec{X}_I, \lambda_0)` is the residual vector.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of Transform objects to be optimized. The parameters of these
        transformations will be updated during the optimization process.

    seq_chains : Sequence[Sequence[int]]
        A sequence of chains, where each chain is a sequence of indices corresponding to
        the transformations in seq_transforms that are part of that chain. The order of
        the indices in each chain defines the order of the transformations in the chain.

    seq_inputs : Sequence[numpy.ndarray]
        A sequence of input points for each chain, where each element is a numpy array
        of shape (n_points, input_dim) containing the input points for that chain.

    seq_outputs : Sequence[numpy.ndarray]
        A sequence of output points for each chain, where each element is a numpy array
        of shape (n_points, output_dim) containing the output points for that chain.

    seq_masks : Sequence[numpy.ndarray]
        A sequence of mask arrays for each chain, where each element is a numpy array of
        shape (n_params,) containing boolean values indicating which parameters of the
        transformations in that chain are being optimized (True) and which are kept
        fixed (False).

    seq_guesses : Sequence[numpy.ndarray]
        A sequence of initial guess arrays for each transformation, where each element is
        a numpy array of shape (n_params,) containing the initial guess for the
        parameters of that transformation.

    seq_transform_kwargs : Sequence[Dict]
        A sequence of dictionaries containing additional keyword arguments to be passed to
        the _transform method of each transformation during the computation of the
        residuals and Jacobian. Each element in the sequence corresponds to a transformation
        in seq_transforms, and the dictionary contains the keyword arguments for that transformation.

    max_iterations : Optional[int]
        The maximum number of iterations for the optimization process. If None, there is
        no limit on the number of iterations.

    max_time : Optional[int]
        The maximum time in seconds for the optimization process. If None, there is no
        time limit for the optimization.

    ftol : Optional[Real]
        The tolerance for the cost function value to declare convergence. The optimization
        will stop when the cost function value changes less than ftol between iterations.

    xtol : Optional[Real]
        The tolerance for the parameters to declare convergence. The optimization will stop
        when the change in the parameters is less than xtol between iterations.

    gtol : Optional[Real]
        The tolerance for the gradient to declare convergence. The optimization will stop
        when the norm of the gradient is less than gtol.

    loss : Optional[str]
        The loss function to be used for robust optimization. If None, the standard least
        squares loss is used. Possible values include 'linear', 'soft_l1', 'huber', 'cauchy',
        and 'arctan'.

    filter_nans : bool
        If True, the function will filter out any NaN or infinite values in the residuals and
        Jacobian by setting them to zero. This can help to prevent the optimization from
        diverging due to invalid values, but it may also affect the convergence of the
        optimization if there are many NaN or infinite values.

    verbose_level : int
        The level of verbosity for the optimization process. Higher values will print more
        detailed information about the optimization process, including the cost function value,
        parameter values, and convergence status at each iteration.

    return_history : bool, optional
        If True, the function will return a history of the optimization process, which
        is a list of tuples containing the parameters and intermediate results at each
        iteration. If False, no history will be returned.
        Default is False, which means no history is returned.

    _pretext : Optional[str], optional
        A pretext to display before the optimization process starts. This can be used to provide
        context or information about the optimization process that is about to begin.
        Default is None, which means no pretext is displayed.

    _sparse : bool, optional
        If True, the Jacobian matrix will be treated as a sparse matrix for the optimization process.
        This can be beneficial for large problems where the Jacobian is sparse, as it can reduce
        memory usage and improve computational efficiency. Default is False, which means the Jacobian
        will be treated as a dense matrix.

    Returns
    -------
    parameters : Tuple[numpy.ndarray]
        The optimized parameters of the transformations, where each element in the tuple
        corresponds to a transformation in seq_transforms, and is a numpy array of
        shape (n_params,) containing the optimized parameters for that transformation.

    history : Optional[Sequence[Tuple[numpy.ndarray]]]
        If return_history is True, a history of the optimization process is also returned, which is
        a list of tuples containing the parameters and intermediate results at each iteration.
        If return_history is False, this will be None.

    """
    n_transforms = len(seq_transforms)
    n_chains = len(seq_chains)

    if n_transforms == 0:
        raise ValueError("No transformations to optimize.")
    if n_chains == 0:
        raise ValueError("No chains to optimize.")

    params_initial = numpy.concatenate(
        [seq_guesses[i][seq_masks[i]] for i in range(n_transforms)]
    )  # shape (sum(n_reduced_parameters),)

    f, jac = _build_residual_jacobian_functions(
        seq_transforms,
        seq_chains,
        seq_inputs,
        seq_outputs,
        seq_masks,
        seq_guesses,
        seq_transform_kwargs,
        filter_nans,
        _sparse=_sparse,
    )

    if verbose_level >= 3:
        _study_jacobian_least_squares(
            f(params_initial),
            jac(params_initial),
            params_initial,
            _pretext,
            _start=True,
            _sparse=_sparse,
        )

    # Run the least squares optimization with Gauss-Newton method
    start_time = time.time()
    params = params_initial.copy()
    iteration = 0
    R = None
    J = None
    JTR = None
    JTJ = None
    cost = None
    optimality = None
    last_cost = None
    last_optimality = None
    history = [] if return_history else None
    end = False

    if verbose_level >= 2:
        header = f" {'Iteration':^10}   {'Total nfev':^10}   {'Cost':^10}   {'Cost reduction':^15}   {'Step norm':^10}   {'Optimality':^10}"
        print(header)

    while not end:

        if R is None or J is None:
            R, J = f(params), jac(params)
        if JTR is None:
            JTR = -J.T @ R  # Warning (-) sign for the gradient as J = dR/dparams here
        if JTJ is None:
            JTJ = J.T @ J

        if not _sparse:
            delta = numpy.linalg.solve(JTJ, JTR)
        else:
            delta = scipy.sparse.linalg.spsolve(JTJ, JTR)

        params = params + delta

        if cost is None:
            cost = 0.5 * numpy.dot(R, R)
        last_cost = cost

        if optimality is None:
            optimality = numpy.linalg.norm(JTR, ord=numpy.inf)
        last_optimality = optimality

        R, J, JTR, JTJ, cost, optimality = (
            None,
            None,
            None,
            None,
            None,
            None,
        )  # Invalidate cached values

        if verbose_level >= 2 and iteration == 0:
            print(
                f" {iteration:^10}   {iteration+1:^10}   {last_cost:^10.3e}   {'':^15}   {'':^10}   {last_optimality:^10.3e}"
            )

        if verbose_level >= 2 or ftol is not None:
            if R is None or J is None:
                R, J = f(params), jac(params)
            cost = 0.5 * numpy.dot(R, R)
            cost_reduction = last_cost - cost

        if verbose_level >= 2 or gtol is not None:
            if JTR is None:
                JTR = -J.T @ R  # Warning (-) sign for the gradient
            optimality = numpy.linalg.norm(JTR, ord=numpy.inf)

        if verbose_level >= 2 or xtol is not None:
            step_norm = numpy.linalg.norm(delta)
            norm = numpy.linalg.norm(params)

        if verbose_level >= 2:
            print(
                f" {iteration+1:^10}   {iteration+1:^10}   {cost:^10.3e}   {cost_reduction:^15.3e}   {step_norm:^10.3e}   {optimality:^10.3e}"
            )

        if return_history:
            parameters = []
            index = 0
            for i in range(n_transforms):
                p = seq_guesses[i].copy()
                p[seq_masks[i]] = params[index : index + numpy.sum(seq_masks[i])]
                parameters.append(p)
                index += numpy.sum(seq_masks[i])

            parameters = tuple(parameters)  # len n_transforms, each shape (n_params,)
            history.append(parameters)

        if ftol is not None:
            if cost_reduction < ftol * cost and cost_reduction >= 0:
                if verbose_level >= 1:
                    print(
                        f"Cost reduction {cost_reduction:.3e} is less than ftol * cost {ftol * cost:.3e}, stopping optimization."
                    )
                end = True

        if xtol is not None:
            if step_norm < xtol * (xtol + norm):
                if verbose_level >= 1:
                    print(
                        f"Step norm {step_norm:.3e} is less than xtol * (xtol + norm) {xtol * (xtol + norm):.3e}, stopping optimization."
                    )
                end = True

        if gtol is not None:
            if optimality < gtol:
                if verbose_level >= 1:
                    print(
                        f"Optimality {optimality:.3e} is less than gtol {gtol:.3e}, stopping optimization."
                    )
                end = True

        if max_iterations is not None and iteration >= max_iterations:
            if verbose_level >= 1:
                print(
                    f"Maximum number of iterations {max_iterations} reached, stopping optimization."
                )
            end = True
        elif max_time is not None and (time.time() - start_time) > max_time:
            if verbose_level >= 1:
                print(
                    f"Maximum time of {max_time} seconds exceeded, stopping optimization."
                )
            end = True

        iteration += 1

    if verbose_level >= 3:
        _study_jacobian_least_squares(
            f(params),
            jac(params),
            params,
            _pretext,
            _start=False,
            _sparse=_sparse,
        )

    parameters = []
    index = 0
    for i in range(n_transforms):
        p = seq_guesses[i].copy()
        p[seq_masks[i]] = params[index : index + numpy.sum(seq_masks[i])]
        parameters.append(p)
        index += numpy.sum(seq_masks[i])
    parameters = tuple(parameters)  # len n_transforms, each shape (n_params,)

    if return_history:
        return parameters, history
    return parameters


def optimize_parameters_gn(
    transform: Transform,
    input_points: ArrayLike,
    output_points: ArrayLike,
    *,
    guess: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None,
    transform_kwargs: Optional[Dict] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_history: bool = False,
    inplace: bool = False,
) -> numpy.ndarray:
    r"""
    Optimize the ``parameters`` of a :class:`Transform` object such that the transformed
    input points match the output points using a Gauss-Newton optimization method.

    Lets consider a set of input points :math:`\vec{X}_I` with shape (..., input_dim)
    and a set of output points :math:`\vec{X}_O` with shape (..., output_dim).
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

    Where :math:`J = \nabla_{\lambda} T (\vec{X}_I, \lambda_0)` is the Jacobian matrix
    of the transformation with respect to the parameters, and
    :math:`R = \vec{X}_O - T(\vec{X}_I, \lambda_0)` is the residual vector.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.


    Parameters
    ----------
    transform : :class:`Transform`
        The transformation object to be optimized. The ``constants`` attribute of the
        transformation must be set before calling this function. If the ``parameters``
        attribute of the transformation is set, it will be used as the initial guess for
        the optimization if the `guess` parameter is None. Note that the input
        :class:`Transform` object is not modified during the optimization process,
        a copy of the object is created and modified internally to perform the
        optimization if `inplace` is False.

    input_points : ArrayLike
        The input points with shape (..., input_dim) such that their transformation
        is expected to match the output points.

    output_points : ArrayLike
        The output points to be matched with shape (..., output_dim).

    guess : Optional[ArrayLike], optional
        The initial guess for the parameters of the transformation with shape
        (n_params,). If None, the current parameters of the transformation are used.

    mask : Optional[ArrayLike], optional
        A mask array of shape (n_params,) indicating which parameters should be
        optimized. Elements with a value of True are optimized, while elements with a
        value of False are kept fixed. Default is None, which means all parameters are
        optimized.

    transform_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``transform._transform`` method.
        Default is None, which means no additional keyword arguments are passed to the
        transformation.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations.
        The optimization process is stopped when the number of iterations
        exceeds ``max_iterations``. Default is None, which means no limit on the
        number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``.
        The default value is None, which means the ftol criterion is not used for
        stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters.
        The optimization process is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        The default value is None, which means the xtol criterion is not used for
        stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient.
        The optimization process is stopped when ``norm(g_scaled, ord=numpy.inf) < gtol``
        where g_scaled is the value of the gradient scaled to account for the
        presence of the bounds. The default value is None, which means the gtol
        criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set to
        ``1e-8``. If any of the stopping criteria is already specified, it will
        be overridden by the user-specified value.

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_history : bool, optional
        If True, the function returns a history of the parameters during the
        optimization process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformation. If False (default), a copy of the transformation is created
        and modified internally to perform the optimization, leaving the input transformation
        unchanged.


    Returns
    -------
    parameters : numpy.ndarray
        The optimized parameters of the transformation with shape (n_params,).
        This array contains both the optimized parameters (corresponding to True values
        in the `mask`) and the fixed parameters (corresponding to False values in the
        `mask`), where the fixed parameters are equal to their initial values.

    history : List[numpy.ndarray], optional
        A history of the optimization process including the parameters with shape
        (n_params,). Returned only if `return_history` is True.


    See Also
    --------
    pycvcam.optimize.optimize_parameters_trf
        Optimize the parameters of a transformation using the least squares method with
        the Trust Region Reflective algorithm allowing for bounds and scaling of
        the parameters, and robust optimization.

    pycvcam.optimize.optimize_camera_gn
        Optimize the parameters of a camera transformation using the Gauss-Newton method.

    pycvcam.optimize.optimize_chains_gn
        Optimize the parameters of a set of transformations organized in chains using the
        Gauss-Newton method.

    """
    # -------------
    # Input Formats Check
    # -------------
    if not isinstance(transform, Transform):
        raise TypeError(
            f"transform must be an instance of Transform, got {type(transform)}"
        )
    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")
    if not inplace:
        transform = transform.copy()

    input_points = numpy.asarray(input_points, dtype=numpy.float64)
    output_points = numpy.asarray(output_points, dtype=numpy.float64)

    if input_points.ndim < 2 or output_points.ndim < 2:
        raise ValueError(
            f"Input and output points must have at least 2 dimensions, "
            f"got {input_points.ndim} and {output_points.ndim} dimensions respectively."
        )
    if input_points.shape[-1] != transform.input_dim:
        raise ValueError(
            f"Last dimension of input points must be {transform.input_dim}, "
            f"got {input_points.shape[-1]}"
        )
    if output_points.shape[-1] != transform.output_dim:
        raise ValueError(
            f"Last dimension of output points must be {transform.output_dim}, "
            f"got {output_points.shape[-1]}"
        )
    if input_points.shape[:-1] != output_points.shape[:-1]:
        raise ValueError(
            f"Input and output points must have the same structure, "
            f"got {input_points.shape} and {output_points.shape} respectively."
        )
    if input_points.shape[0] == 0:
        raise ValueError("Input and output points must have at least one point.")

    input_points = input_points.reshape(-1, transform.input_dim)
    output_points = output_points.reshape(-1, transform.output_dim)

    if guess is None and not transform.is_set():
        raise ValueError(
            f"Initial guess for the parameters is required when the current parameters "
            f"of the transformation are not set."
        )
    elif guess is None and transform.is_set():
        if transform.parameters is None:
            guess = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess = transform.parameters.copy()
    else:
        guess = numpy.asarray(guess, dtype=numpy.float64)
    if guess.ndim != 1 or guess.size != transform.n_params:
        raise ValueError(
            f"Guess must be a 1D array with {transform.n_params} parameters, "
            f"got {guess.ndim} dimensions and {guess.size} parameters."
        )

    if mask is None:
        mask = numpy.ones(transform.n_params, dtype=bool)
    else:
        mask = numpy.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size != transform.n_params:
        raise ValueError(
            f"Mask must be a 1D array with {transform.n_params} parameters, "
            f"got {mask.ndim} dimensions and {mask.size} parameters."
        )

    if transform_kwargs is None:
        transform_kwargs = {}
    if not isinstance(transform_kwargs, dict):
        raise TypeError(
            f"transform_kwargs must be a dictionary, got {type(transform_kwargs)}"
        )

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if transform.n_params == 0:
        if verbose_level >= 1:
            print(
                "The transformation has no parameters to optimize, returning the current parameters."
            )
        out = guess.copy()
        if return_history:
            return out, [out]
        else:
            return out

    if not any(mask):
        if verbose_level >= 1:
            print(
                "No parameters to optimize (all parameters are fixed), returning the current parameters."
            )
        out = guess.copy()
        if return_history:
            return out, [out]
        else:
            return out

    # -------------
    # Optimization
    # -------------
    seq_transforms = (transform,)
    seq_chains = ((0,),)
    seq_inputs = (input_points,)
    seq_outputs = (output_points,)
    seq_masks = (mask,)
    seq_guesses = (guess,)
    seq_transform_kwargs = (transform_kwargs,)

    out = _solve_optimize_chains_gauss_newton(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_history=return_history,
        _pretext=None,
        _sparse=False,
    )

    if return_history:
        p, h = out
        return p[0], [h_i[0] for h_i in h]
    else:
        return out[0]


def optimize_parameters_trf(
    transform: Transform,
    input_points: ArrayLike,
    output_points: ArrayLike,
    *,
    guess: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None,
    scale: Optional[ArrayLike] = None,
    bounds: Optional[ArrayLike] = None,
    transform_kwargs: Optional[Dict] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    loss: Optional[str] = None,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_result: bool = False,
    return_history: bool = False,
    inplace: bool = False,
) -> numpy.ndarray:
    r"""
    Optimize the ``parameters`` of a :class:`Transform` object such that the transformed
    input points match the output points using the ``scipy.optimize.least_squares``
    method. The computation is done with Trust Region Reflective algorithm.

    Lets consider a set of input points :math:`\vec{X}_I` with shape (..., input_dim)
    and a set of output points :math:`\vec{X}_O` with shape (..., output_dim).
    We search :math:`\lambda` such that:

    .. math::

        \vec{X}_O = \text{Transform}(\vec{X}_I, \lambda) = T(\vec{X}_I, \lambda)

    .. note::

        The current parameters of the transformation are not directly modified.

    We define the residual vector as:

    .. math::

        R(\lambda) = \vec{X}_O - T(\vec{X}_I, \lambda)

    The optimization problem is then defined as:

    .. math::

        \min_{\lambda} \|R(\lambda)\|^2

    We have the following system of equations for the Jacobian matrix:

    .. math::

        \min_{\lambda} \|R(\lambda)\|^2 \approx \min_{\Delta \lambda} \|R(\lambda_i) - J(\lambda_i) \Delta \lambda\|^2

    .. math::

        J^T(\lambda_i) J(\lambda_i) \Delta \lambda = J^T(\lambda_i) R(\lambda_i)

    where :math:`J(\lambda) = \frac{\partial T(\vec{X}_I, \lambda)}{\partial \lambda}`
    is the Jacobian matrix of the transformation with respect to the parameters.

    .. note::

        This method can be used to optimize the parameters of any transformation that
        implements the `_transform` method.

    For more information about the optimization method, please refer to the
    ``scipy.optimize.least_squares`` documentation.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.


    Parameters
    ----------
    transform : :class:`Transform`
        The transformation object to be optimized. The ``constants`` attribute of the
        transformation must be set before calling this function. If the ``parameters``
        attribute of the transformation is set, it will be used as the initial guess for
        the optimization if the `guess` parameter is None. Note that the input
        :class:`Transform` object is not modified during the optimization process,
        a copy of the object is created and modified internally to perform the
        optimization if `inplace` is False.

    input_points : ArrayLike
        The input points with shape (..., input_dim) such that their transformation
        is expected to match the output points.

    output_points : ArrayLike
        The output points to be matched with shape (..., output_dim).

    guess : Optional[ArrayLike], optional
        The initial guess for the parameters of the transformation with shape
        (n_params,). If None, the current parameters of the transformation are used.

    mask : Optional[ArrayLike], optional
        A mask array of shape (n_params,) indicating which parameters should be
        optimized. Elements with a value of True are optimized, while elements with a
        value of False are kept fixed. Default is None, which means all parameters are
        optimized.

    scale : Optional[ArrayLike], optional
        An array of shape (n_params,) indicating the scale of each parameter for the
        optimization. This is used to improve the convergence of the optimization by
        scaling the parameters to a similar range. Default is None, which means no
        scaling is applied (i.e., all parameters are scaled to 1).

    bounds : Optional[ArrayLike], optional
        The bounds for the parameters of the transformation with shape (2, n_params).
        The first row contains the lower bounds and the second row contains the upper
        bounds for each parameter. Default is None, which means no bounds are applied.
        Set ``+/- numpy.inf`` for no bound on a specific parameter.

    transform_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``transform._transform`` method.
        Default is None, which means no additional keyword arguments are passed to the
        transformation.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations.
        The optimization process is stopped when the number of iterations
        exceeds ``max_iterations``. Default is None, which means no limit on the
        number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``.
        The default value is None, which means the ftol criterion is not used for
        stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters.
        The optimization process is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        The default value is None, which means the xtol criterion is not used for
        stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient.
        The optimization process is stopped when ``norm(g_scaled, ord=numpy.inf) < gtol``
        where g_scaled is the value of the gradient scaled to account for the
        presence of the bounds. The default value is None, which means the gtol
        criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set to
        ``1e-8``. If any of the stopping criteria is already specified, it will
        be overridden by the user-specified value.

    loss : Optional[str], optional
        If specified, the optimization will use a robust loss function to reduce the
        influence of outliers in the optimization. The available loss functions are
        'huber', 'cauchy', 'arctan', 'soft_l1', and 'linear'. Default is None, which
        means the standard least squares loss is used (i.e., 'linear').

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_result : bool, optional
        If True, the function returns the ``scipy.optimize.OptimizeResult`` object
        containing information about the convergence of the optimization process.
        Default is False, which means only the optimized parameters are returned.
        If ``n_params`` is 0, or all parameters are masked, the result output will be
        None.

    return_history : bool, optional
        If True, the function returns a history of the parameters during the
        optimization process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformation. If False (default), a copy of the transformation is created
        and modified internally to perform the optimization, leaving the input transformation
        unchanged.


    Returns
    -------
    parameters : numpy.ndarray
        The optimized parameters of the transformation with shape (n_params,).
        This array contains both the optimized parameters (corresponding to True values
        in the `mask`) and the fixed parameters (corresponding to False values in the
        `mask`), where the fixed parameters are equal to their initial values.

    result : scipy.optimize.OptimizeResult, optional
        The result of the optimization process containing information about the
        convergence of the optimization. Returned only if `return_result` is True.

        .. warning::

            Only contains the parameters that were optimized (i.e., the parameters
            corresponding to True values in the `mask`).

    history : List[numpy.ndarray], optional
        A history of the optimization process including the parameters with shape
        (n_params,). Returned only if `return_history` is True.


    See Also
    --------
    scipy.optimize.least_squares
        For more information about the optimization method.

    pycvcam.optimize.optimize_parameters_gn
        Optimize the parameters of a transformation using the Gauss-Newton method.

    pycvcam.optimize.optimize_camera_trf
        Optimize the parameters of a camera transformation using the least squares
        method with the Trust Region Reflective algorithm.

    pycvcam.optimize.optimize_chains_trf
        Optimize the parameters of a set of transformations organized in chains using
        the least squares method with the Trust Region Reflective algorithm.

    """
    # -------------
    # Input Formats Check
    # -------------
    if not isinstance(transform, Transform):
        raise TypeError(
            f"transform must be an instance of Transform, got {type(transform)}"
        )
    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")
    if not inplace:
        transform = transform.copy()

    input_points = numpy.asarray(input_points, dtype=numpy.float64)
    output_points = numpy.asarray(output_points, dtype=numpy.float64)

    if input_points.ndim < 2 or output_points.ndim < 2:
        raise ValueError(
            f"Input and output points must have at least 2 dimensions, "
            f"got {input_points.ndim} and {output_points.ndim} dimensions respectively."
        )
    if input_points.shape[-1] != transform.input_dim:
        raise ValueError(
            f"Last dimension of input points must be {transform.input_dim}, "
            f"got {input_points.shape[-1]}"
        )
    if output_points.shape[-1] != transform.output_dim:
        raise ValueError(
            f"Last dimension of output points must be {transform.output_dim}, "
            f"got {output_points.shape[-1]}"
        )
    if input_points.shape[:-1] != output_points.shape[:-1]:
        raise ValueError(
            f"Input and output points must have the same structure, "
            f"got {input_points.shape} and {output_points.shape} respectively."
        )
    if input_points.shape[0] == 0:
        raise ValueError("Input and output points must have at least one point.")

    input_points = input_points.reshape(-1, transform.input_dim)
    output_points = output_points.reshape(-1, transform.output_dim)

    if guess is None and not transform.is_set():
        raise ValueError(
            f"Initial guess for the parameters is required when the current parameters "
            f"of the transformation are not set."
        )
    elif guess is None and transform.is_set():
        if transform.parameters is None:
            guess = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess = transform.parameters.copy()
    else:
        guess = numpy.asarray(guess, dtype=numpy.float64)
    if guess.ndim != 1 or guess.size != transform.n_params:
        raise ValueError(
            f"Guess must be a 1D array with {transform.n_params} parameters, "
            f"got {guess.ndim} dimensions and {guess.size} parameters."
        )

    if mask is None:
        mask = numpy.ones(transform.n_params, dtype=bool)
    else:
        mask = numpy.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size != transform.n_params:
        raise ValueError(
            f"Mask must be a 1D array with {transform.n_params} parameters, "
            f"got {mask.ndim} dimensions and {mask.size} parameters."
        )

    if scale is None:
        scale = numpy.ones(transform.n_params, dtype=numpy.float64)
    else:
        scale = numpy.asarray(scale, dtype=numpy.float64)
    if scale.ndim != 1 or scale.size != transform.n_params:
        raise ValueError(
            f"Scale must be a 1D array with {transform.n_params} parameters, "
            f"got {scale.ndim} dimensions and {scale.size} parameters."
        )

    if bounds is None:
        bounds = numpy.array(
            [[-numpy.inf] * transform.n_params, [numpy.inf] * transform.n_params],
            dtype=numpy.float64,
        )
    else:
        bounds = numpy.asarray(bounds, dtype=numpy.float64)
    if bounds.ndim != 2 or bounds.shape != (2, transform.n_params):
        raise ValueError(
            f"Bounds must be a 2D array with shape (2, {transform.n_params}), "
            f"got {bounds.ndim} dimensions and shape {bounds.shape}."
        )
    if not all(
        bounds[0, i] <= guess[i] <= bounds[1, i] for i in range(transform.n_params)
    ):
        raise ValueError(
            f"Initial guess must be within the bounds for each parameter. "
            f"Got guess {guess} and bounds {bounds}."
        )

    if transform_kwargs is None:
        transform_kwargs = {}
    if not isinstance(transform_kwargs, dict):
        raise TypeError(
            f"transform_kwargs must be a dictionary, got {type(transform_kwargs)}"
        )

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if loss is not None:
        if not isinstance(loss, str) or loss not in [
            "huber",
            "cauchy",
            "arctan",
            "soft_l1",
            "linear",
        ]:
            raise ValueError(
                f"loss must be one of 'huber', 'cauchy', 'arctan', 'soft_l1', or 'linear', got {loss}"
            )
    else:
        loss = "linear"

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_result, bool):
        raise TypeError(f"return_result must be a boolean, got {type(return_result)}")

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if transform.n_params == 0:
        if verbose_level >= 1:
            print(
                "The transformation has no parameters to optimize, returning the current parameters."
            )
        out = guess.copy()
        if return_history and return_result:
            return out, None, [out]
        elif return_history and not return_result:
            return out, [out]
        elif not return_history and return_result:
            return out, None
        else:
            return out

    if not any(mask):
        if verbose_level >= 1:
            print(
                "No parameters to optimize (all parameters are fixed), returning the current parameters."
            )
        out = guess.copy()
        if return_history and return_result:
            return out, None, [out]
        elif return_history and not return_result:
            return out, [out]
        elif not return_history and return_result:
            return out, None
        else:
            return out

    # -------------
    # Optimization
    # -------------
    seq_transforms = (transform,)
    seq_chains = ((0,),)
    seq_inputs = (input_points,)
    seq_outputs = (output_points,)
    seq_masks = (mask,)
    seq_guesses = (guess,)
    seq_bounds = (bounds,)
    seq_scales = (scale,)
    seq_transform_kwargs = (transform_kwargs,)

    out = _solve_optimize_chains_trf_scipy(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_bounds=seq_bounds,
        seq_scales=seq_scales,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        loss=loss,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_result=return_result,
        return_history=return_history,
        _pretext=None,
        _sparse=False,
    )

    if return_history and return_result:
        p, r, h = out
        return p[0], r, [h_i[0] for h_i in h]
    elif return_history and not return_result:
        p, h = out
        return p[0], [h_i[0] for h_i in h]
    elif not return_history and return_result:
        p, r = out
        return p[0], r
    else:
        return out[0]


def optimize_parameters_lm(
    transform: Transform,
    input_points: ArrayLike,
    output_points: ArrayLike,
    *,
    guess: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None,
    transform_kwargs: Optional[Dict] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    loss: Optional[str] = None,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_result: bool = False,
    return_history: bool = False,
    inplace: bool = False,
) -> numpy.ndarray:
    r"""
    Optimize the ``parameters`` of a :class:`Transform` object such that the transformed
    input points match the output points using the ``scipy.optimize.least_squares``
    method. The computation is done with the Levenberg-Marquardt algorithm.

    Lets consider a set of input points :math:`\vec{X}_I` with shape (..., input_dim)
    and a set of output points :math:`\vec{X}_O` with shape (..., output_dim).
    We search :math:`\lambda` such that:

    .. math::

        \vec{X}_O = \text{Transform}(\vec{X}_I, \lambda) = T(\vec{X}_I, \lambda)

    .. note::

        The current parameters of the transformation are not directly modified.

    We define the residual vector as:

    .. math::

        R(\lambda) = \vec{X}_O - T(\vec{X}_I, \lambda)

    The optimization problem is then defined as:

    .. math::

        \min_{\lambda} \|R(\lambda)\|^2

    We have the following system of equations for the Jacobian matrix:

    .. math::

        \min_{\lambda} \|R(\lambda)\|^2 \approx \min_{\Delta \lambda} \|R(\lambda_i) - J(\lambda_i) \Delta \lambda\|^2

    .. math::

        J^T(\lambda_i) J(\lambda_i) \Delta \lambda = J^T(\lambda_i) R(\lambda_i)

    where :math:`J(\lambda) = \frac{\partial T(\vec{X}_I, \lambda)}{\partial \lambda}`
    is the Jacobian matrix of the transformation with respect to the parameters.

    .. note::

        This method can be used to optimize the parameters of any transformation that
        implements the `_transform` method.

    For more information about the optimization method, please refer to the
    ``scipy.optimize.least_squares`` documentation.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.


    Parameters
    ----------
    transform : :class:`Transform`
        The transformation object to be optimized. The ``constants`` attribute of the
        transformation must be set before calling this function. If the ``parameters``
        attribute of the transformation is set, it will be used as the initial guess for
        the optimization if the `guess` parameter is None. Note that the input
        :class:`Transform` object is not modified during the optimization process,
        a copy of the object is created and modified internally to perform the
        optimization if `inplace` is False.

    input_points : ArrayLike
        The input points with shape (..., input_dim) such that their transformation
        is expected to match the output points.

    output_points : ArrayLike
        The output points to be matched with shape (..., output_dim).

    guess : Optional[ArrayLike], optional
        The initial guess for the parameters of the transformation with shape
        (n_params,). If None, the current parameters of the transformation are used.

    mask : Optional[ArrayLike], optional
        A mask array of shape (n_params,) indicating which parameters should be
        optimized. Elements with a value of True are optimized, while elements with a
        value of False are kept fixed. Default is None, which means all parameters are
        optimized.

    transform_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``transform._transform`` method.
        Default is None, which means no additional keyword arguments are passed to the
        transformation.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations.
        The optimization process is stopped when the number of iterations
        exceeds ``max_iterations``. Default is None, which means no limit on the
        number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``.
        The default value is None, which means the ftol criterion is not used for
        stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters.
        The optimization process is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        The default value is None, which means the xtol criterion is not used for
        stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient.
        The optimization process is stopped when ``norm(g_scaled, ord=numpy.inf) < gtol``
        where g_scaled is the value of the gradient scaled to account for the
        presence of the bounds. The default value is None, which means the gtol
        criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set to
        ``1e-8``. If any of the stopping criteria is already specified, it will
        be overridden by the user-specified value.

    loss : Optional[str], optional
        If specified, the optimization will use a robust loss function to reduce the
        influence of outliers in the optimization. The available loss functions are
        'huber', 'cauchy', 'arctan', 'soft_l1', and 'linear'. Default is None, which
        means the standard least squares loss is used (i.e., 'linear').

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_result : bool, optional
        If True, the function returns the ``scipy.optimize.OptimizeResult`` object
        containing information about the convergence of the optimization process.
        Default is False, which means only the optimized parameters are returned.
        If ``n_params`` is 0, or all parameters are masked, the result output will be
        None.

    return_history : bool, optional
        If True, the function returns a history of the parameters during the
        optimization process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformation. If False (default), a copy of the transformation is created
        and modified internally to perform the optimization, leaving the input transformation
        unchanged.


    Returns
    -------
    parameters : numpy.ndarray
        The optimized parameters of the transformation with shape (n_params,).
        This array contains both the optimized parameters (corresponding to True values
        in the `mask`) and the fixed parameters (corresponding to False values in the
        `mask`), where the fixed parameters are equal to their initial values.

    result : scipy.optimize.OptimizeResult, optional
        The result of the optimization process containing information about the
        convergence of the optimization. Returned only if `return_result` is True.

        .. warning::

            Only contains the parameters that were optimized (i.e., the parameters
            corresponding to True values in the `mask`).

    history : List[numpy.ndarray], optional
        A history of the optimization process including the parameters with shape
        (n_params,). Returned only if `return_history` is True.


    See Also
    --------
    scipy.optimize.least_squares
        For more information about the optimization method.

    pycvcam.optimize.optimize_parameters_gn
        Optimize the parameters of a transformation using the Gauss-Newton method.

    pycvcam.optimize.optimize_camera_lm
        Optimize the parameters of a camera transformation using the least squares
        method with the Levenberg-Marquardt algorithm.

    pycvcam.optimize.optimize_chains_lm
        Optimize the parameters of a set of transformations organized in chains using
        the least squares method with the Levenberg-Marquardt algorithm.

    """
    # -------------
    # Input Formats Check
    # -------------
    if not isinstance(transform, Transform):
        raise TypeError(
            f"transform must be an instance of Transform, got {type(transform)}"
        )
    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")
    if not inplace:
        transform = transform.copy()

    input_points = numpy.asarray(input_points, dtype=numpy.float64)
    output_points = numpy.asarray(output_points, dtype=numpy.float64)

    if input_points.ndim < 2 or output_points.ndim < 2:
        raise ValueError(
            f"Input and output points must have at least 2 dimensions, "
            f"got {input_points.ndim} and {output_points.ndim} dimensions respectively."
        )
    if input_points.shape[-1] != transform.input_dim:
        raise ValueError(
            f"Last dimension of input points must be {transform.input_dim}, "
            f"got {input_points.shape[-1]}"
        )
    if output_points.shape[-1] != transform.output_dim:
        raise ValueError(
            f"Last dimension of output points must be {transform.output_dim}, "
            f"got {output_points.shape[-1]}"
        )
    if input_points.shape[:-1] != output_points.shape[:-1]:
        raise ValueError(
            f"Input and output points must have the same structure, "
            f"got {input_points.shape} and {output_points.shape} respectively."
        )
    if input_points.shape[0] == 0:
        raise ValueError("Input and output points must have at least one point.")

    input_points = input_points.reshape(-1, transform.input_dim)
    output_points = output_points.reshape(-1, transform.output_dim)

    if guess is None and not transform.is_set():
        raise ValueError(
            f"Initial guess for the parameters is required when the current parameters "
            f"of the transformation are not set."
        )
    elif guess is None and transform.is_set():
        if transform.parameters is None:
            guess = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess = transform.parameters.copy()
    else:
        guess = numpy.asarray(guess, dtype=numpy.float64)
    if guess.ndim != 1 or guess.size != transform.n_params:
        raise ValueError(
            f"Guess must be a 1D array with {transform.n_params} parameters, "
            f"got {guess.ndim} dimensions and {guess.size} parameters."
        )

    if mask is None:
        mask = numpy.ones(transform.n_params, dtype=bool)
    else:
        mask = numpy.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size != transform.n_params:
        raise ValueError(
            f"Mask must be a 1D array with {transform.n_params} parameters, "
            f"got {mask.ndim} dimensions and {mask.size} parameters."
        )

    if transform_kwargs is None:
        transform_kwargs = {}
    if not isinstance(transform_kwargs, dict):
        raise TypeError(
            f"transform_kwargs must be a dictionary, got {type(transform_kwargs)}"
        )

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if loss is not None:
        if not isinstance(loss, str) or loss not in [
            "huber",
            "cauchy",
            "arctan",
            "soft_l1",
            "linear",
        ]:
            raise ValueError(
                f"loss must be one of 'huber', 'cauchy', 'arctan', 'soft_l1', or 'linear', got {loss}"
            )
    else:
        loss = "linear"

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_result, bool):
        raise TypeError(f"return_result must be a boolean, got {type(return_result)}")

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if transform.n_params == 0:
        if verbose_level >= 1:
            print(
                "The transformation has no parameters to optimize, returning the current parameters."
            )
        out = guess.copy()
        if return_history and return_result:
            return out, None, [out]
        elif return_history and not return_result:
            return out, [out]
        elif not return_history and return_result:
            return out, None
        else:
            return out

    if not any(mask):
        if verbose_level >= 1:
            print(
                "No parameters to optimize (all parameters are fixed), returning the current parameters."
            )
        out = guess.copy()
        if return_history and return_result:
            return out, None, [out]
        elif return_history and not return_result:
            return out, [out]
        elif not return_history and return_result:
            return out, None
        else:
            return out

    # -------------
    # Optimization
    # -------------
    seq_transforms = (transform,)
    seq_chains = ((0,),)
    seq_inputs = (input_points,)
    seq_outputs = (output_points,)
    seq_masks = (mask,)
    seq_guesses = (guess,)
    seq_transform_kwargs = (transform_kwargs,)

    out = _solve_optimize_chains_lm_scipy(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        loss=loss,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_result=return_result,
        return_history=return_history,
        _pretext=None,
        _sparse=False,
    )

    if return_history and return_result:
        p, r, h = out
        return p[0], r, [h_i[0] for h_i in h]
    elif return_history and not return_result:
        p, h = out
        return p[0], [h_i[0] for h_i in h]
    elif not return_history and return_result:
        p, r = out
        return p[0], r
    else:
        return out[0]


def optimize_camera_gn(
    intrinsic: Optional[Intrinsic],
    distortion: Optional[Distortion],
    extrinsic: Optional[Extrinsic],
    world_points: ArrayLike,
    image_points: ArrayLike,
    *,
    guess_intrinsic: Optional[ArrayLike] = None,
    guess_distortion: Optional[ArrayLike] = None,
    guess_extrinsic: Optional[ArrayLike] = None,
    mask_intrinsic: Optional[ArrayLike] = None,
    mask_distortion: Optional[ArrayLike] = None,
    mask_extrinsic: Optional[ArrayLike] = None,
    intrinsic_kwargs: Optional[Dict] = None,
    distortion_kwargs: Optional[Dict] = None,
    extrinsic_kwargs: Optional[Dict] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_history: bool = False,
    inplace: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Optimize the parameters of the intrinsic, distortion, and extrinsic transformations
    of a camera model such that the projection of the world points matches the image
    points using a Gauss-Newton optimization method.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.

    Parameters
    ----------
    intrinsic : Optional[Intrinsic]
        The intrinsic transformation of the camera model to be optimized. If None, the
        intrinsic transformation is not included in the optimization.

    distortion : Optional[Distortion]
        The distortion transformation of the camera model to be optimized. If None, the
        distortion transformation is not included in the optimization.

    extrinsic : Optional[Extrinsic]
        The extrinsic transformation of the camera model to be optimized. If None, the
        extrinsic transformation is not included in the optimization.

    world_points : ArrayLike
        The world points with shape (..., 3) such that their projection is expected to
        match the image points.

    image_points : ArrayLike
        The image points to be matched with shape (..., 2).

    guess_intrinsic : Optional[ArrayLike], optional
        The initial guess for the parameters of the intrinsic transformation with shape
        (n_intrinsic_params,). If None, the current parameters of the intrinsic
        transformation are used. Default is None.

    guess_distortion : Optional[ArrayLike], optional
        The initial guess for the parameters of the distortion transformation with shape
        (n_distortion_params,). If None, the current parameters of the distortion
        transformation are used. Default is None.

    guess_extrinsic : Optional[ArrayLike], optional
        The initial guess for the parameters of the extrinsic transformation with shape
        (n_extrinsic_params,). If None, the current parameters of the extrinsic
        transformation are used. Default is None.

    mask_intrinsic : Optional[ArrayLike], optional
        A mask array of shape (n_intrinsic_params,) indicating which parameters of the
        intrinsic transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the intrinsic transformation are optimized.

    mask_distortion : Optional[ArrayLike], optional
        A mask array of shape (n_distortion_params,) indicating which parameters of the
        distortion transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the distortion transformation are optimized.

    mask_extrinsic : Optional[ArrayLike], optional
        A mask array of shape (n_extrinsic_params,) indicating which parameters of the
        extrinsic transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the extrinsic transformation are optimized.

    intrinsic_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``intrinsic._transform`` method. Default is
        None, which means no additional keyword arguments are passed to the intrinsic
        transformation.

    distortion_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``distortion._transform`` method. Default
        is None, which means no additional keyword arguments are passed to the
        distortion transformation.

    extrinsic_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``extrinsic._transform`` method. Default is
        None, which means no additional keyword arguments are passed to the extrinsic
        transformation.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations.
        The optimization process is stopped when the number of iterations
        exceeds ``max_iterations``. Default is None, which means no limit on the
        number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``.
        The default value is None, which means the ftol criterion is not used for
        stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters.
        The optimization process is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        The default value is None, which means the xtol criterion is not used for
        stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient.
        The optimization process is stopped when ``norm(g_scaled, ord=numpy.inf) < gtol``
        where g_scaled is the value of the gradient scaled to account for the
        presence of the bounds. The default value is None, which means the gtol
        criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set to
        ``1e-8``. If any of the stopping criteria is already specified, it will
        be overridden by the user-specified value.

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_history : bool, optional
        If True, the function returns a history of the parameters during the
        optimization process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformations. If False (default), copies of the transformations are created
        and modified internally to perform the optimization, leaving the input transformations
        unchanged.


    Returns
    -------
    parameters : Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        A tuple containing the optimized parameters of the intrinsic, distortion, and
        extrinsic transformations with shapes (n_intrinsic_params,), (n_distortion_params,),
        and (n_extrinsic_params,) respectively. Each array contains both the optimized
        parameters (corresponding to True values in the respective masks) and the fixed
        parameters (corresponding to False values in the respective masks), where the fixed
        parameters are equal to their initial values.

    history : List[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]], optional
        A history of the optimization process including the parameters of the intrinsic,
        distortion, and extrinsic transformations with shapes (n_intrinsic_params,),
        (n_distortion_params,), and (n_extrinsic_params,) respectively. Returned only if
        `return_history` is True.


    See Also
    --------
    pycvcam.optimize.optimize_parameters_gn
        Optimize the parameters of a transformation using the Gauss-Newton method.

    pycvcam.optimize.optimize_camera_trf
        Optimize the parameters of a camera transformation using the least squares
        method with the Trust Region Reflective algorithm.

    pycvcam.optimize.optimize_chains_gn
        Optimize the parameters of a set of transformations organized in chains using
        the Gauss-Newton method.

    """
    # -------------
    # Build the composite transformation
    # -------------
    skip_intrinsic = False
    skip_distortion = False
    skip_extrinsic = False
    if intrinsic is None:
        skip_intrinsic = True
        intrinsic = NoIntrinsic()
    if distortion is None:
        skip_distortion = True
        distortion = NoDistortion()
    if extrinsic is None:
        skip_extrinsic = True
        extrinsic = NoExtrinsic()

    if not isinstance(intrinsic, Intrinsic):
        raise TypeError(
            f"intrinsic must be an instance of Intrinsic or None, got {type(intrinsic)}"
        )
    if not isinstance(distortion, Distortion):
        raise TypeError(
            f"distortion must be an instance of Distortion or None, got {type(distortion)}"
        )
    if not isinstance(extrinsic, Extrinsic):
        raise TypeError(
            f"extrinsic must be an instance of Extrinsic or None, got {type(extrinsic)}"
        )

    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")
    if not inplace:
        intrinsic = intrinsic.copy()
        distortion = distortion.copy()
        extrinsic = extrinsic.copy()

    if guess_intrinsic is None and not intrinsic.is_set():
        raise ValueError(
            "Initial guess for the parameters of the intrinsic transformation is required "
            "when the current parameters of the intrinsic transformation are not set."
        )
    elif guess_intrinsic is None and intrinsic.is_set():
        if intrinsic.parameters is None:
            guess_intrinsic = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_intrinsic = intrinsic.parameters.copy()
    else:
        guess_intrinsic = numpy.asarray(guess_intrinsic, dtype=numpy.float64)
    if guess_intrinsic.ndim != 1 or guess_intrinsic.size != intrinsic.n_params:
        raise ValueError(
            f"guess_intrinsic must be a 1D array with {intrinsic.n_params} parameters, "
            f"got {guess_intrinsic.ndim} dimensions and {guess_intrinsic.size} parameters."
        )

    if guess_distortion is None and not distortion.is_set():
        raise ValueError(
            "Initial guess for the parameters of the distortion transformation is required "
            "when the current parameters of the distortion transformation are not set."
        )
    elif guess_distortion is None and distortion.is_set():
        if distortion.parameters is None:
            guess_distortion = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_distortion = distortion.parameters.copy()
    else:
        guess_distortion = numpy.asarray(guess_distortion, dtype=numpy.float64)
    if guess_distortion.ndim != 1 or guess_distortion.size != distortion.n_params:
        raise ValueError(
            f"guess_distortion must be a 1D array with {distortion.n_params} parameters, "
            f"got {guess_distortion.ndim} dimensions and {guess_distortion.size} parameters."
        )

    if guess_extrinsic is None and not extrinsic.is_set():
        raise ValueError(
            "Initial guess for the parameters of the extrinsic transformation is required "
            "when the current parameters of the extrinsic transformation are not set."
        )
    elif guess_extrinsic is None and extrinsic.is_set():
        if extrinsic.parameters is None:
            guess_extrinsic = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_extrinsic = extrinsic.parameters.copy()
    else:
        guess_extrinsic = numpy.asarray(guess_extrinsic, dtype=numpy.float64)
    if guess_extrinsic.ndim != 1 or guess_extrinsic.size != extrinsic.n_params:
        raise ValueError(
            f"guess_extrinsic must be a 1D array with {extrinsic.n_params} parameters, "
            f"got {guess_extrinsic.ndim} dimensions and {guess_extrinsic.size} parameters."
        )

    if mask_intrinsic is None:
        mask_intrinsic = numpy.ones(intrinsic.n_params, dtype=bool)
    else:
        mask_intrinsic = numpy.asarray(mask_intrinsic, dtype=bool)
    if mask_intrinsic.ndim != 1 or mask_intrinsic.size != intrinsic.n_params:
        raise ValueError(
            f"mask_intrinsic must be a 1D array with {intrinsic.n_params} parameters, "
            f"got {mask_intrinsic.ndim} dimensions and {mask_intrinsic.size} parameters."
        )

    if mask_distortion is None:
        mask_distortion = numpy.ones(distortion.n_params, dtype=bool)
    else:
        mask_distortion = numpy.asarray(mask_distortion, dtype=bool)
    if mask_distortion.ndim != 1 or mask_distortion.size != distortion.n_params:
        raise ValueError(
            f"mask_distortion must be a 1D array with {distortion.n_params} parameters, "
            f"got {mask_distortion.ndim} dimensions and {mask_distortion.size} parameters."
        )

    if mask_extrinsic is None:
        mask_extrinsic = numpy.ones(extrinsic.n_params, dtype=bool)
    else:
        mask_extrinsic = numpy.asarray(mask_extrinsic, dtype=bool)
    if mask_extrinsic.ndim != 1 or mask_extrinsic.size != extrinsic.n_params:
        raise ValueError(
            f"mask_extrinsic must be a 1D array with {extrinsic.n_params} parameters, "
            f"got {mask_extrinsic.ndim} dimensions and {mask_extrinsic.size} parameters."
        )

    if intrinsic_kwargs is None:
        intrinsic_kwargs = {}
    if not isinstance(intrinsic_kwargs, dict):
        raise TypeError(
            f"intrinsic_kwargs must be a dictionary, got {type(intrinsic_kwargs)}"
        )

    if distortion_kwargs is None:
        distortion_kwargs = {}
    if not isinstance(distortion_kwargs, dict):
        raise TypeError(
            f"distortion_kwargs must be a dictionary, got {type(distortion_kwargs)}"
        )

    if extrinsic_kwargs is None:
        extrinsic_kwargs = {}
    if not isinstance(extrinsic_kwargs, dict):
        raise TypeError(
            f"extrinsic_kwargs must be a dictionary, got {type(extrinsic_kwargs)}"
        )

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if extrinsic.n_params + distortion.n_params + intrinsic.n_params == 0:
        if verbose_level >= 1:
            print(
                "The transformation has no parameters to optimize, returning the current parameters."
            )
        i_out = guess_intrinsic.copy() if not skip_intrinsic else None
        d_out = guess_distortion.copy() if not skip_distortion else None
        e_out = guess_extrinsic.copy() if not skip_extrinsic else None
        out = (i_out, d_out, e_out)
        if return_history:
            return out, [out]
        else:
            return out

    if not any(mask_intrinsic) and not any(mask_distortion) and not any(mask_extrinsic):
        if verbose_level >= 1:
            print(
                "No parameters to optimize (all parameters are fixed), returning the current parameters."
            )
        i_out = guess_intrinsic.copy() if not skip_intrinsic else None
        d_out = guess_distortion.copy() if not skip_distortion else None
        e_out = guess_extrinsic.copy() if not skip_extrinsic else None
        out = (i_out, d_out, e_out)
        if return_history:
            return out, [out]
        else:
            return out

    # -------------
    # Optimization
    # -------------
    _pretext = None
    if verbose_level >= 3:
        _pretext = ""
        n_pi = mask_intrinsic.sum()
        n_pd = mask_distortion.sum()
        n_pe = mask_extrinsic.sum()
        if n_pi > 0:
            _pretext += f"{n_pi} Intrinsic parameters to optimize - Parameters 0 to {n_pi - 1}\n"
        if n_pd > 0:
            _pretext += f"{n_pd} Distortion parameters to optimize - Parameters {n_pi} to {n_pi+ n_pd - 1}\n"
        if n_pe > 0:
            _pretext += f"{n_pe} Extrinsic parameters to optimize - Parameters {n_pi + n_pd} to {n_pi + n_pd + n_pe - 1}"

    seq_transforms = (intrinsic, distortion, extrinsic)
    seq_chains = ((2, 1, 0),)
    seq_inputs = (world_points,)
    seq_outputs = (image_points,)
    seq_masks = (mask_intrinsic, mask_distortion, mask_extrinsic)
    seq_guesses = (guess_intrinsic, guess_distortion, guess_extrinsic)
    seq_transform_kwargs = (intrinsic_kwargs, distortion_kwargs, extrinsic_kwargs)

    out = _solve_optimize_chains_gauss_newton(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_history=return_history,
        _pretext=_pretext,
        _sparse=False,
    )

    return out


def optimize_camera_trf(
    intrinsic: Optional[Intrinsic],
    distortion: Optional[Distortion],
    extrinsic: Optional[Extrinsic],
    world_points: ArrayLike,
    image_points: ArrayLike,
    *,
    guess_intrinsic: Optional[ArrayLike] = None,
    guess_distortion: Optional[ArrayLike] = None,
    guess_extrinsic: Optional[ArrayLike] = None,
    mask_intrinsic: Optional[ArrayLike] = None,
    mask_distortion: Optional[ArrayLike] = None,
    mask_extrinsic: Optional[ArrayLike] = None,
    scale_intrinsic: Optional[ArrayLike] = None,
    scale_distortion: Optional[ArrayLike] = None,
    scale_extrinsic: Optional[ArrayLike] = None,
    bounds_intrinsic: Optional[ArrayLike] = None,
    bounds_distortion: Optional[ArrayLike] = None,
    bounds_extrinsic: Optional[ArrayLike] = None,
    intrinsic_kwargs: Optional[Dict] = None,
    distortion_kwargs: Optional[Dict] = None,
    extrinsic_kwargs: Optional[Dict] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    loss: Optional[str] = None,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_result: bool = False,
    return_history: bool = False,
    inplace: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Optimize the parameters of the intrinsic, distortion, and extrinsic transformations
    of a camera model such that the projection of the world points matches the image
    points using the ``scipy.optimize.least_squares`` method.
    The computation is done with Trust Region Reflective algorithm.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.

    Parameters
    ----------
    intrinsic : Optional[Intrinsic]
        The intrinsic transformation of the camera model to be optimized. If None, the
        intrinsic transformation is not included in the optimization.

    distortion : Optional[Distortion]
        The distortion transformation of the camera model to be optimized. If None, the
        distortion transformation is not included in the optimization.

    extrinsic : Optional[Extrinsic]
        The extrinsic transformation of the camera model to be optimized. If None, the
        extrinsic transformation is not included in the optimization.

    world_points : ArrayLike
        The world points with shape (..., 3) such that their projection is expected to
        match the image points.

    image_points : ArrayLike
        The image points to be matched with shape (..., 2).

    guess_intrinsic : Optional[ArrayLike], optional
        The initial guess for the parameters of the intrinsic transformation with shape
        (n_intrinsic_params,). If None, the current parameters of the intrinsic
        transformation are used. Default is None.

    guess_distortion : Optional[ArrayLike], optional
        The initial guess for the parameters of the distortion transformation with shape
        (n_distortion_params,). If None, the current parameters of the distortion
        transformation are used. Default is None.

    guess_extrinsic : Optional[ArrayLike], optional
        The initial guess for the parameters of the extrinsic transformation with shape
        (n_extrinsic_params,). If None, the current parameters of the extrinsic
        transformation are used. Default is None.

    mask_intrinsic : Optional[ArrayLike], optional
        A mask array of shape (n_intrinsic_params,) indicating which parameters of the
        intrinsic transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the intrinsic transformation are optimized.

    mask_distortion : Optional[ArrayLike], optional
        A mask array of shape (n_distortion_params,) indicating which parameters of the
        distortion transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the distortion transformation are optimized.

    mask_extrinsic : Optional[ArrayLike], optional
        A mask array of shape (n_extrinsic_params,) indicating which parameters of the
        extrinsic transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the extrinsic transformation are optimized.

    scale_intrinsic : Optional[ArrayLike], optional
        An array of shape (n_intrinsic_params,) indicating the scale of each parameter
        of the intrinsic transformation for the optimization. This is used to improve
        the convergence of the optimization by scaling the parameters to a similar
        range. Default is None, which means no scaling is applied (i.e., all parameters
        of the intrinsic transformation are scaled to 1).

    scale_distortion : Optional[ArrayLike], optional
        An array of shape (n_distortion_params,) indicating the scale of each parameter
        of the distortion transformation for the optimization. This is used to improve
        the convergence of the optimization by scaling the parameters to a similar
        range. Default is None, which means no scaling is applied (i.e., all parameters
        of the distortion transformation are scaled to 1).

    scale_extrinsic : Optional[ArrayLike], optional
        An array of shape (n_extrinsic_params,) indicating the scale of each parameter
        of the extrinsic transformation for the optimization. This is used to improve
        the convergence of the optimization by scaling the parameters to a similar
        range. Default is None, which means no scaling is applied (i.e., all parameters
        of the extrinsic transformation are scaled to 1).

    bounds_intrinsic : Optional[ArrayLike], optional
        The bounds for the parameters of the intrinsic transformation with shape
        (2, n_intrinsic_params). The first row contains the lower bounds and the second
        row contains the upper bounds for each parameter of the intrinsic transformation.
        Default is None, which means no bounds are applied to the parameters of the
        intrinsic transformation. Set ``+/- numpy.inf`` for no bound on a specific
        parameter.

    bounds_distortion : Optional[ArrayLike], optional
        The bounds for the parameters of the distortion transformation with shape
        (2, n_distortion_params). The first row contains the lower bounds and the second
        row contains the upper bounds for each parameter of the distortion
        transformation. Default is None, which means no bounds are applied to the
        parameters of the distortion transformation. Set ``+/- numpy.inf`` for no bound
        on a specific parameter.

    bounds_extrinsic : Optional[ArrayLike], optional
        The bounds for the parameters of the extrinsic transformation with shape
        (2, n_extrinsic_params). The first row contains the lower bounds and the second
        row contains the upper bounds for each parameter of the extrinsic transformation.
        Default is None, which means no bounds are applied to the parameters of the
        extrinsic transformation. Set ``+/- numpy.inf`` for no bound on a specific
        parameter.

    intrinsic_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``intrinsic._transform`` method. Default is
        None, which means no additional keyword arguments are passed to the intrinsic
        transformation.

    distortion_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``distortion._transform`` method. Default
        is None, which means no additional keyword arguments are passed to the
        distortion transformation.

    extrinsic_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``extrinsic._transform`` method. Default is
        None, which means no additional keyword arguments are passed to the extrinsic
        transformation.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations.
        The optimization process is stopped when the number of iterations
        exceeds ``max_iterations``. Default is None, which means no limit on the
        number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``.
        The default value is None, which means the ftol criterion is not used for
        stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters.
        The optimization process is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        The default value is None, which means the xtol criterion is not used for
        stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient.
        The optimization process is stopped when ``norm(g_scaled, ord=numpy.inf) < gtol``
        where g_scaled is the value of the gradient scaled to account for the
        presence of the bounds. The default value is None, which means the gtol
        criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set to
        ``1e-8``. If any of the stopping criteria is already specified, it will
        be overridden by the user-specified value.

    loss : Optional[str], optional
        If specified, the optimization will use a robust loss function to reduce the
        influence of outliers in the optimization. The available loss functions are
        'huber', 'cauchy', 'arctan', 'soft_l1', and 'linear'. Default is None, which
        means the standard least squares loss is used (i.e., 'linear').

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_result : bool, optional
        If True, the function returns the ``scipy.optimize.OptimizeResult`` object
        containing information about the convergence of the optimization process.
        Default is False, which means only the optimized parameters are returned.
        If all transformations are None, or all parameters of the transformations are
        masked, the result output will be None.

    return_history : bool, optional
        If True, the function returns a history of the parameters during
        the optimization process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformations. If False (default), copies of the transformations are created
        and modified internally to perform the optimization, leaving the input transformations
        unchanged.


    Returns
    -------
    parameters : Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        A tuple containing the optimized parameters of the intrinsic, distortion, and
        extrinsic transformations with shapes (n_intrinsic_params,), (n_distortion_params,),
        and (n_extrinsic_params,) respectively. Each array contains both the optimized
        parameters (corresponding to True values in the respective masks) and the fixed
        parameters (corresponding to False values in the respective masks), where the fixed
        parameters are equal to their initial values.

    result : scipy.optimize.OptimizeResult, optional
        The result of the optimization process containing information about the
        convergence of the optimization. Returned only if `return_result` is True and
        at least one of the transformations is not None and has at least one parameter
        to optimize, otherwise None.

        .. warning::

            Only contains the parameters that were optimized (i.e., the parameters
            corresponding to True values in the `mask_intrinsic`, `mask_distortion`, and
            `mask_extrinsic`), and not the full parameter vectors of the transformations.

    history : List[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]], optional
        A history of the optimization process including the parameters of the intrinsic,
        distortion, and extrinsic transformations with shapes (n_intrinsic_params,),
        (n_distortion_params,), and (n_extrinsic_params,) respectively. Returned only if
        `return_history` is True.


    See Also
    --------
    pycvcam.optimize.optimize_parameters_trf
        Optimize the parameters of a transformation using the least squares method with
        the Trust Region Reflective algorithm.

    pycvcam.optimize.optimize_camera_gn
        Optimize the parameters of a camera transformation using the Gauss-Newton method.

    pycvcam.optimize.optimize_chains_trf
        Optimize the parameters of a set of transformations organized in chains using the
        least squares method with the Trust Region Reflective algorithm.

    """
    # -------------
    # Build the composite transformation
    # -------------
    skip_intrinsic = False
    skip_distortion = False
    skip_extrinsic = False
    if intrinsic is None:
        skip_intrinsic = True
        intrinsic = NoIntrinsic()
    if distortion is None:
        skip_distortion = True
        distortion = NoDistortion()
    if extrinsic is None:
        skip_extrinsic = True
        extrinsic = NoExtrinsic()

    if not isinstance(intrinsic, Intrinsic):
        raise TypeError(
            f"intrinsic must be an instance of Intrinsic or None, got {type(intrinsic)}"
        )
    if not isinstance(distortion, Distortion):
        raise TypeError(
            f"distortion must be an instance of Distortion or None, got {type(distortion)}"
        )
    if not isinstance(extrinsic, Extrinsic):
        raise TypeError(
            f"extrinsic must be an instance of Extrinsic or None, got {type(extrinsic)}"
        )

    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")
    if not inplace:
        intrinsic = intrinsic.copy()
        distortion = distortion.copy()
        extrinsic = extrinsic.copy()

    if guess_intrinsic is None and not intrinsic.is_set():
        raise ValueError(
            "Initial guess for the parameters of the intrinsic transformation is required "
            "when the current parameters of the intrinsic transformation are not set."
        )
    elif guess_intrinsic is None and intrinsic.is_set():
        if intrinsic.parameters is None:
            guess_intrinsic = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_intrinsic = intrinsic.parameters.copy()
    else:
        guess_intrinsic = numpy.asarray(guess_intrinsic, dtype=numpy.float64)
    if guess_intrinsic.ndim != 1 or guess_intrinsic.size != intrinsic.n_params:
        raise ValueError(
            f"guess_intrinsic must be a 1D array with {intrinsic.n_params} parameters, "
            f"got {guess_intrinsic.ndim} dimensions and {guess_intrinsic.size} parameters."
        )

    if guess_distortion is None and not distortion.is_set():
        raise ValueError(
            "Initial guess for the parameters of the distortion transformation is required "
            "when the current parameters of the distortion transformation are not set."
        )
    elif guess_distortion is None and distortion.is_set():
        if distortion.parameters is None:
            guess_distortion = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_distortion = distortion.parameters.copy()
    else:
        guess_distortion = numpy.asarray(guess_distortion, dtype=numpy.float64)
    if guess_distortion.ndim != 1 or guess_distortion.size != distortion.n_params:
        raise ValueError(
            f"guess_distortion must be a 1D array with {distortion.n_params} parameters, "
            f"got {guess_distortion.ndim} dimensions and {guess_distortion.size} parameters."
        )

    if guess_extrinsic is None and not extrinsic.is_set():
        raise ValueError(
            "Initial guess for the parameters of the extrinsic transformation is required "
            "when the current parameters of the extrinsic transformation are not set."
        )
    elif guess_extrinsic is None and extrinsic.is_set():
        if extrinsic.parameters is None:
            guess_extrinsic = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_extrinsic = extrinsic.parameters.copy()
    else:
        guess_extrinsic = numpy.asarray(guess_extrinsic, dtype=numpy.float64)
    if guess_extrinsic.ndim != 1 or guess_extrinsic.size != extrinsic.n_params:
        raise ValueError(
            f"guess_extrinsic must be a 1D array with {extrinsic.n_params} parameters, "
            f"got {guess_extrinsic.ndim} dimensions and {guess_extrinsic.size} parameters."
        )

    if mask_intrinsic is None:
        mask_intrinsic = numpy.ones(intrinsic.n_params, dtype=bool)
    else:
        mask_intrinsic = numpy.asarray(mask_intrinsic, dtype=bool)
    if mask_intrinsic.ndim != 1 or mask_intrinsic.size != intrinsic.n_params:
        raise ValueError(
            f"mask_intrinsic must be a 1D array with {intrinsic.n_params} parameters, "
            f"got {mask_intrinsic.ndim} dimensions and {mask_intrinsic.size} parameters."
        )

    if mask_distortion is None:
        mask_distortion = numpy.ones(distortion.n_params, dtype=bool)
    else:
        mask_distortion = numpy.asarray(mask_distortion, dtype=bool)
    if mask_distortion.ndim != 1 or mask_distortion.size != distortion.n_params:
        raise ValueError(
            f"mask_distortion must be a 1D array with {distortion.n_params} parameters, "
            f"got {mask_distortion.ndim} dimensions and {mask_distortion.size} parameters."
        )

    if mask_extrinsic is None:
        mask_extrinsic = numpy.ones(extrinsic.n_params, dtype=bool)
    else:
        mask_extrinsic = numpy.asarray(mask_extrinsic, dtype=bool)
    if mask_extrinsic.ndim != 1 or mask_extrinsic.size != extrinsic.n_params:
        raise ValueError(
            f"mask_extrinsic must be a 1D array with {extrinsic.n_params} parameters, "
            f"got {mask_extrinsic.ndim} dimensions and {mask_extrinsic.size} parameters."
        )

    if scale_intrinsic is None:
        scale_intrinsic = numpy.ones(intrinsic.n_params, dtype=numpy.float64)
    else:
        scale_intrinsic = numpy.asarray(scale_intrinsic, dtype=numpy.float64)
    if scale_intrinsic.ndim != 1 or scale_intrinsic.size != intrinsic.n_params:
        raise ValueError(
            f"scale_intrinsic must be a 1D array with {intrinsic.n_params} parameters, "
            f"got {scale_intrinsic.ndim} dimensions and {scale_intrinsic.size} parameters."
        )

    if scale_distortion is None:
        scale_distortion = numpy.ones(distortion.n_params, dtype=numpy.float64)
    else:
        scale_distortion = numpy.asarray(scale_distortion, dtype=numpy.float64)
    if scale_distortion.ndim != 1 or scale_distortion.size != distortion.n_params:
        raise ValueError(
            f"scale_distortion must be a 1D array with {distortion.n_params} parameters, "
            f"got {scale_distortion.ndim} dimensions and {scale_distortion.size} parameters."
        )

    if scale_extrinsic is None:
        scale_extrinsic = numpy.ones(extrinsic.n_params, dtype=numpy.float64)
    else:
        scale_extrinsic = numpy.asarray(scale_extrinsic, dtype=numpy.float64)
    if scale_extrinsic.ndim != 1 or scale_extrinsic.size != extrinsic.n_params:
        raise ValueError(
            f"scale_extrinsic must be a 1D array with {extrinsic.n_params} parameters, "
            f"got {scale_extrinsic.ndim} dimensions and {scale_extrinsic.size} parameters."
        )

    if bounds_intrinsic is None:
        bounds_intrinsic = numpy.array(
            [[-numpy.inf] * intrinsic.n_params, [numpy.inf] * intrinsic.n_params],
            dtype=numpy.float64,
        )
    else:
        bounds_intrinsic = numpy.asarray(bounds_intrinsic, dtype=numpy.float64)
    if bounds_intrinsic.ndim != 2 or bounds_intrinsic.shape != (2, intrinsic.n_params):
        raise ValueError(
            f"bounds_intrinsic must be a 2D array with shape (2, {intrinsic.n_params}), "
            f"got {bounds_intrinsic.ndim} dimensions and shape {bounds_intrinsic.shape}."
        )

    if bounds_distortion is None:
        bounds_distortion = numpy.array(
            [[-numpy.inf] * distortion.n_params, [numpy.inf] * distortion.n_params],
            dtype=numpy.float64,
        )
    else:
        bounds_distortion = numpy.asarray(bounds_distortion, dtype=numpy.float64)
    if bounds_distortion.ndim != 2 or bounds_distortion.shape != (
        2,
        distortion.n_params,
    ):
        raise ValueError(
            f"bounds_distortion must be a 2D array with shape (2, {distortion.n_params}), "
            f"got {bounds_distortion.ndim} dimensions and shape {bounds_distortion.shape}."
        )

    if bounds_extrinsic is None:
        bounds_extrinsic = numpy.array(
            [[-numpy.inf] * extrinsic.n_params, [numpy.inf] * extrinsic.n_params],
            dtype=numpy.float64,
        )
    else:
        bounds_extrinsic = numpy.asarray(bounds_extrinsic, dtype=numpy.float64)
    if bounds_extrinsic.ndim != 2 or bounds_extrinsic.shape != (2, extrinsic.n_params):
        raise ValueError(
            f"bounds_extrinsic must be a 2D array with shape (2, {extrinsic.n_params}), "
            f"got {bounds_extrinsic.ndim} dimensions and shape {bounds_extrinsic.shape}."
        )

    if intrinsic_kwargs is None:
        intrinsic_kwargs = {}
    if not isinstance(intrinsic_kwargs, dict):
        raise TypeError(
            f"intrinsic_kwargs must be a dictionary, got {type(intrinsic_kwargs)}"
        )

    if distortion_kwargs is None:
        distortion_kwargs = {}
    if not isinstance(distortion_kwargs, dict):
        raise TypeError(
            f"distortion_kwargs must be a dictionary, got {type(distortion_kwargs)}"
        )

    if extrinsic_kwargs is None:
        extrinsic_kwargs = {}
    if not isinstance(extrinsic_kwargs, dict):
        raise TypeError(
            f"extrinsic_kwargs must be a dictionary, got {type(extrinsic_kwargs)}"
        )

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if loss is not None:
        if not isinstance(loss, str) or loss not in [
            "huber",
            "cauchy",
            "arctan",
            "soft_l1",
            "linear",
        ]:
            raise ValueError(
                f"loss must be one of 'huber', 'cauchy', 'arctan', 'soft_l1', or 'linear', got {loss}"
            )
    else:
        loss = "linear"

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_result, bool):
        raise TypeError(f"return_result must be a boolean, got {type(return_result)}")

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if extrinsic.n_params + distortion.n_params + intrinsic.n_params == 0:
        if verbose_level >= 1:
            print(
                "The transformation has no parameters to optimize, returning the current parameters."
            )
        i_out = guess_intrinsic.copy() if not skip_intrinsic else None
        d_out = guess_distortion.copy() if not skip_distortion else None
        e_out = guess_extrinsic.copy() if not skip_extrinsic else None
        out = (i_out, d_out, e_out)
        if return_history and return_result:
            return out, None, [out]
        elif return_history and not return_result:
            return out, [out]
        elif not return_history and return_result:
            return out, None
        else:
            return out

    if not any(mask_intrinsic) and not any(mask_distortion) and not any(mask_extrinsic):
        if verbose_level >= 1:
            print(
                "No parameters to optimize (all parameters are fixed), returning the current parameters."
            )
        i_out = guess_intrinsic.copy() if not skip_intrinsic else None
        d_out = guess_distortion.copy() if not skip_distortion else None
        e_out = guess_extrinsic.copy() if not skip_extrinsic else None
        out = (i_out, d_out, e_out)
        if return_history and return_result:
            return out, None, [out]
        elif return_history and not return_result:
            return out, [out]
        elif not return_history and return_result:
            return out, None
        else:
            return out

    # -------------
    # Optimization
    # -------------
    _pretext = None
    if verbose_level >= 3:
        _pretext = ""
        n_pi = mask_intrinsic.sum()
        n_pd = mask_distortion.sum()
        n_pe = mask_extrinsic.sum()
        if n_pi > 0:
            _pretext += f"{n_pi} Intrinsic parameters to optimize - Parameters 0 to {n_pi - 1}\n"
        if n_pd > 0:
            _pretext += f"{n_pd} Distortion parameters to optimize - Parameters {n_pi} to {n_pi+ n_pd - 1}\n"
        if n_pe > 0:
            _pretext += f"{n_pe} Extrinsic parameters to optimize - Parameters {n_pi + n_pd} to {n_pi + n_pd + n_pe - 1}"

    seq_transforms = (intrinsic, distortion, extrinsic)
    seq_chains = ((2, 1, 0),)
    seq_inputs = (world_points,)
    seq_outputs = (image_points,)
    seq_masks = (mask_intrinsic, mask_distortion, mask_extrinsic)
    seq_guesses = (guess_intrinsic, guess_distortion, guess_extrinsic)
    seq_scales = (scale_intrinsic, scale_distortion, scale_extrinsic)
    seq_bounds = (bounds_intrinsic, bounds_distortion, bounds_extrinsic)
    seq_transform_kwargs = (intrinsic_kwargs, distortion_kwargs, extrinsic_kwargs)

    out = _solve_optimize_chains_trf_scipy(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_scales=seq_scales,
        seq_bounds=seq_bounds,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        loss=loss,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_result=return_result,
        return_history=return_history,
        _pretext=_pretext,
        _sparse=False,
    )

    return out


def optimize_camera_lm(
    intrinsic: Optional[Intrinsic],
    distortion: Optional[Distortion],
    extrinsic: Optional[Extrinsic],
    world_points: ArrayLike,
    image_points: ArrayLike,
    *,
    guess_intrinsic: Optional[ArrayLike] = None,
    guess_distortion: Optional[ArrayLike] = None,
    guess_extrinsic: Optional[ArrayLike] = None,
    mask_intrinsic: Optional[ArrayLike] = None,
    mask_distortion: Optional[ArrayLike] = None,
    mask_extrinsic: Optional[ArrayLike] = None,
    intrinsic_kwargs: Optional[Dict] = None,
    distortion_kwargs: Optional[Dict] = None,
    extrinsic_kwargs: Optional[Dict] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    loss: Optional[str] = None,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_result: bool = False,
    return_history: bool = False,
    inplace: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Optimize the parameters of the intrinsic, distortion, and extrinsic transformations
    of a camera model such that the projection of the world points matches the image
    points using the ``scipy.optimize.least_squares`` method.
    The computation is done with the Levenberg-Marquardt algorithm.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.

    Parameters
    ----------
    intrinsic : Optional[Intrinsic]
        The intrinsic transformation of the camera model to be optimized. If None, the
        intrinsic transformation is not included in the optimization.

    distortion : Optional[Distortion]
        The distortion transformation of the camera model to be optimized. If None, the
        distortion transformation is not included in the optimization.

    extrinsic : Optional[Extrinsic]
        The extrinsic transformation of the camera model to be optimized. If None, the
        extrinsic transformation is not included in the optimization.

    world_points : ArrayLike
        The world points with shape (..., 3) such that their projection is expected to
        match the image points.

    image_points : ArrayLike
        The image points to be matched with shape (..., 2).

    guess_intrinsic : Optional[ArrayLike], optional
        The initial guess for the parameters of the intrinsic transformation with shape
        (n_intrinsic_params,). If None, the current parameters of the intrinsic
        transformation are used. Default is None.

    guess_distortion : Optional[ArrayLike], optional
        The initial guess for the parameters of the distortion transformation with shape
        (n_distortion_params,). If None, the current parameters of the distortion
        transformation are used. Default is None.

    guess_extrinsic : Optional[ArrayLike], optional
        The initial guess for the parameters of the extrinsic transformation with shape
        (n_extrinsic_params,). If None, the current parameters of the extrinsic
        transformation are used. Default is None.

    mask_intrinsic : Optional[ArrayLike], optional
        A mask array of shape (n_intrinsic_params,) indicating which parameters of the
        intrinsic transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the intrinsic transformation are optimized.

    mask_distortion : Optional[ArrayLike], optional
        A mask array of shape (n_distortion_params,) indicating which parameters of the
        distortion transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the distortion transformation are optimized.

    mask_extrinsic : Optional[ArrayLike], optional
        A mask array of shape (n_extrinsic_params,) indicating which parameters of the
        extrinsic transformation should be optimized. Elements with a value of True are
        optimized, while elements with a value of False are kept fixed. Default is None,
        which means all parameters of the extrinsic transformation are optimized.

    intrinsic_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``intrinsic._transform`` method. Default is
        None, which means no additional keyword arguments are passed to the intrinsic
        transformation.

    distortion_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``distortion._transform`` method. Default
        is None, which means no additional keyword arguments are passed to the
        distortion transformation.

    extrinsic_kwargs : Optional[Dict], optional
        Additional keyword arguments for the ``extrinsic._transform`` method. Default is
        None, which means no additional keyword arguments are passed to the extrinsic
        transformation.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations.
        The optimization process is stopped when the number of iterations
        exceeds ``max_iterations``. Default is None, which means no limit on the
        number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``.
        The default value is None, which means the ftol criterion is not used for
        stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters.
        The optimization process is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        The default value is None, which means the xtol criterion is not used for
        stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient.
        The optimization process is stopped when ``norm(g_scaled, ord=numpy.inf) < gtol``
        where g_scaled is the value of the gradient scaled to account for the
        presence of the bounds. The default value is None, which means the gtol
        criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set to
        ``1e-8``. If any of the stopping criteria is already specified, it will
        be overridden by the user-specified value.

    loss : Optional[str], optional
        If specified, the optimization will use a robust loss function to reduce the
        influence of outliers in the optimization. The available loss functions are
        'huber', 'cauchy', 'arctan', 'soft_l1', and 'linear'. Default is None, which
        means the standard least squares loss is used (i.e., 'linear').

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_result : bool, optional
        If True, the function returns the ``scipy.optimize.OptimizeResult`` object
        containing information about the convergence of the optimization process.
        Default is False, which means only the optimized parameters are returned.
        If all transformations are None, or all parameters of the transformations are
        masked, the result output will be None.

    return_history : bool, optional
        If True, the function returns a history of the parameters during
        the optimization process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformations. If False (default), copies of the transformations are created
        and modified internally to perform the optimization, leaving the input transformations
        unchanged.


    Returns
    -------
    parameters : Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        A tuple containing the optimized parameters of the intrinsic, distortion, and
        extrinsic transformations with shapes (n_intrinsic_params,), (n_distortion_params,),
        and (n_extrinsic_params,) respectively. Each array contains both the optimized
        parameters (corresponding to True values in the respective masks) and the fixed
        parameters (corresponding to False values in the respective masks), where the fixed
        parameters are equal to their initial values.

    result : scipy.optimize.OptimizeResult, optional
        The result of the optimization process containing information about the
        convergence of the optimization. Returned only if `return_result` is True and
        at least one of the transformations is not None and has at least one parameter
        to optimize, otherwise None.

        .. warning::

            Only contains the parameters that were optimized (i.e., the parameters
            corresponding to True values in the `mask_intrinsic`, `mask_distortion`, and
            `mask_extrinsic`), and not the full parameter vectors of the transformations.

    history : List[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]], optional
        A history of the optimization process including the parameters of the intrinsic,
        distortion, and extrinsic transformations with shapes (n_intrinsic_params,),
        (n_distortion_params,), and (n_extrinsic_params,) respectively. Returned only if
        `return_history` is True.


    See Also
    --------
    pycvcam.optimize.optimize_parameters_lm
        Optimize the parameters of a transformation using the least squares method with
        the Levenberg-Marquardt algorithm.

    pycvcam.optimize.optimize_camera_gn
        Optimize the parameters of a camera transformation using the Gauss-Newton method.

    pycvcam.optimize.optimize_chains_lm
        Optimize the parameters of a set of transformations organized in chains using the
        least squares method with the Levenberg-Marquardt algorithm.

    """
    # -------------
    # Build the composite transformation
    # -------------
    skip_intrinsic = False
    skip_distortion = False
    skip_extrinsic = False
    if intrinsic is None:
        skip_intrinsic = True
        intrinsic = NoIntrinsic()
    if distortion is None:
        skip_distortion = True
        distortion = NoDistortion()
    if extrinsic is None:
        skip_extrinsic = True
        extrinsic = NoExtrinsic()

    if not isinstance(intrinsic, Intrinsic):
        raise TypeError(
            f"intrinsic must be an instance of Intrinsic or None, got {type(intrinsic)}"
        )
    if not isinstance(distortion, Distortion):
        raise TypeError(
            f"distortion must be an instance of Distortion or None, got {type(distortion)}"
        )
    if not isinstance(extrinsic, Extrinsic):
        raise TypeError(
            f"extrinsic must be an instance of Extrinsic or None, got {type(extrinsic)}"
        )

    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")
    if not inplace:
        intrinsic = intrinsic.copy()
        distortion = distortion.copy()
        extrinsic = extrinsic.copy()

    if guess_intrinsic is None and not intrinsic.is_set():
        raise ValueError(
            "Initial guess for the parameters of the intrinsic transformation is required "
            "when the current parameters of the intrinsic transformation are not set."
        )
    elif guess_intrinsic is None and intrinsic.is_set():
        if intrinsic.parameters is None:
            guess_intrinsic = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_intrinsic = intrinsic.parameters.copy()
    else:
        guess_intrinsic = numpy.asarray(guess_intrinsic, dtype=numpy.float64)
    if guess_intrinsic.ndim != 1 or guess_intrinsic.size != intrinsic.n_params:
        raise ValueError(
            f"guess_intrinsic must be a 1D array with {intrinsic.n_params} parameters, "
            f"got {guess_intrinsic.ndim} dimensions and {guess_intrinsic.size} parameters."
        )

    if guess_distortion is None and not distortion.is_set():
        raise ValueError(
            "Initial guess for the parameters of the distortion transformation is required "
            "when the current parameters of the distortion transformation are not set."
        )
    elif guess_distortion is None and distortion.is_set():
        if distortion.parameters is None:
            guess_distortion = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_distortion = distortion.parameters.copy()
    else:
        guess_distortion = numpy.asarray(guess_distortion, dtype=numpy.float64)
    if guess_distortion.ndim != 1 or guess_distortion.size != distortion.n_params:
        raise ValueError(
            f"guess_distortion must be a 1D array with {distortion.n_params} parameters, "
            f"got {guess_distortion.ndim} dimensions and {guess_distortion.size} parameters."
        )

    if guess_extrinsic is None and not extrinsic.is_set():
        raise ValueError(
            "Initial guess for the parameters of the extrinsic transformation is required "
            "when the current parameters of the extrinsic transformation are not set."
        )
    elif guess_extrinsic is None and extrinsic.is_set():
        if extrinsic.parameters is None:
            guess_extrinsic = numpy.zeros((0,), dtype=numpy.float64)
        else:
            guess_extrinsic = extrinsic.parameters.copy()
    else:
        guess_extrinsic = numpy.asarray(guess_extrinsic, dtype=numpy.float64)
    if guess_extrinsic.ndim != 1 or guess_extrinsic.size != extrinsic.n_params:
        raise ValueError(
            f"guess_extrinsic must be a 1D array with {extrinsic.n_params} parameters, "
            f"got {guess_extrinsic.ndim} dimensions and {guess_extrinsic.size} parameters."
        )

    if mask_intrinsic is None:
        mask_intrinsic = numpy.ones(intrinsic.n_params, dtype=bool)
    else:
        mask_intrinsic = numpy.asarray(mask_intrinsic, dtype=bool)
    if mask_intrinsic.ndim != 1 or mask_intrinsic.size != intrinsic.n_params:
        raise ValueError(
            f"mask_intrinsic must be a 1D array with {intrinsic.n_params} parameters, "
            f"got {mask_intrinsic.ndim} dimensions and {mask_intrinsic.size} parameters."
        )

    if mask_distortion is None:
        mask_distortion = numpy.ones(distortion.n_params, dtype=bool)
    else:
        mask_distortion = numpy.asarray(mask_distortion, dtype=bool)
    if mask_distortion.ndim != 1 or mask_distortion.size != distortion.n_params:
        raise ValueError(
            f"mask_distortion must be a 1D array with {distortion.n_params} parameters, "
            f"got {mask_distortion.ndim} dimensions and {mask_distortion.size} parameters."
        )

    if mask_extrinsic is None:
        mask_extrinsic = numpy.ones(extrinsic.n_params, dtype=bool)
    else:
        mask_extrinsic = numpy.asarray(mask_extrinsic, dtype=bool)
    if mask_extrinsic.ndim != 1 or mask_extrinsic.size != extrinsic.n_params:
        raise ValueError(
            f"mask_extrinsic must be a 1D array with {extrinsic.n_params} parameters, "
            f"got {mask_extrinsic.ndim} dimensions and {mask_extrinsic.size} parameters."
        )

    if intrinsic_kwargs is None:
        intrinsic_kwargs = {}
    if not isinstance(intrinsic_kwargs, dict):
        raise TypeError(
            f"intrinsic_kwargs must be a dictionary, got {type(intrinsic_kwargs)}"
        )

    if distortion_kwargs is None:
        distortion_kwargs = {}
    if not isinstance(distortion_kwargs, dict):
        raise TypeError(
            f"distortion_kwargs must be a dictionary, got {type(distortion_kwargs)}"
        )

    if extrinsic_kwargs is None:
        extrinsic_kwargs = {}
    if not isinstance(extrinsic_kwargs, dict):
        raise TypeError(
            f"extrinsic_kwargs must be a dictionary, got {type(extrinsic_kwargs)}"
        )

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if loss is not None:
        if not isinstance(loss, str) or loss not in [
            "huber",
            "cauchy",
            "arctan",
            "soft_l1",
            "linear",
        ]:
            raise ValueError(
                f"loss must be one of 'huber', 'cauchy', 'arctan', 'soft_l1', or 'linear', got {loss}"
            )
    else:
        loss = "linear"

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_result, bool):
        raise TypeError(f"return_result must be a boolean, got {type(return_result)}")

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if extrinsic.n_params + distortion.n_params + intrinsic.n_params == 0:
        if verbose_level >= 1:
            print(
                "The transformation has no parameters to optimize, returning the current parameters."
            )
        i_out = guess_intrinsic.copy() if not skip_intrinsic else None
        d_out = guess_distortion.copy() if not skip_distortion else None
        e_out = guess_extrinsic.copy() if not skip_extrinsic else None
        out = (i_out, d_out, e_out)
        if return_history and return_result:
            return out, None, [out]
        elif return_history and not return_result:
            return out, [out]
        elif not return_history and return_result:
            return out, None
        else:
            return out

    if not any(mask_intrinsic) and not any(mask_distortion) and not any(mask_extrinsic):
        if verbose_level >= 1:
            print(
                "No parameters to optimize (all parameters are fixed), returning the current parameters."
            )
        i_out = guess_intrinsic.copy() if not skip_intrinsic else None
        d_out = guess_distortion.copy() if not skip_distortion else None
        e_out = guess_extrinsic.copy() if not skip_extrinsic else None
        out = (i_out, d_out, e_out)
        if return_history and return_result:
            return out, None, [out]
        elif return_history and not return_result:
            return out, [out]
        elif not return_history and return_result:
            return out, None
        else:
            return out

    # -------------
    # Optimization
    # -------------
    _pretext = None
    if verbose_level >= 3:
        _pretext = ""
        n_pi = mask_intrinsic.sum()
        n_pd = mask_distortion.sum()
        n_pe = mask_extrinsic.sum()
        if n_pi > 0:
            _pretext += f"{n_pi} Intrinsic parameters to optimize - Parameters 0 to {n_pi - 1}\n"
        if n_pd > 0:
            _pretext += f"{n_pd} Distortion parameters to optimize - Parameters {n_pi} to {n_pi+ n_pd - 1}\n"
        if n_pe > 0:
            _pretext += f"{n_pe} Extrinsic parameters to optimize - Parameters {n_pi + n_pd} to {n_pi + n_pd + n_pe - 1}"

    seq_transforms = (intrinsic, distortion, extrinsic)
    seq_chains = ((2, 1, 0),)
    seq_inputs = (world_points,)
    seq_outputs = (image_points,)
    seq_masks = (mask_intrinsic, mask_distortion, mask_extrinsic)
    seq_guesses = (guess_intrinsic, guess_distortion, guess_extrinsic)
    seq_transform_kwargs = (intrinsic_kwargs, distortion_kwargs, extrinsic_kwargs)

    out = _solve_optimize_chains_lm_scipy(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        loss=loss,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_result=return_result,
        return_history=return_history,
        _pretext=_pretext,
        _sparse=False,
    )

    return out


def optimize_chains_gn(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[int]],
    seq_inputs: Sequence[ArrayLike],
    seq_outputs: Sequence[ArrayLike],
    *,
    seq_guesses: Optional[Sequence[ArrayLike]] = None,
    seq_masks: Optional[Sequence[ArrayLike]] = None,
    seq_transform_kwargs: Optional[Sequence[Dict]] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_history: bool = False,
    inplace: bool = False,
) -> Tuple[numpy.ndarray, ...]:
    r"""
    Optimize several :class:`Transform` objects according multiple chains of
    transformations using the least squares method with the Gauss-Newton algorithm.

    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a tuple of :math:`N_T`
    :class:`Transform` objects, and :math:`(C_0, C_1, ..., C_{N_C-1})` be a
    tuple of :math:`N_C` chains of transformations.

    A chain :math:`C_i` is defined as a tuple of indices corresponding to the
    transformations in the chain. For example:

    .. code-block:: console

        C_0 = (1, 4, 8) -----> C_0(X) = T_1 ∘ T_4 ∘ T_8(X)

    The optimization process is then defined as:

    .. math::

        \min_{\lambda_0, \lambda_1, ..., \lambda_{N_T-1}} \sum_{i=0}^{N_C-1} \|R_i(\lambda_0, \lambda_1, ..., \lambda_{N_T-1})\|^2

    where :math:`R_i` is the residual vector for the chain :math:`C_i` defined as:

    .. math::

        R_i(\lambda_0, \lambda_1, ..., \lambda_{N_T-1}) = \vec{X}_O - C_i(\vec{X}_I, \lambda_0, \lambda_1, ..., \lambda_{N_T-1})

    .. note::

        This method can be used to optimize the parameters of any transformations that
        implement the `_transform` method.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of :math:`N_T` :class:`Transform` objects to be optimized.
        The ``constants`` attribute of each transformation must be set before calling
        this function. If the ``parameters`` attribute of a transformation is set,
        it will be used as the initial guess for the optimization if the `seq_guesses`
        parameter is None. Note that the input :class:`Transform` objects are not
        modified during the optimization process, a copy of each object is created
        and modified internally to perform the optimization if `inplace` is False.

    seq_chains : Sequence[Sequence[int]]
        A sequence of :math:`N_C` chains of transformations. Each chain is defined as a
        sequence of indices corresponding to the transformations in the chain. Each
        chain must be non-empty and contain valid indices (i.e., integers between 0
        and :math:`N_T-1`).

    seq_inputs : Sequence[ArrayLike]
        A sequence of :math:`N_C` arrays of input points with shape (..., input_dim)
        such that their transformation through the corresponding chain is expected to
        match the output points.

    seq_outputs : Sequence[ArrayLike]
        A sequence of :math:`N_C` arrays of output points to be matched with shape
        (..., output_dim).

    seq_guesses : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of initial guesses for the parameters of each
        transformation with shape (n_params,). If None or if ``seq_guesses[i]`` is None,
        the associated parameters of the transformation ``seq_transforms[i]`` are used.
        Default is None.

    seq_masks : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of masks with shape (n_params,) indicating
        which parameters of each transformation should be optimized. Elements with a
        value of True are optimized, while elements with a value of False are kept
        fixed. If None or if ``seq_masks[i]`` is None, all parameters of the transformation
        ``seq_transforms[i]`` are optimized. Default is None.

    seq_transform_kwargs : Optional[Sequence[Dict]], optional
        A sequence of :math:`N_T` dictionaries of additional keyword arguments for the
        ``_transform`` method of each transformation. If None or if
        ``seq_transform_kwargs[i]`` is None, no additional keyword arguments are passed to
        the transformation ``seq_transforms[i]``. Default is None.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations. The optimization process is stopped
        when the number of iterations exceeds ``max_iterations``. Default is None, which
        means no limit on the number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function. The optimization process is
        stopped when ``dF < ftol * F``. The default value is None, which means the ftol
        criterion is not used for stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters. The optimization process is
        stopped when ``norm(dx) < xtol * (xtol + norm(x))``. The default value is None,
        which means the xtol criterion is not used for stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient. The optimization process is stopped
        when ``norm(g_scaled, ord=numpy.inf) < gtol`` where g_scaled is the value of the
        gradient scaled to account for the presence of the bounds. The default value is
        None, which means the gtol criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set
        to ``1e-8``. If any of the stopping criteria is already specified, it will be
        overridden by the user-specified value.

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_history : bool, optional
        If True, the function returns a history of the parameters during the optimization
        process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformations. If False (default), copies of the transformations are created
        and modified internally to perform the optimization, leaving the input transformations
        unchanged.


    Returns
    -------
    parameters : Tuple[numpy.ndarray, ...]
        A tuple of :math:`N_T` arrays of optimized parameters for each transformation
        with shape (n_params,). Each array contains both the optimized parameters
        (corresponding to True values in the `mask`) and the fixed parameters
        (corresponding to False values in the `mask`), where the fixed parameters are
        equal to their initial values.

    history : List[Tuple[numpy.ndarray, ...]], optional
        A history of the optimization process including the parameters of each
        transformation with shape (n_params,). Returned only if `return_history` is True.


    See Also
    --------
    pycvcam.optimize.optimize_parameters_gn
        Optimize the parameters of a transformation using the least squares method with
        the Gauss-Newton algorithm.

    pycvcam.optimize.optimize_chains_trf
        Optimize the parameters of a set of transformations organized in chains using the
        least squares method with the Trust Region Reflective algorithm.

    pycvcam.optimize.optimize_camera_gn
        Optimize the parameters of a camera transformation using the Gauss-Newton method.

    """
    # -------------
    # Input Formats Check
    # -------------
    if not isinstance(seq_transforms, Sequence):
        raise TypeError(
            f"seq_transforms must be a sequence of Transform objects, got {type(seq_transforms)}"
        )
    if not all(isinstance(t, Transform) for t in seq_transforms):
        raise TypeError(
            f"All elements of seq_transforms must be instances of Transform, "
            f"got {[type(t) for t in seq_transforms]}"
        )
    n_transforms = len(seq_transforms)

    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")

    if not inplace:
        seq_transforms = [t.copy() for t in seq_transforms]

    if not isinstance(seq_chains, Sequence):
        raise TypeError(
            f"seq_chains must be a sequence of sequences of integers, got {type(seq_chains)}"
        )
    if not all(isinstance(c, Sequence) for c in seq_chains):
        raise TypeError(
            f"All elements of seq_chains must be sequences of integers, "
            f"got {[type(c) for c in seq_chains]}"
        )
    if not all(
        all(isinstance(i, Integral) and 0 <= i < len(seq_transforms) for i in c)
        for c in seq_chains
    ):
        raise ValueError(
            f"All elements of seq_chains must be sequences of valid indices corresponding "
            f"to the transformations in seq_transforms. Got seq_chains {seq_chains} and number of "
            f"transformations {len(seq_transforms)}."
        )
    if not all(len(c) > 0 for c in seq_chains):
        raise ValueError(
            f"All seq_chains must be non-empty. Got seq_chains {seq_chains}."
        )
    if not all(len(set(c)) == len(c) for c in seq_chains):
        raise ValueError(
            f"All seq_chains must not contain duplicate indices. Got seq_chains {seq_chains}."
        )
    n_chains = len(seq_chains)

    if not isinstance(seq_inputs, Sequence):
        raise TypeError(
            f"seq_inputs must be a sequence of arrays, got {type(seq_inputs)}"
        )
    if not len(seq_inputs) == n_chains:
        raise ValueError(
            f"seq_inputs must have the same length as seq_chains, got {len(seq_inputs)} "
            f"and {n_chains} respectively."
        )
    seq_inputs = [numpy.asarray(p, dtype=numpy.float64) for p in seq_inputs]

    if not isinstance(seq_outputs, Sequence):
        raise TypeError(
            f"seq_outputs must be a sequence of arrays, got {type(seq_outputs)}"
        )
    if not len(seq_outputs) == n_chains:
        raise ValueError(
            f"seq_outputs must have the same length as seq_chains, got {len(seq_outputs)} "
            f"and {n_chains} respectively."
        )
    seq_outputs = [numpy.asarray(p, dtype=numpy.float64) for p in seq_outputs]

    for i, (in_p, out_p) in enumerate(zip(seq_inputs, seq_outputs)):
        if in_p.ndim < 2 or out_p.ndim < 2:
            raise ValueError(
                f"Input and output points must have at least 2 dimensions, got "
                f"{in_p.ndim} and {out_p.ndim} dimensions respectively for chain {i}."
            )
        if in_p.shape[-1] != seq_transforms[seq_chains[i][0]].input_dim:
            raise ValueError(
                f"Last dimension of input points must be {seq_transforms[seq_chains[i][0]].input_dim}, "
                f"got {in_p.shape[-1]} for chain {i}."
            )
        if out_p.shape[-1] != seq_transforms[seq_chains[i][-1]].output_dim:
            raise ValueError(
                f"Last dimension of output points must be {seq_transforms[seq_chains[i][-1]].output_dim}, "
                f"got {out_p.shape[-1]} for chain {i}."
            )
        if in_p.shape[:-1] != out_p.shape[:-1]:
            raise ValueError(
                f"Input and output points must have the same structure, got {in_p.shape} "
                f"and {out_p.shape} respectively for chain {i}."
            )
        if in_p.shape[0] == 0:
            raise ValueError(
                f"Input and output points must have at least one point for chain {i}."
            )
        seq_inputs[i] = in_p.reshape(-1, seq_transforms[seq_chains[i][0]].input_dim)
        seq_outputs[i] = out_p.reshape(-1, seq_transforms[seq_chains[i][-1]].output_dim)

    if seq_guesses is None:
        seq_guesses = [None for _ in range(n_transforms)]
    if not isinstance(seq_guesses, Sequence):
        raise TypeError(
            f"seq_guesses must be a sequence of arrays or None, got {type(seq_guesses)}"
        )
    if not len(seq_guesses) == n_transforms:
        raise ValueError(
            f"seq_guesses must have the same length as transforms, got {len(seq_guesses)} and "
            f"{n_transforms} respectively."
        )
    for i, (g, t) in enumerate(zip(seq_guesses, seq_transforms)):
        if g is None and not t.is_set():
            raise ValueError(
                f"Initial guess for the parameters of transformation {i} is required "
                f"when the current parameters of the transformation are not set."
            )
        elif g is None and t.is_set():
            if t.parameters is None:
                g = numpy.zeros((0,), dtype=numpy.float64)
            else:
                g = t.parameters.copy()
        else:
            g = numpy.asarray(g, dtype=numpy.float64)
        if g.ndim != 1 or g.size != t.n_params:
            raise ValueError(
                f"Guess for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {g.ndim} dimensions and {g.size} parameters."
            )
        seq_guesses[i] = g

    if seq_masks is None:
        seq_masks = [None for _ in range(n_transforms)]
    if not isinstance(seq_masks, Sequence):
        raise TypeError(
            f"seq_masks must be a sequence of arrays or None, got {type(seq_masks)}"
        )
    if not len(seq_masks) == n_transforms:
        raise ValueError(
            f"seq_masks must have the same length as transforms, got {len(seq_masks)} and "
            f"{n_transforms} respectively."
        )
    for i, (m, t) in enumerate(zip(seq_masks, seq_transforms)):
        if m is None:
            m = numpy.ones(t.n_params, dtype=bool)
        else:
            m = numpy.asarray(m, dtype=bool)
        if m.ndim != 1 or m.size != t.n_params:
            raise ValueError(
                f"Mask for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {m.ndim} dimensions and {m.size} parameters."
            )
        seq_masks[i] = m

    if seq_transform_kwargs is None:
        seq_transform_kwargs = [None for _ in range(n_transforms)]
    if not isinstance(seq_transform_kwargs, Sequence):
        raise TypeError(
            f"seq_transform_kwargs must be a sequence of dictionaries or None, got "
            f"{type(seq_transform_kwargs)}"
        )
    if not len(seq_transform_kwargs) == n_transforms:
        raise ValueError(
            f"seq_transform_kwargs must have the same length as transforms, got "
            f"{len(seq_transform_kwargs)} and {n_transforms} respectively."
        )
    for i, tk in enumerate(seq_transform_kwargs):
        if tk is None:
            tk = {}
        if not isinstance(tk, dict):
            raise TypeError(
                f"seq_transform_kwargs for transformation {i} must be a dictionary or None, got {type(tk)}"
            )
        seq_transform_kwargs[i] = tk

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if all(t.n_params == 0 for t in seq_transforms):
        out = tuple(seq_guesses[i].copy() for i in range(n_transforms))
        if return_history:
            return out, [out]
        else:
            return out

    if all(not any(m) for m in seq_masks):
        out = tuple(seq_guesses[i].copy() for i in range(n_transforms))
        if return_history:
            return out, [out]
        else:
            return out

    # -------------
    # Optimization
    # -------------
    _pretext = None
    if verbose_level >= 3:
        _pretext = ""
        n_p = [seq_masks[i].sum() for i in range(n_transforms)]
        count_p = 0
        for i, n in enumerate(n_p):
            if n > 0:
                _end = ""
                if i < n_transforms - 1:
                    _end = "\n"
                _pretext += (
                    f"Transformation {i} - {n} parameters to optimize - Parameters {count_p} to {count_p + n - 1}"
                    + _end
                )
                count_p += n

    out = _solve_optimize_chains_gauss_newton(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_history=return_history,
        _pretext=_pretext,
        _sparse=True if n_chains > 1 else False,
    )

    return out


def optimize_chains_trf(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[int]],
    seq_inputs: Sequence[ArrayLike],
    seq_outputs: Sequence[ArrayLike],
    *,
    seq_guesses: Optional[Sequence[ArrayLike]] = None,
    seq_masks: Optional[Sequence[ArrayLike]] = None,
    seq_scales: Optional[Sequence[ArrayLike]] = None,
    seq_bounds: Optional[Sequence[ArrayLike]] = None,
    seq_transform_kwargs: Optional[Sequence[Dict]] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    loss: Optional[str] = None,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_result: bool = False,
    return_history: bool = False,
    inplace: bool = False,
) -> Tuple[numpy.ndarray, ...]:
    r"""
    Optimize several :class:`Transform` objects according multiple chains of
    transformations using the ``scipy.optimize.least_squares`` method.
    The computation is done with Trust Region Reflective algorithm.

    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a tuple of :math:`N_T`
    :class:`Transform` objects, and :math:`(C_0, C_1, ..., C_{N_C-1})` be a
    tuple of :math:`N_C` chains of transformations.

    A chain :math:`C_i` is defined as a tuple of indices corresponding to the
    transformations in the chain. For example:

    .. code-block:: console

        C_0 = (1, 4, 8) -----> C_0(X) = T_1 ∘ T_4 ∘ T_8(X)

    The optimization process is then defined as:

    .. math::

        \min_{\lambda_0, \lambda_1, ..., \lambda_{N_T-1}} \sum_{i=0}^{N_C-1} \|R_i(\lambda_0, \lambda_1, ..., \lambda_{N_T-1})\|^2

    where :math:`R_i` is the residual vector for the chain :math:`C_i` defined as:

    .. math::

        R_i(\lambda_0, \lambda_1, ..., \lambda_{N_T-1}) = \vec{X}_O - C_i(\vec{X}_I, \lambda_0, \lambda_1, ..., \lambda_{N_T-1})

    .. note::

        This method can be used to optimize the parameters of any transformations that
        implement the `_transform` method.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of :math:`N_T` :class:`Transform` objects to be optimized.
        The ``constants`` attribute of each transformation must be set before calling
        this function. If the ``parameters`` attribute of a transformation is set,
        it will be used as the initial guess for the optimization if the `seq_guesses`
        parameter is None. Note that the input :class:`Transform` objects are not
        modified during the optimization process, a copy of each object is created
        and modified internally to perform the optimization if `inplace` is False.

    seq_chains : Sequence[Sequence[int]]
        A sequence of :math:`N_C` chains of transformations. Each chain is defined as a
        sequence of indices corresponding to the transformations in the chain. Each
        chain must be non-empty and contain valid indices (i.e., integers between 0
        and :math:`N_T-1`).

    seq_inputs : Sequence[ArrayLike]
        A sequence of :math:`N_C` arrays of input points with shape (..., input_dim)
        such that their transformation through the corresponding chain is expected to
        match the output points.

    seq_outputs : Sequence[ArrayLike]
        A sequence of :math:`N_C` arrays of output points to be matched with shape
        (..., output_dim).

    seq_guesses : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of initial guesses for the parameters of each
        transformation with shape (n_params,). If None or if ``seq_guesses[i]`` is None,
        the associated parameters of the transformation ``seq_transforms[i]`` are used.
        Default is None.

    seq_masks : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of masks with shape (n_params,) indicating
        which parameters of each transformation should be optimized. Elements with a
        value of True are optimized, while elements with a value of False are kept
        fixed. If None or if ``seq_masks[i]`` is None, all parameters of the transformation
        ``seq_transforms[i]`` are optimized. Default is None.

    seq_scales : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of scales with shape (n_params,) indicating the
        scale of each parameter of each transformation for the optimization. This is
        used to improve the convergence of the optimization by scaling the parameters to
        a similar range. If None or if ``seq_scales[i]`` is None, no scaling is applied to
        the parameters of the transformation ``seq_transforms[i]`` (i.e., all parameters are
        scaled to 1). Default is None.

    seq_bounds : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of bounds with shape (2, n_params) for each
        transformation. The first row contains the lower bounds and the second row
        contains the upper bounds for each parameter. If None or if ``seq_bounds[i]`` is
        None, no bounds are applied to the parameters of the transformation
        ``seq_transforms[i]`` (i.e., bounds are set to ``+/- numpy.inf``). Default is None.

    seq_transform_kwargs : Optional[Sequence[Dict]], optional
        A sequence of :math:`N_T` dictionaries of additional keyword arguments for the
        ``_transform`` method of each transformation. If None or if
        ``seq_transform_kwargs[i]`` is None, no additional keyword arguments are passed to
        the transformation ``seq_transforms[i]``. Default is None.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations. The optimization process is stopped
        when the number of iterations exceeds ``max_iterations``. Default is None, which
        means no limit on the number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function. The optimization process is
        stopped when ``dF < ftol * F``. The default value is None, which means the ftol
        criterion is not used for stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters. The optimization process is
        stopped when ``norm(dx) < xtol * (xtol + norm(x))``. The default value is None,
        which means the xtol criterion is not used for stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient. The optimization process is stopped
        when ``norm(g_scaled, ord=numpy.inf) < gtol`` where g_scaled is the value of the
        gradient scaled to account for the presence of the bounds. The default value is
        None, which means the gtol criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set
        to ``1e-8``. If any of the stopping criteria is already specified, it will be
        overridden by the user-specified value.

    loss : Optional[str], optional
        If specified, the optimization will use a robust loss function to reduce the
        influence of outliers in the optimization. The available loss functions are
        'huber', 'cauchy', 'arctan', 'soft_l1', and 'linear'. Default is None, which
        means the standard least squares loss is used (i.e., 'linear').

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_result : bool, optional
        If True, the function returns a dictionary containing the optimization result
        including the optimized parameters, the cost, the number of iterations, and
        other information about the optimization process. Default is False.

    return_history : bool, optional
        If True, the function returns a history of the parameters during the optimization
        process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformations. If False (default), copies of the transformations are created
        and modified internally to perform the optimization, leaving the input transformations
        unchanged.


    Returns
    -------
    parameters : Tuple[numpy.ndarray, ...]
        A tuple of :math:`N_T` arrays of optimized parameters for each transformation
        with shape (n_params,). Each array contains both the optimized parameters
        (corresponding to True values in the `mask`) and the fixed parameters
        (corresponding to False values in the `mask`), where the fixed parameters are
        equal to their initial values.

    result : scipy.optimize.OptimizeResult, optional
        If return_result is True, the full result object from scipy.optimize.least_squares is also returned,
        which includes information about the optimization process, such as the optimized parameters,
        cost function value, number of iterations, and convergence status.

    history : List[Tuple[numpy.ndarray, ...]], optional
        A history of the optimization process including the parameters of each
        transformation with shape (n_params,). Returned only if `return_history` is True.


    See Also
    --------
    pycvcam.optimize.optimize_parameters_trf
        Optimize the parameters of a transformation using the least squares method
        with the Trust Region Reflective algorithm.

    pycvcam.optimize.optimize_camera_trf
        Optimize the parameters of a camera transformation using the Trust Region Reflective method.

    pycvcam.optimize.optimize_chains_gn
        Optimize the parameters of a set of transformations organized in chains using the
        least squares method with the Gauss-Newton algorithm.

    """
    # -------------
    # Input Formats Check
    # -------------
    if not isinstance(seq_transforms, Sequence):
        raise TypeError(
            f"seq_transforms must be a sequence of Transform objects, got {type(seq_transforms)}"
        )
    if not all(isinstance(t, Transform) for t in seq_transforms):
        raise TypeError(
            f"All elements of seq_transforms must be instances of Transform, "
            f"got {[type(t) for t in seq_transforms]}"
        )
    n_transforms = len(seq_transforms)

    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")

    if not inplace:
        seq_transforms = [t.copy() for t in seq_transforms]

    if not isinstance(seq_chains, Sequence):
        raise TypeError(
            f"seq_chains must be a sequence of sequences of integers, got {type(seq_chains)}"
        )
    if not all(isinstance(c, Sequence) for c in seq_chains):
        raise TypeError(
            f"All elements of seq_chains must be sequences of integers, "
            f"got {[type(c) for c in seq_chains]}"
        )
    if not all(
        all(isinstance(i, Integral) and 0 <= i < len(seq_transforms) for i in c)
        for c in seq_chains
    ):
        raise ValueError(
            f"All elements of seq_chains must be sequences of valid indices corresponding "
            f"to the transformations in seq_transforms. Got seq_chains {seq_chains} and number of "
            f"transformations {len(seq_transforms)}."
        )
    if not all(len(c) > 0 for c in seq_chains):
        raise ValueError(
            f"All seq_chains must be non-empty. Got seq_chains {seq_chains}."
        )
    if not all(len(set(c)) == len(c) for c in seq_chains):
        raise ValueError(
            f"All seq_chains must not contain duplicate indices. Got seq_chains {seq_chains}."
        )
    n_chains = len(seq_chains)

    if not isinstance(seq_inputs, Sequence):
        raise TypeError(
            f"seq_inputs must be a sequence of arrays, got {type(seq_inputs)}"
        )
    if not len(seq_inputs) == n_chains:
        raise ValueError(
            f"seq_inputs must have the same length as seq_chains, got {len(seq_inputs)} "
            f"and {n_chains} respectively."
        )
    seq_inputs = [numpy.asarray(p, dtype=numpy.float64) for p in seq_inputs]

    if not isinstance(seq_outputs, Sequence):
        raise TypeError(
            f"seq_outputs must be a sequence of arrays, got {type(seq_outputs)}"
        )
    if not len(seq_outputs) == n_chains:
        raise ValueError(
            f"seq_outputs must have the same length as seq_chains, got {len(seq_outputs)} "
            f"and {n_chains} respectively."
        )
    seq_outputs = [numpy.asarray(p, dtype=numpy.float64) for p in seq_outputs]

    for i, (in_p, out_p) in enumerate(zip(seq_inputs, seq_outputs)):
        if in_p.ndim < 2 or out_p.ndim < 2:
            raise ValueError(
                f"Input and output points must have at least 2 dimensions, got "
                f"{in_p.ndim} and {out_p.ndim} dimensions respectively for chain {i}."
            )
        if in_p.shape[-1] != seq_transforms[seq_chains[i][0]].input_dim:
            raise ValueError(
                f"Last dimension of input points must be {seq_transforms[seq_chains[i][0]].input_dim}, "
                f"got {in_p.shape[-1]} for chain {i}."
            )
        if out_p.shape[-1] != seq_transforms[seq_chains[i][-1]].output_dim:
            raise ValueError(
                f"Last dimension of output points must be {seq_transforms[seq_chains[i][-1]].output_dim}, "
                f"got {out_p.shape[-1]} for chain {i}."
            )
        if in_p.shape[:-1] != out_p.shape[:-1]:
            raise ValueError(
                f"Input and output points must have the same structure, got {in_p.shape} "
                f"and {out_p.shape} respectively for chain {i}."
            )
        if in_p.shape[0] == 0:
            raise ValueError(
                f"Input and output points must have at least one point for chain {i}."
            )
        seq_inputs[i] = in_p.reshape(-1, seq_transforms[seq_chains[i][0]].input_dim)
        seq_outputs[i] = out_p.reshape(-1, seq_transforms[seq_chains[i][-1]].output_dim)

    if seq_guesses is None:
        seq_guesses = [None for _ in range(n_transforms)]
    if not isinstance(seq_guesses, Sequence):
        raise TypeError(
            f"seq_guesses must be a sequence of arrays or None, got {type(seq_guesses)}"
        )
    if not len(seq_guesses) == n_transforms:
        raise ValueError(
            f"seq_guesses must have the same length as transforms, got {len(seq_guesses)} and "
            f"{n_transforms} respectively."
        )
    for i, (g, t) in enumerate(zip(seq_guesses, seq_transforms)):
        if g is None and not t.is_set():
            raise ValueError(
                f"Initial guess for the parameters of transformation {i} is required "
                f"when the current parameters of the transformation are not set."
            )
        elif g is None and t.is_set():
            if t.parameters is None:
                g = numpy.zeros((0,), dtype=numpy.float64)
            else:
                g = t.parameters.copy()
        else:
            g = numpy.asarray(g, dtype=numpy.float64)
        if g.ndim != 1 or g.size != t.n_params:
            raise ValueError(
                f"Guess for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {g.ndim} dimensions and {g.size} parameters."
            )
        seq_guesses[i] = g

    if seq_masks is None:
        seq_masks = [None for _ in range(n_transforms)]
    if not isinstance(seq_masks, Sequence):
        raise TypeError(
            f"seq_masks must be a sequence of arrays or None, got {type(seq_masks)}"
        )
    if not len(seq_masks) == n_transforms:
        raise ValueError(
            f"seq_masks must have the same length as transforms, got {len(seq_masks)} and "
            f"{n_transforms} respectively."
        )
    for i, (m, t) in enumerate(zip(seq_masks, seq_transforms)):
        if m is None:
            m = numpy.ones(t.n_params, dtype=bool)
        else:
            m = numpy.asarray(m, dtype=bool)
        if m.ndim != 1 or m.size != t.n_params:
            raise ValueError(
                f"Mask for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {m.ndim} dimensions and {m.size} parameters."
            )
        seq_masks[i] = m

    if seq_scales is None:
        seq_scales = [None for _ in range(n_transforms)]
    if not isinstance(seq_scales, Sequence):
        raise TypeError(
            f"seq_scales must be a sequence of arrays or None, got {type(seq_scales)}"
        )
    if not len(seq_scales) == n_transforms:
        raise ValueError(
            f"seq_scales must have the same length as transforms, got {len(seq_scales)} and "
            f"{n_transforms} respectively."
        )
    for i, (s, t) in enumerate(zip(seq_scales, seq_transforms)):
        if s is None:
            s = numpy.ones(t.n_params, dtype=numpy.float64)
        else:
            s = numpy.asarray(s, dtype=numpy.float64)
        if s.ndim != 1 or s.size != t.n_params:
            raise ValueError(
                f"Scale for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {s.ndim} dimensions and {s.size} parameters."
            )
        seq_scales[i] = s

    if seq_bounds is None:
        seq_bounds = [None for _ in range(n_transforms)]
    if not isinstance(seq_bounds, Sequence):
        raise TypeError(
            f"seq_bounds must be a sequence of arrays or None, got {type(seq_bounds)}"
        )
    if not len(seq_bounds) == n_transforms:
        raise ValueError(
            f"seq_bounds must have the same length as transforms, got {len(seq_bounds)} and "
            f"{n_transforms} respectively."
        )
    for i, (b, t) in enumerate(zip(seq_bounds, seq_transforms)):
        if b is None:
            b = numpy.array(
                [[-numpy.inf] * t.n_params, [numpy.inf] * t.n_params],
                dtype=numpy.float64,
            )
        else:
            b = numpy.asarray(b, dtype=numpy.float64)
        if b.ndim != 2 or b.shape != (2, t.n_params):
            raise ValueError(
                f"Bounds for transformation {i} must be a 2D array with shape (2, {t.n_params}), "
                f"got {b.ndim} dimensions and shape {b.shape}."
            )
        if not all(b[0, j] <= seq_guesses[i][j] <= b[1, j] for j in range(t.n_params)):
            raise ValueError(
                f"Initial guess for transformation {i} must be within the bounds for each parameter. "
                f"Got guess {seq_guesses[i]} and bounds {b}."
            )
        seq_bounds[i] = b

    if seq_transform_kwargs is None:
        seq_transform_kwargs = [None for _ in range(n_transforms)]
    if not isinstance(seq_transform_kwargs, Sequence):
        raise TypeError(
            f"seq_transform_kwargs must be a sequence of dictionaries or None, got "
            f"{type(seq_transform_kwargs)}"
        )
    if not len(seq_transform_kwargs) == n_transforms:
        raise ValueError(
            f"seq_transform_kwargs must have the same length as transforms, got "
            f"{len(seq_transform_kwargs)} and {n_transforms} respectively."
        )
    for i, tk in enumerate(seq_transform_kwargs):
        if tk is None:
            tk = {}
        if not isinstance(tk, dict):
            raise TypeError(
                f"seq_transform_kwargs for transformation {i} must be a dictionary or None, got {type(tk)}"
            )
        seq_transform_kwargs[i] = tk

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if loss is not None:
        if not isinstance(loss, str) or loss not in [
            "huber",
            "cauchy",
            "arctan",
            "soft_l1",
            "linear",
        ]:
            raise ValueError(
                f"loss must be one of 'huber', 'cauchy', 'arctan', 'soft_l1', or 'linear', got {loss}"
            )
    else:
        loss = "linear"

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_result, bool):
        raise TypeError(f"return_result must be a boolean, got {type(return_result)}")

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if all(t.n_params == 0 for t in seq_transforms):
        out = tuple(seq_guesses[i].copy() for i in range(n_transforms))
        if return_history:
            return out, [out]
        else:
            return out

    if all(not any(m) for m in seq_masks):
        out = tuple(seq_guesses[i].copy() for i in range(n_transforms))
        if return_history:
            return out, [out]
        else:
            return out

    # -------------
    # Optimization
    # -------------
    _pretext = None
    if verbose_level >= 3:
        _pretext = ""
        n_p = [seq_masks[i].sum() for i in range(n_transforms)]
        count_p = 0
        for i, n in enumerate(n_p):
            if n > 0:
                _end = ""
                if i < n_transforms - 1:
                    _end = "\n"
                _pretext += (
                    f"Transformation {i} - {n} parameters to optimize - Parameters {count_p} to {count_p + n - 1}"
                    + _end
                )
                count_p += n

    out = _solve_optimize_chains_trf_scipy(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_scales=seq_scales,
        seq_bounds=seq_bounds,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        loss=loss,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_result=return_result,
        return_history=return_history,
        _pretext=_pretext,
        _sparse=True if n_chains > 1 else False,
    )

    return out


def optimize_chains_lm(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[int]],
    seq_inputs: Sequence[ArrayLike],
    seq_outputs: Sequence[ArrayLike],
    *,
    seq_guesses: Optional[Sequence[ArrayLike]] = None,
    seq_masks: Optional[Sequence[ArrayLike]] = None,
    seq_transform_kwargs: Optional[Sequence[Dict]] = None,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    loss: Optional[str] = None,
    filter_nans: bool = False,
    verbose_level: Integral = 0,
    return_result: bool = False,
    return_history: bool = False,
    inplace: bool = False,
) -> Tuple[numpy.ndarray, ...]:
    r"""
    Optimize several :class:`Transform` objects according multiple chains of
    transformations using the ``scipy.optimize.least_squares`` method.
    The computation is done with Levenberg-Marquardt algorithm.

    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a tuple of :math:`N_T`
    :class:`Transform` objects, and :math:`(C_0, C_1, ..., C_{N_C-1})` be a
    tuple of :math:`N_C` chains of transformations.

    A chain :math:`C_i` is defined as a tuple of indices corresponding to the
    transformations in the chain. For example:

    .. code-block:: console

        C_0 = (1, 4, 8) -----> C_0(X) = T_1 ∘ T_4 ∘ T_8(X)

    The optimization process is then defined as:

    .. math::

        \min_{\lambda_0, \lambda_1, ..., \lambda_{N_T-1}} \sum_{i=0}^{N_C-1} \|R_i(\lambda_0, \lambda_1, ..., \lambda_{N_T-1})\|^2

    where :math:`R_i` is the residual vector for the chain :math:`C_i` defined as:

    .. math::

        R_i(\lambda_0, \lambda_1, ..., \lambda_{N_T-1}) = \vec{X}_O - C_i(\vec{X}_I, \lambda_0, \lambda_1, ..., \lambda_{N_T-1})

    .. note::

        This method can be used to optimize the parameters of any transformations that
        implement the `_transform` method.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of :math:`N_T` :class:`Transform` objects to be optimized.
        The ``constants`` attribute of each transformation must be set before calling
        this function. If the ``parameters`` attribute of a transformation is set,
        it will be used as the initial guess for the optimization if the `seq_guesses`
        parameter is None. Note that the input :class:`Transform` objects are not
        modified during the optimization process, a copy of each object is created
        and modified internally to perform the optimization if `inplace` is False.

    seq_chains : Sequence[Sequence[int]]
        A sequence of :math:`N_C` chains of transformations. Each chain is defined as a
        sequence of indices corresponding to the transformations in the chain. Each
        chain must be non-empty and contain valid indices (i.e., integers between 0
        and :math:`N_T-1`).

    seq_inputs : Sequence[ArrayLike]
        A sequence of :math:`N_C` arrays of input points with shape (..., input_dim)
        such that their transformation through the corresponding chain is expected to
        match the output points.

    seq_outputs : Sequence[ArrayLike]
        A sequence of :math:`N_C` arrays of output points to be matched with shape
        (..., output_dim).

    seq_guesses : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of initial guesses for the parameters of each
        transformation with shape (n_params,). If None or if ``seq_guesses[i]`` is None,
        the associated parameters of the transformation ``seq_transforms[i]`` are used.
        Default is None.

    seq_masks : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of masks with shape (n_params,) indicating
        which parameters of each transformation should be optimized. Elements with a
        value of True are optimized, while elements with a value of False are kept
        fixed. If None or if ``seq_masks[i]`` is None, all parameters of the transformation
        ``seq_transforms[i]`` are optimized. Default is None.

    seq_transform_kwargs : Optional[Sequence[Dict]], optional
        A sequence of :math:`N_T` dictionaries of additional keyword arguments for the
        ``_transform`` method of each transformation. If None or if
        ``seq_transform_kwargs[i]`` is None, no additional keyword arguments are passed to
        the transformation ``seq_transforms[i]``. Default is None.

    max_iterations : Optional[Integral], optional
        Stop criterion by the number of iterations. The optimization process is stopped
        when the number of iterations exceeds ``max_iterations``. Default is None, which
        means no limit on the number of iterations.

    max_time : Optional[Real], optional
        Stop criterion by the computation time of the optimization. The optimization
        process is stopped when the time since the start of the optimization exceeds
        ``max_time`` seconds. Default is None, which means no limit on the time.

    ftol : Optional[Real], optional
        Stop criterion by the change of the cost function. The optimization process is
        stopped when ``dF < ftol * F``. The default value is None, which means the ftol
        criterion is not used for stopping the optimization.

    xtol : Optional[Real], optional
        Stop criterion by the change of the parameters. The optimization process is
        stopped when ``norm(dx) < xtol * (xtol + norm(x))``. The default value is None,
        which means the xtol criterion is not used for stopping the optimization.

    gtol : Optional[Real], optional
        Stop criterion by the norm of the gradient. The optimization process is stopped
        when ``norm(g_scaled, ord=numpy.inf) < gtol`` where g_scaled is the value of the
        gradient scaled to account for the presence of the bounds. The default value is
        None, which means the gtol criterion is not used for stopping the optimization.

    auto : bool, optional
        If True, the stopping criteria (``ftol``, ``xtol``, and ``gtol``) are all set
        to ``1e-8``. If any of the stopping criteria is already specified, it will be
        overridden by the user-specified value.

    loss : Optional[str], optional
        If specified, the optimization will use a robust loss function to reduce the
        influence of outliers in the optimization. The available loss functions are
        'huber', 'cauchy', 'arctan', 'soft_l1', and 'linear'. Default is None, which
        means the standard least squares loss is used (i.e., 'linear').

    filter_nans : bool, optional
        If True, NaN values in the residuals are filtered out before computing the cost
        and the Jacobian. This can help improve the robustness of the optimization in
        the presence of outliers or invalid data. Default is False.

        .. warning::

            The optimization can try to expulse all points as outliers to reduce
            the cost to zero, which can lead to a failure of the optimization.
            Use with caution.

    verbose_level : int, optional
        Level of algorithm’s verbosity:
        - 0 (default) : work silently.
        - 1 : display a termination report.
        - 2 : display progress during iterations.
        - 3 : display initial jacobian analysis and progress during iterations.

    return_result : bool, optional
        If True, the function returns a dictionary containing the optimization result
        including the optimized parameters, the cost, the number of iterations, and
        other information about the optimization process. Default is False.

    return_history : bool, optional
        If True, the function returns a history of the parameters during the optimization
        process. Default is False.

    inplace : bool, optional
        If True, the optimization is performed in-place, modifying the parameters of the
        input transformations. If False (default), copies of the transformations are created
        and modified internally to perform the optimization, leaving the input transformations
        unchanged.


    Returns
    -------
    parameters : Tuple[numpy.ndarray, ...]
        A tuple of :math:`N_T` arrays of optimized parameters for each transformation
        with shape (n_params,). Each array contains both the optimized parameters
        (corresponding to True values in the `mask`) and the fixed parameters
        (corresponding to False values in the `mask`), where the fixed parameters are
        equal to their initial values.

    result : scipy.optimize.OptimizeResult, optional
        If return_result is True, the full result object from scipy.optimize.least_squares is also returned,
        which includes information about the optimization process, such as the optimized parameters,
        cost function value, number of iterations, and convergence status.

    history : List[Tuple[numpy.ndarray, ...]], optional
        A history of the optimization process including the parameters of each
        transformation with shape (n_params,). Returned only if `return_history` is True.


    See Also
    --------
    pycvcam.optimize.optimize_parameters_lm
        Optimize the parameters of a transformation using the least squares method
        with the Levenberg-Marquardt algorithm.

    pycvcam.optimize.optimize_camera_lm
        Optimize the parameters of a camera transformation using the
        Levenberg-Marquardt method.

    pycvcam.optimize.optimize_chains_gn
        Optimize the parameters of a set of transformations organized in chains using the
        least squares method with the Gauss-Newton algorithm.

    """
    # -------------
    # Input Formats Check
    # -------------
    if not isinstance(seq_transforms, Sequence):
        raise TypeError(
            f"seq_transforms must be a sequence of Transform objects, got {type(seq_transforms)}"
        )
    if not all(isinstance(t, Transform) for t in seq_transforms):
        raise TypeError(
            f"All elements of seq_transforms must be instances of Transform, "
            f"got {[type(t) for t in seq_transforms]}"
        )
    n_transforms = len(seq_transforms)

    if not isinstance(inplace, bool):
        raise TypeError(f"inplace must be a boolean, got {type(inplace)}")

    if not inplace:
        seq_transforms = [t.copy() for t in seq_transforms]

    if not isinstance(seq_chains, Sequence):
        raise TypeError(
            f"seq_chains must be a sequence of sequences of integers, got {type(seq_chains)}"
        )
    if not all(isinstance(c, Sequence) for c in seq_chains):
        raise TypeError(
            f"All elements of seq_chains must be sequences of integers, "
            f"got {[type(c) for c in seq_chains]}"
        )
    if not all(
        all(isinstance(i, Integral) and 0 <= i < len(seq_transforms) for i in c)
        for c in seq_chains
    ):
        raise ValueError(
            f"All elements of seq_chains must be sequences of valid indices corresponding "
            f"to the transformations in seq_transforms. Got seq_chains {seq_chains} and number of "
            f"transformations {len(seq_transforms)}."
        )
    if not all(len(c) > 0 for c in seq_chains):
        raise ValueError(
            f"All seq_chains must be non-empty. Got seq_chains {seq_chains}."
        )
    if not all(len(set(c)) == len(c) for c in seq_chains):
        raise ValueError(
            f"All seq_chains must not contain duplicate indices. Got seq_chains {seq_chains}."
        )
    n_chains = len(seq_chains)

    if not isinstance(seq_inputs, Sequence):
        raise TypeError(
            f"seq_inputs must be a sequence of arrays, got {type(seq_inputs)}"
        )
    if not len(seq_inputs) == n_chains:
        raise ValueError(
            f"seq_inputs must have the same length as seq_chains, got {len(seq_inputs)} "
            f"and {n_chains} respectively."
        )
    seq_inputs = [numpy.asarray(p, dtype=numpy.float64) for p in seq_inputs]

    if not isinstance(seq_outputs, Sequence):
        raise TypeError(
            f"seq_outputs must be a sequence of arrays, got {type(seq_outputs)}"
        )
    if not len(seq_outputs) == n_chains:
        raise ValueError(
            f"seq_outputs must have the same length as seq_chains, got {len(seq_outputs)} "
            f"and {n_chains} respectively."
        )
    seq_outputs = [numpy.asarray(p, dtype=numpy.float64) for p in seq_outputs]

    for i, (in_p, out_p) in enumerate(zip(seq_inputs, seq_outputs)):
        if in_p.ndim < 2 or out_p.ndim < 2:
            raise ValueError(
                f"Input and output points must have at least 2 dimensions, got "
                f"{in_p.ndim} and {out_p.ndim} dimensions respectively for chain {i}."
            )
        if in_p.shape[-1] != seq_transforms[seq_chains[i][0]].input_dim:
            raise ValueError(
                f"Last dimension of input points must be {seq_transforms[seq_chains[i][0]].input_dim}, "
                f"got {in_p.shape[-1]} for chain {i}."
            )
        if out_p.shape[-1] != seq_transforms[seq_chains[i][-1]].output_dim:
            raise ValueError(
                f"Last dimension of output points must be {seq_transforms[seq_chains[i][-1]].output_dim}, "
                f"got {out_p.shape[-1]} for chain {i}."
            )
        if in_p.shape[:-1] != out_p.shape[:-1]:
            raise ValueError(
                f"Input and output points must have the same structure, got {in_p.shape} "
                f"and {out_p.shape} respectively for chain {i}."
            )
        if in_p.shape[0] == 0:
            raise ValueError(
                f"Input and output points must have at least one point for chain {i}."
            )
        seq_inputs[i] = in_p.reshape(-1, seq_transforms[seq_chains[i][0]].input_dim)
        seq_outputs[i] = out_p.reshape(-1, seq_transforms[seq_chains[i][-1]].output_dim)

    if seq_guesses is None:
        seq_guesses = [None for _ in range(n_transforms)]
    if not isinstance(seq_guesses, Sequence):
        raise TypeError(
            f"seq_guesses must be a sequence of arrays or None, got {type(seq_guesses)}"
        )
    if not len(seq_guesses) == n_transforms:
        raise ValueError(
            f"seq_guesses must have the same length as transforms, got {len(seq_guesses)} and "
            f"{n_transforms} respectively."
        )
    for i, (g, t) in enumerate(zip(seq_guesses, seq_transforms)):
        if g is None and not t.is_set():
            raise ValueError(
                f"Initial guess for the parameters of transformation {i} is required "
                f"when the current parameters of the transformation are not set."
            )
        elif g is None and t.is_set():
            if t.parameters is None:
                g = numpy.zeros((0,), dtype=numpy.float64)
            else:
                g = t.parameters.copy()
        else:
            g = numpy.asarray(g, dtype=numpy.float64)
        if g.ndim != 1 or g.size != t.n_params:
            raise ValueError(
                f"Guess for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {g.ndim} dimensions and {g.size} parameters."
            )
        seq_guesses[i] = g

    if seq_masks is None:
        seq_masks = [None for _ in range(n_transforms)]
    if not isinstance(seq_masks, Sequence):
        raise TypeError(
            f"seq_masks must be a sequence of arrays or None, got {type(seq_masks)}"
        )
    if not len(seq_masks) == n_transforms:
        raise ValueError(
            f"seq_masks must have the same length as transforms, got {len(seq_masks)} and "
            f"{n_transforms} respectively."
        )
    for i, (m, t) in enumerate(zip(seq_masks, seq_transforms)):
        if m is None:
            m = numpy.ones(t.n_params, dtype=bool)
        else:
            m = numpy.asarray(m, dtype=bool)
        if m.ndim != 1 or m.size != t.n_params:
            raise ValueError(
                f"Mask for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {m.ndim} dimensions and {m.size} parameters."
            )
        seq_masks[i] = m

    if seq_transform_kwargs is None:
        seq_transform_kwargs = [None for _ in range(n_transforms)]
    if not isinstance(seq_transform_kwargs, Sequence):
        raise TypeError(
            f"seq_transform_kwargs must be a sequence of dictionaries or None, got "
            f"{type(seq_transform_kwargs)}"
        )
    if not len(seq_transform_kwargs) == n_transforms:
        raise ValueError(
            f"seq_transform_kwargs must have the same length as transforms, got "
            f"{len(seq_transform_kwargs)} and {n_transforms} respectively."
        )
    for i, tk in enumerate(seq_transform_kwargs):
        if tk is None:
            tk = {}
        if not isinstance(tk, dict):
            raise TypeError(
                f"seq_transform_kwargs for transformation {i} must be a dictionary or None, got {type(tk)}"
            )
        seq_transform_kwargs[i] = tk

    if max_iterations is not None:
        if not isinstance(max_iterations, Integral) or max_iterations <= 0:
            raise TypeError(
                f"max_iterations must be a positive integer, got {max_iterations}"
            )
        max_iterations = int(max_iterations)

    if max_time is not None:
        if not isinstance(max_time, Real) or max_time <= 0:
            raise TypeError(f"max_time must be a positive float, got {max_time}")
        max_time = float(max_time)

    if not isinstance(auto, bool):
        raise TypeError(f"auto must be a boolean, got {type(auto)}")
    if auto:
        ftol, xtol, gtol = 1e-8, 1e-8, 1e-8

    if ftol is not None:
        if not isinstance(ftol, Real) or ftol <= 0:
            raise TypeError(f"ftol must be a positive float, got {ftol}")
        ftol = float(ftol)

    if xtol is not None:
        if not isinstance(xtol, Real) or xtol <= 0:
            raise TypeError(f"xtol must be a positive float, got {xtol}")
        xtol = float(xtol)

    if gtol is not None:
        if not isinstance(gtol, Real) or gtol <= 0:
            raise TypeError(f"gtol must be a positive float, got {gtol}")
        gtol = float(gtol)

    if loss is not None:
        if not isinstance(loss, str) or loss not in [
            "huber",
            "cauchy",
            "arctan",
            "soft_l1",
            "linear",
        ]:
            raise ValueError(
                f"loss must be one of 'huber', 'cauchy', 'arctan', 'soft_l1', or 'linear', got {loss}"
            )
    else:
        loss = "linear"

    if not isinstance(filter_nans, bool):
        raise TypeError(f"filter_nans must be a boolean, got {type(filter_nans)}")

    if not isinstance(verbose_level, Integral) or not (0 <= verbose_level <= 3):
        raise TypeError(
            f"verbose_level must be an integer between 0 and 3, got {verbose_level}"
        )
    verbose_level = int(verbose_level)

    if not isinstance(return_result, bool):
        raise TypeError(f"return_result must be a boolean, got {type(return_result)}")

    if not isinstance(return_history, bool):
        raise TypeError(f"return_history must be a boolean, got {type(return_history)}")

    if ftol is None and xtol is None and gtol is None:
        raise ValueError(
            "At least one of ftol, xtol, or gtol must be specified for stopping criteria."
        )

    # -------------
    # Edge Cases
    # -------------
    if all(t.n_params == 0 for t in seq_transforms):
        out = tuple(seq_guesses[i].copy() for i in range(n_transforms))
        if return_history:
            return out, [out]
        else:
            return out

    if all(not any(m) for m in seq_masks):
        out = tuple(seq_guesses[i].copy() for i in range(n_transforms))
        if return_history:
            return out, [out]
        else:
            return out

    # -------------
    # Optimization
    # -------------
    _pretext = None
    if verbose_level >= 3:
        _pretext = ""
        n_p = [seq_masks[i].sum() for i in range(n_transforms)]
        count_p = 0
        for i, n in enumerate(n_p):
            if n > 0:
                _end = ""
                if i < n_transforms - 1:
                    _end = "\n"
                _pretext += (
                    f"Transformation {i} - {n} parameters to optimize - Parameters {count_p} to {count_p + n - 1}"
                    + _end
                )
                count_p += n

    out = _solve_optimize_chains_trf_scipy(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_inputs=seq_inputs,
        seq_outputs=seq_outputs,
        seq_masks=seq_masks,
        seq_guesses=seq_guesses,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        loss=loss,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_result=return_result,
        return_history=return_history,
        _pretext=_pretext,
        _sparse=False,
    )

    return out
