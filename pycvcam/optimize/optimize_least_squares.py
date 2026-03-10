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

    if _start:
        density = numpy.count_nonzero(jacobian) / (m * n)
        print(f"Density: {density*100:.2f}%")

    # SVD
    U, S, Vt = numpy.linalg.svd(jacobian, full_matrices=False)
    sigma_max = S[0]
    sigma_min = S[-1]
    cond_number = sigma_max / sigma_min
    print(f"Singular values (max/min): {sigma_max:.3e} / {sigma_min:.3e}")
    print(f"Condition number: {cond_number:.3e}")

    # Variance contribution of each singular value (1/sigma^2)
    print("\nSingular values and their contribution to the variance:")
    print(
        f"| {"Index":^10} | {"Singular Value \u03BB":^18} | {"Var = 1/\u03BB^2":^20} |"
    )
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
    cov = sigma2 * numpy.linalg.inv(jacobian.T @ jacobian)
    print("\nEstimated variances of the parameters:")
    print(
        f"| {'Parameter':^10} | {'Value P':^15} | {'Var = \u03C3^2 (J.T J)^-1':^20} | {'Ratio \u221AV/|P|':^15} |"
    )
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

        print(
            f"| {i:^10} | {parameters[i]:^15.3e} | {var:^20.3e} | {rel_sqrt_str:^15} |"
        )
    if _start:
        print("\n" + "-" * 50)
        print("Optimization in progress...")
        print("-" * 50 + "\n")
    if not _start:
        print("\n" + "=" * 50 + "\n")


def _build_optimize_parameters_lsq_functions(
    object_class: Transform,
    input_points: numpy.ndarray,
    output_points: numpy.ndarray,
    mask: numpy.ndarray,
    guess: numpy.ndarray,
    transform_kwargs: Dict,
    return_history: bool,
    max_iterations: Optional[int],
    max_time: Optional[int],
    filter_nans: bool,
) -> Tuple[Callable, Callable]:

    last_params = None
    last_res = None
    last_jac = None
    history = []
    count_call = 0
    start_time = time.time()

    def compute_func(
        params: numpy.ndarray,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        nonlocal last_params, last_res, last_jac
        if last_params is None or not numpy.array_equal(params, last_params):
            n_points = input_points.shape[0]
            input_dim = input_points.shape[1]
            output_dim = output_points.shape[1]
            parameters = guess.copy()
            parameters[mask] = params
            object_class.parameters = parameters
            transformed_points, _, jacobian_dp = object_class._transform(
                input_points, dx=False, dp=True, **transform_kwargs
            )  # shape (n_points, output_dim) and (n_points, output_dim, n_parameters)
            R = output_points - transformed_points  # shape (n_points, output_dim)
            R = R.ravel()  # shape (n_points * output_dim,)
            jacobian_dp = -jacobian_dp[:, :, mask]  # shape (..., n_params)
            jacobian_dp = jacobian_dp.reshape(
                n_points * output_dim, -1
            )  # shape (n_points*output_dim, n_params)

            if filter_nans:
                filter_mask_R = ~numpy.isfinite(R)
                filter_mask_J = ~numpy.isfinite(jacobian_dp)
                filter_mask = filter_mask_R | numpy.any(filter_mask_J, axis=1)
                R[filter_mask] = 0.0
                jacobian_dp[filter_mask, :] = 0.0
                if numpy.all(filter_mask):
                    raise ValueError(
                        "All residuals are NaN or infinite, filtering set 0 matrix, optimization cannot proceed."
                    )

            last_params = params.copy()
            last_res = R
            last_jac = jacobian_dp

        return last_res, last_jac

    def residuals_func(
        params: numpy.ndarray,
    ) -> numpy.ndarray:
        res, _ = compute_func(params)
        return res

    def jacobian_func(
        params: numpy.ndarray,
    ) -> numpy.ndarray:
        _, jac = compute_func(params)
        return jac

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        nonlocal history, start_time, count_call

        if return_history:
            parameters = guess.copy()
            parameters[mask] = intermediate_result.x
            history.append((parameters.copy(), intermediate_result))

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

    return residuals_func, jacobian_func, callback, get_history


def optimize_parameters_least_squares(
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
    _pretext: Optional[str] = None,
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
        optimization.

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
        - 2 : display progress during iterations (scipy).
        - 3 : display initial jacobian analysis and scipy progress during iterations.

    return_result : bool, optional
        If True, the function returns the ``scipy.optimize.OptimizeResult`` object
        containing information about the convergence of the optimization process.
        Default is False, which means only the optimized parameters are returned.
        If ``n_params`` is 0, or all parameters are masked, the result output will be
        None.

    return_history : bool, optional
        If True, the function returns a history of the optimization process,
        including the parameters and the corresponding ``scipy.optimize.OptimizeResult``
        object at each iteration. Default is False, which means the history is not
        returned. If ``n_params`` is 0, or all parameters are masked, the history output
        will be None.


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

    history : List[Tuple[numpy.ndarray, scipy.optimize.OptimizeResult]], optional
        A history of the optimization process including the parameters with shape
        (n_params,) and the corresponding scipy OptimizeResult object at each iteration.
        Returned only if `return_history` is True.

        .. warning::

            The OptimizeResult objects in the history only contain the parameters that
            were optimized (i.e., the parameters corresponding to True values in the
            `mask`), and not the full parameter vector of the transformation.


    See Also
    --------
    scipy.optimize.least_squares
        For more information about the optimization method.

    pycvcam.optimize.optimize_chain_parameters_least_squares
        For optimizing several transformations in a chain of transformations.

    """
    # -------------
    # Input Formats Check
    # -------------
    if not isinstance(transform, Transform):
        raise TypeError(
            f"transform must be an instance of Transform, got {type(transform)}"
        )

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
        out = numpy.zeros((0,), dtype=numpy.float64)
        if return_result and return_history:
            return out, None, None
        elif return_result:
            return out, None
        elif return_history:
            return out, None
        else:
            return out

    if not any(mask):
        out = guess.copy()
        if return_result and return_history:
            return out, None, None
        elif return_result:
            return out, None
        elif return_history:
            return out, None
        else:
            return out

    # -------------
    # Optimization
    # -------------
    object_class = copy.deepcopy(transform)

    residuals_func, jacobian_func, callback, get_history = (
        _build_optimize_parameters_lsq_functions(
            object_class,
            input_points,
            output_points,
            mask,
            guess,
            transform_kwargs,
            return_history,
            max_iterations,
            max_time,
            filter_nans,
        )
    )

    # Crop the problem to the optimized parameters
    params_initial = guess[mask]  # shape (n_params,)
    params_bounds = bounds[:, mask]  # shape (2, n_params)
    params_scale = scale[mask]  # shape (n_params,)

    if verbose_level >= 3:
        _study_jacobian_least_squares(
            residuals_func(params_initial),
            jacobian_func(params_initial),
            params_initial,
            _pretext,
            _start=True,
        )

    # Run the least squares optimization
    result = scipy.optimize.least_squares(
        fun=residuals_func,
        x0=params_initial,
        jac=jacobian_func,
        bounds=params_bounds,
        x_scale=params_scale,
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
            residuals_func(result.x),
            jacobian_func(result.x),
            result.x,
            _pretext,
            _start=False,
        )

    parameters = guess
    parameters[mask] = result.x
    parameters = parameters.copy()  # shape (n_params,)

    if return_result and return_history:
        return parameters, result, get_history()
    elif return_result:
        return parameters, result
    elif return_history:
        return parameters, get_history()
    else:
        return parameters


def optimize_camera_least_squares(
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
) -> Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray], Optional[numpy.ndarray]]:
    """
    Optimize the parameters of the intrinsic, distortion, and extrinsic transformations
    of a camera model such that the projection of the world points matches the image
    points using the ``scipy.optimize.least_squares`` method.
    The computation is done with Trust Region Reflective algorithm.

    The optimization is performed by calling
    the ``optimize_parameters_least_squares`` function for a composite transformation
    that combines the intrinsic, distortion, and extrinsic transformations of the camera
    model.

    For more information about the optimization method, please refer to the
    ``scipy.optimize.least_squares`` documentation and the
    ``optimize_parameters_least_squares`` function documentation.

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
        If True, the function returns a history of the optimization process,
        including the parameters and the corresponding ``scipy.optimize.OptimizeResult``
        object at each iteration. Default is False, which means the history is not
        returned. If all transformations are None, or all parameters of the
        transformations are masked, the history output will be None.

    Returns
    -------
    intrinsic_parameters : Optional[numpy.ndarray]
        The optimized parameters of the intrinsic transformation with shape
        (n_intrinsic_params,). This array contains both the optimized parameters
        (corresponding to True values in the `mask_intrinsic`) and the fixed parameters
        (corresponding to False values in the `mask_intrinsic`), where the fixed
        parameters are equal to their initial values. Returned only if `intrinsic` is
        not None, otherwise None.

    distortion_parameters : Optional[numpy.ndarray]
        The optimized parameters of the distortion transformation with shape
        (n_distortion_params,). This array contains both the optimized parameters
        (corresponding to True values in the `mask_distortion`) and the fixed parameters
        (corresponding to False values in the `mask_distortion`), where the fixed
        parameters are equal to their initial values. Returned only if `distortion` is
        not None, otherwise None.

    extrinsic_parameters : Optional[numpy.ndarray]
        The optimized parameters of the extrinsic transformation with shape
        (n_extrinsic_params,). This array contains both the optimized parameters
        (corresponding to True values in the `mask_extrinsic`) and the fixed parameters
        (corresponding to False values in the `mask_extrinsic`), where the fixed
        parameters are equal to their initial values. Returned only if `extrinsic` is
        not None, otherwise None.

    result : scipy.optimize.OptimizeResult, optional
        The result of the optimization process containing information about the
        convergence of the optimization. Returned only if `return_result` is True and
        at least one of the transformations is not None and has at least one parameter
        to optimize, otherwise None.

        .. warning::

            Only contains the parameters that were optimized (i.e., the parameters
            corresponding to True values in the `mask_intrinsic`, `mask_distortion`, and
            `mask_extrinsic`), and not the full parameter vectors of the transformations.

    history : List[Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray], Optional[numpy.ndarray], scipy.optimize.OptimizeResult]], optional
        A history of the optimization process including the parameters with shape
        (n_params,) and the corresponding scipy OptimizeResult object at each iteration.
        Returned only if `return_history` is True and at least one of the transformations
        is not None and has at least one parameter to optimize, otherwise None.

        .. warning::

            The OptimizeResult objects in the history only contain the parameters that
            were optimized (i.e., the parameters corresponding to True values in the
            `mask_intrinsic`, `mask_distortion`, and `mask_extrinsic`), and not the full
            parameter vectors of the transformations.

    See Also
    --------
    optimize_parameters_least_squares
        For optimizing the parameters of a single transformation.

    scipy.optimize.least_squares
        For more information about the optimization method.

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

    transform = TransformComposition([extrinsic, distortion, intrinsic])

    if guess_intrinsic is None and not intrinsic.is_set():
        raise ValueError(
            "Initial guess for the parameters of the intrinsic transformation is required "
            "when the current parameters of the intrinsic transformation are not set."
        )
    elif guess_intrinsic is None and intrinsic.is_set():
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
        guess_extrinsic = extrinsic.parameters.copy()
    else:
        guess_extrinsic = numpy.asarray(guess_extrinsic, dtype=numpy.float64)
    if guess_extrinsic.ndim != 1 or guess_extrinsic.size != extrinsic.n_params:
        raise ValueError(
            f"guess_extrinsic must be a 1D array with {extrinsic.n_params} parameters, "
            f"got {guess_extrinsic.ndim} dimensions and {guess_extrinsic.size} parameters."
        )

    guess = numpy.concatenate([guess_extrinsic, guess_distortion, guess_intrinsic])

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

    mask = numpy.concatenate([mask_extrinsic, mask_distortion, mask_intrinsic])

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

    scale = numpy.concatenate([scale_extrinsic, scale_distortion, scale_intrinsic])

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

    bounds = numpy.concatenate(
        [bounds_extrinsic, bounds_distortion, bounds_intrinsic], axis=1
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

    transform_kwargs = {
        "list_kwargs": [extrinsic_kwargs, distortion_kwargs, intrinsic_kwargs]
    }

    # -------------
    # Optimize the parameters of the composite transformation
    # -------------
    _pretext = None
    if verbose_level >= 3:
        _pretext = ""
        n_pi = mask_intrinsic.sum()
        n_pd = mask_distortion.sum()
        n_pe = mask_extrinsic.sum()
        if n_pe > 0:
            _pretext += f"{n_pe} Extrinsic parameters to optimize - Parameters 0 to {n_pe - 1}\n"
        if n_pd > 0:
            _pretext += f"{n_pd} Distortion parameters to optimize - Parameters {n_pe} to {n_pe+ n_pd - 1}\n"
        if n_pi > 0:
            _pretext += f"{n_pi} Intrinsic parameters to optimize - Parameters {n_pe + n_pd} to {n_pe + n_pd + n_pi - 1}"

    parameters, result, history = optimize_parameters_least_squares(
        transform,
        input_points=world_points,
        output_points=image_points,
        mask=mask,
        guess=guess,
        scale=scale,
        bounds=bounds,
        transform_kwargs=transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        auto=auto,
        loss=loss,
        filter_nans=filter_nans,
        verbose_level=verbose_level,
        return_result=True,
        return_history=True,
        _pretext=_pretext,
    )

    extrinsic_parameters = (
        parameters[: extrinsic.n_params] if not skip_extrinsic else None
    )
    distortion_parameters = (
        parameters[extrinsic.n_params : extrinsic.n_params + distortion.n_params]
        if not skip_distortion
        else None
    )
    intrinsic_parameters = (
        parameters[extrinsic.n_params + distortion.n_params :]
        if not skip_intrinsic
        else None
    )

    if return_history and history is not None:
        # Convert the history to contain the full parameter vectors of the transformations
        full_history = []
        for params, res in history:
            e_params = params[: extrinsic.n_params] if not skip_extrinsic else None
            d_params = (
                params[extrinsic.n_params : extrinsic.n_params + distortion.n_params]
                if not skip_distortion
                else None
            )
            i_params = (
                params[extrinsic.n_params + distortion.n_params :]
                if not skip_intrinsic
                else None
            )
            full_history.append((i_params, d_params, e_params, res))
        history = full_history

    if return_result and return_history:
        return (
            intrinsic_parameters,
            distortion_parameters,
            extrinsic_parameters,
            result,
            history,
        )
    elif return_result:
        return intrinsic_parameters, distortion_parameters, extrinsic_parameters, result
    elif return_history:
        return (
            intrinsic_parameters,
            distortion_parameters,
            extrinsic_parameters,
            history,
        )
    else:
        return intrinsic_parameters, distortion_parameters, extrinsic_parameters


def _build_optimize_chain_parameters_lsq_functions(
    object_classes: Sequence[Transform],
    chains: Sequence[Sequence[int]],
    input_points: Sequence[numpy.ndarray],
    output_points: Sequence[numpy.ndarray],
    mask: Sequence[numpy.ndarray],
    guess: Sequence[numpy.ndarray],
    transform_kwargs: Sequence[Dict],
    return_history: bool,
    max_iterations: Optional[int],
    max_time: Optional[int],
    filter_nans: bool,
) -> Tuple[Callable, Callable]:

    last_params = None
    last_res = None
    last_jac = None

    n_transform = len(object_classes)
    n_chains = len(chains)

    t_n_parameters = [p.size for p in guess]  # len n_transform
    t_n_reduced_parameters = [numpy.sum(m) for m in mask]  # len n_transform
    t_jac_start_p = numpy.cumsum([0] + t_n_parameters[:-1])  # len n_transform
    t_jac_end_p = numpy.cumsum(t_n_parameters)  # len n_transform
    t_jac_start_rp = numpy.cumsum([0] + t_n_reduced_parameters[:-1])  # len n_transform
    t_jac_end_rp = numpy.cumsum(t_n_reduced_parameters)  # len n_transform

    c_n_points = [p.shape[0] for p in input_points]  # len n_chains
    c_input_dim = [p.shape[1] for p in input_points]  # len n_chains
    c_output_dim = [p.shape[1] for p in output_points]  # len n_chains
    c_n_equations = [n * o for n, o in zip(c_n_points, c_output_dim)]  # len n_chains
    c_mask = [numpy.concatenate([mask[i] for i in c]) for c in chains]  # len n_chains
    c_n_parameters = [sum(t_n_parameters[i] for i in c) for c in chains]  # len n_chains
    c_n_reduced_parameters = [
        sum(t_n_reduced_parameters[i] for i in c) for c in chains
    ]  # len n_chains
    c_jac_start_p = numpy.cumsum([0] + c_n_parameters[:-1])  # len n_chains
    c_jac_end_p = numpy.cumsum(c_n_parameters)  # len n_chains
    c_jac_start_rp = numpy.cumsum([0] + c_n_reduced_parameters[:-1])  # len n_chains
    c_jac_end_rp = numpy.cumsum(c_n_reduced_parameters)  # len n_chains
    c_jac_start_e = numpy.cumsum([0] + c_n_equations[:-1])  # len n_chains
    c_jac_end_e = numpy.cumsum(c_n_equations)  # len n_chains

    history = []
    count_call = 0
    start_time = time.time()

    def compute_func(
        params: numpy.ndarray,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        nonlocal last_params, last_res, last_jac
        if last_params is None or not numpy.array_equal(params, last_params):
            parameters = [g.copy() for g in guess]  # n_transform lgth
            for i, (m, p) in enumerate(zip(mask, params)):
                parameters[i][m] = p
            for i, t in enumerate(object_classes):
                t.parameters = parameters[i]
            # Create the composite transformations for each chain
            R_list = []
            J_list = []
            for i, c in enumerate(chains):
                tc = TransformComposition([object_classes[j] for j in c])
                list_kwargs = [transform_kwargs[j] for j in c]
                transformed_points, _, jacobian_dp = tc._transform(
                    input_points[i], dx=False, dp=True, list_kwargs=list_kwargs
                )  # shape (n_points, output_dim) and (n_points, output_dim, n_parameters)
                R = (
                    output_points[i] - transformed_points
                )  # shape (n_points, output_dim)
                R_list.append(R.flatten())  # shape (n_points * output_dim,)
                jacobian_dp = -jacobian_dp[
                    :, :, c_mask[i]
                ]  # shape (..., n_reduced_parameters)
                J_list.append(
                    jacobian_dp.reshape(c_n_points[i] * c_output_dim[i], -1)
                )  # shape (n_points * output_dim, n_params)
            # Concatenate the residuals and Jacobians for all chains
            R = numpy.concatenate(R_list)  # shape (sum(n_points * output_dim),)
            # Assemble the Jacobian according to the chain structure and the mask
            J_full = scipy.sparse.lil_matrix(
                (sum(c_n_equations), sum(t_n_reduced_parameters))
            )
            for i, c in enumerate(chains):
                local_offsets = numpy.cumsum(
                    [0] + [t_n_reduced_parameters[j] for j in c[:-1]]
                )
                for index, j in enumerate(c):
                    local_start = local_offsets[index]
                    local_end = local_start + t_n_reduced_parameters[j]
                    J_full[
                        c_jac_start_e[i] : c_jac_end_e[i],
                        t_jac_start_rp[j] : t_jac_end_rp[j],
                    ] = J_list[i][:, local_start:local_end]

            if filter_nans:
                filter_mask_R = ~numpy.isfinite(R)
                filter_mask_J = ~numpy.isfinite(J_full.toarray())
                filter_mask = filter_mask_R | numpy.any(filter_mask_J, axis=1)
                R[filter_mask] = 0.0
                J_full[filter_mask, :] = 0.0
                if numpy.all(filter_mask):
                    raise ValueError(
                        "All residuals are NaN or infinite, filtering set empty matrix, optimization cannot proceed."
                    )
                print(
                    f"Warning: Filtered {numpy.sum(filter_mask)} out of {R.size} equations due to NaN or infinite values in the residuals or Jacobian."
                )

            J = J_full.tocsr()  # Convert to CSR for efficient arithmetic operations

            last_res = R
            last_jac = J
            last_params = params.copy()
            return last_res, last_jac

        return last_res, last_jac

    def residuals_func(
        params: numpy.ndarray,
    ) -> numpy.ndarray:
        res, _ = compute_func(params)
        return res

    def jacobian_func(
        params: numpy.ndarray,
    ) -> numpy.ndarray:
        _, jac = compute_func(params)
        return jac

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        nonlocal history, start_time, count_call

        if return_history:
            parameters = guess.copy()
            parameters[mask] = intermediate_result.x
            history.append((parameters.copy(), intermediate_result))

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

    return residuals_func, jacobian_func, callback, get_history


def optimize_chain_parameters_least_squares(
    transforms: Sequence[Transform],
    chains: Sequence[Sequence[int]],
    input_points: Sequence[ArrayLike],
    output_points: Sequence[ArrayLike],
    *,
    guess: Optional[Sequence[ArrayLike]] = None,
    mask: Optional[Sequence[ArrayLike]] = None,
    scale: Optional[Sequence[ArrayLike]] = None,
    bounds: Optional[Sequence[ArrayLike]] = None,
    transform_kwargs: Optional[Sequence[Dict]] = None,
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
) -> Tuple[numpy.ndarray, ...]:
    r"""
    Optimize several :class:`Transform` objects according multiple chains of
    transformations using the ``scipy.optimize.least_squares`` method. The computation
    is done with Trust Region Reflective algorithm.

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

    For more information about the optimization method, please refer to the
    ``scipy.optimize.least_squares`` documentation.

    .. important::

        At least one of the stopping criteria (``ftol``, ``xtol``, or ``gtol``)
        must be specified for the optimization to stop. You can also
        set ``auto`` to True to use ``1e-8`` for all stopping criteria.

    Parameters
    ----------
    transforms : Sequence[Transform]
        A sequence of :math:`N_T` :class:`Transform` objects to be optimized.
        The ``constants`` attribute of each transformation must be set before calling
        this function. If the ``parameters`` attribute of a transformation is set,
        it will be used as the initial guess for the optimization if the `guess`
        parameter is None. Note that the input :class:`Transform` objects are not
        modified during the optimization process, a copy of each object is created
        and modified internally to perform the optimization.

    chains : Sequence[Sequence[int]]
        A sequence of :math:`N_C` chains of transformations. Each chain is defined as a
        sequence of indices corresponding to the transformations in the chain. Each
        chain must be non-empty and contain valid indices (i.e., integers between 0
        and :math:`N_T-1`).

    input_points : Sequence[ArrayLike]
        A sequence of :math:`N_C` arrays of input points with shape (..., input_dim)
        such that their transformation through the corresponding chain is expected to
        match the output points.

    output_points : Sequence[ArrayLike]
        A sequence of :math:`N_C` arrays of output points to be matched with shape
        (..., output_dim).

    guess : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of initial guesses for the parameters of each
        transformation with shape (n_params,). If None or if ``guess[i]`` is None,
        the associated parameters of the transformation ``transforms[i]`` are used.
        Default is None.

    mask : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of masks with shape (n_params,) indicating
        which parameters of each transformation should be optimized. Elements with a
        value of True are optimized, while elements with a value of False are kept
        fixed. If None or if ``mask[i]`` is None, all parameters of the transformation
        ``transforms[i]`` are optimized. Default is None.

    scale : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of scales with shape (n_params,) indicating the
        scale of each parameter of each transformation for the optimization. This is
        used to improve the convergence of the optimization by scaling the parameters to
        a similar range. If None or if ``scale[i]`` is None, no scaling is applied to
        the parameters of the transformation ``transforms[i]`` (i.e., all parameters are
        scaled to 1). Default is None.

    bounds : Optional[Sequence[ArrayLike]], optional
        A sequence of :math:`N_T` arrays of bounds with shape (2, n_params) for each
        transformation. The first row contains the lower bounds and the second row
        contains the upper bounds for each parameter. If None or if ``bounds[i]`` is
        None, no bounds are applied to the parameters of the transformation
        ``transforms[i]`` (i.e., bounds are set to ``+/- numpy.inf``). Default is None.

    transform_kwargs : Optional[Sequence[Dict]], optional
        A sequence of :math:`N_T` dictionaries of additional keyword arguments for the
        ``transform._transform`` method of each transformation. If None or if
        ``transform_kwargs[i]`` is None, no additional keyword arguments are passed to
        the transformation ``transforms[i]``. Default is None.

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
        The loss function to use for robust optimization. If None, the standard least
        squares loss is used. Supported values are 'linear', 'soft_l1', 'huber',
        'cauchy', and 'arctan'. Default is None, which means no robust loss is used
        (i.e., 'linear' loss is used).

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

    return_result : bool, optional
        If True, the function returns a ``scipy.optimize.OptimizeResult`` object
        containing information about the convergence of the optimization process.
        Default is False, which means only the optimized parameters are returned. If all
        parameters of all transformations are masked, the result output will be None.

    return_history : bool, optional
        If True, the function returns a history of the optimization process, including
        the parameters and the corresponding ``scipy.optimize.OptimizeResult`` object at
        each iteration. Default is False, which means the history is not returned. If
        all parameters of all transformations are masked, the history output will be
        None.

    Returns
    -------
    parameters : Tuple[numpy.ndarray, ...]
        A tuple of :math:`N_T` arrays of optimized parameters for each transformation
        with shape (n_params,). Each array contains both the optimized parameters
        (corresponding to True values in the `mask`) and the fixed parameters
        (corresponding to False values in the `mask`), where the fixed parameters are
        equal to their initial values.

    result : scipy.optimize.OptimizeResult, optional
        The result of the optimization process containing information about the
        convergence of the optimization. Returned only if `return_result` is True.

        .. warning::

            Only contains a concatenated array of the parameters
            that were optimized (i.e., the parameters corresponding to True values
            in the `mask`), and not the full parameter vector of each transformation.

    history : List[Tuple[Tuple[numpy.ndarray, ...], scipy.optimize.OptimizeResult]], optional
        A history of the optimization process including a tuple of parameters for each
        transformation with shape (n_params,) and the corresponding scipy OptimizeResult
        object at each iteration. Returned only if `return_history` is True.

        .. warning::

            The OptimizeResult objects in the history only contain a concatenated array
            of the parameters that were optimized (i.e., the parameters corresponding to
            True values in the `mask`), and not the full parameter vector of each
            transformation.

    See Also
    --------
    scipy.optimize.least_squares
        For more information about the optimization method.

    pycvcam.optimize.optimize_parameters_least_squares
        For optimizing the parameters of a single transformation.

    """
    # -------------
    # Input Formats Check
    # -------------
    if not isinstance(transforms, Sequence):
        raise TypeError(
            f"transforms must be a sequence of Transform objects, got {type(transforms)}"
        )
    if not all(isinstance(t, Transform) for t in transforms):
        raise TypeError(
            f"All elements of transforms must be instances of Transform, "
            f"got {[type(t) for t in transforms]}"
        )
    n_transforms = len(transforms)

    if not isinstance(chains, Sequence):
        raise TypeError(
            f"chains must be a sequence of sequences of integers, got {type(chains)}"
        )
    if not all(isinstance(c, Sequence) for c in chains):
        raise TypeError(
            f"All elements of chains must be sequences of integers, "
            f"got {[type(c) for c in chains]}"
        )
    if not all(
        all(isinstance(i, Integral) and 0 <= i < len(transforms) for i in c)
        for c in chains
    ):
        raise ValueError(
            f"All elements of chains must be sequences of valid indices corresponding "
            f"to the transformations in transforms. Got chains {chains} and number of "
            f"transformations {len(transforms)}."
        )
    if not all(len(c) > 0 for c in chains):
        raise ValueError(f"All chains must be non-empty. Got chains {chains}.")
    if not all(len(set(c)) == len(c) for c in chains):
        raise ValueError(
            f"All chains must not contain duplicate indices. Got chains {chains}."
        )
    n_chains = len(chains)

    if not isinstance(input_points, Sequence):
        raise TypeError(
            f"input_points must be a sequence of arrays, got {type(input_points)}"
        )
    if not len(input_points) == n_chains:
        raise ValueError(
            f"input_points must have the same length as chains, got {len(input_points)} "
            f"and {n_chains} respectively."
        )
    input_points = [numpy.asarray(p, dtype=numpy.float64) for p in input_points]

    if not isinstance(output_points, Sequence):
        raise TypeError(
            f"output_points must be a sequence of arrays, got {type(output_points)}"
        )
    if not len(output_points) == n_chains:
        raise ValueError(
            f"output_points must have the same length as chains, got {len(output_points)} "
            f"and {n_chains} respectively."
        )
    output_points = [numpy.asarray(p, dtype=numpy.float64) for p in output_points]

    for i, (in_p, out_p) in enumerate(zip(input_points, output_points)):
        if in_p.ndim < 2 or out_p.ndim < 2:
            raise ValueError(
                f"Input and output points must have at least 2 dimensions, got "
                f"{in_p.ndim} and {out_p.ndim} dimensions respectively for chain {i}."
            )
        if in_p.shape[-1] != transforms[chains[i][0]].input_dim:
            raise ValueError(
                f"Last dimension of input points must be {transforms[chains[i][0]].input_dim}, "
                f"got {in_p.shape[-1]} for chain {i}."
            )
        if out_p.shape[-1] != transforms[chains[i][-1]].output_dim:
            raise ValueError(
                f"Last dimension of output points must be {transforms[chains[i][-1]].output_dim}, "
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
        input_points[i] = in_p.reshape(-1, transforms[chains[i][0]].input_dim)
        output_points[i] = out_p.reshape(-1, transforms[chains[i][-1]].output_dim)

    if guess is None:
        guess = [None for _ in range(n_transforms)]
    if not isinstance(guess, Sequence):
        raise TypeError(
            f"guess must be a sequence of arrays or None, got {type(guess)}"
        )
    if not len(guess) == n_transforms:
        raise ValueError(
            f"guess must have the same length as transforms, got {len(guess)} and "
            f"{n_transforms} respectively."
        )
    for i, (g, t) in enumerate(zip(guess, transforms)):
        if g is None and not t.is_set():
            raise ValueError(
                f"Initial guess for the parameters of transformation {i} is required "
                f"when the current parameters of the transformation are not set."
            )
        elif g is None and t.is_set():
            g = t.parameters.copy()
        else:
            g = numpy.asarray(g, dtype=numpy.float64)
        if g.ndim != 1 or g.size != t.n_params:
            raise ValueError(
                f"Guess for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {g.ndim} dimensions and {g.size} parameters."
            )
        guess[i] = g

    if mask is None:
        mask = [None for _ in range(n_transforms)]
    if not isinstance(mask, Sequence):
        raise TypeError(f"mask must be a sequence of arrays or None, got {type(mask)}")
    if not len(mask) == n_transforms:
        raise ValueError(
            f"mask must have the same length as transforms, got {len(mask)} and "
            f"{n_transforms} respectively."
        )
    for i, (m, t) in enumerate(zip(mask, transforms)):
        if m is None:
            m = numpy.ones(t.n_params, dtype=bool)
        else:
            m = numpy.asarray(m, dtype=bool)
        if m.ndim != 1 or m.size != t.n_params:
            raise ValueError(
                f"Mask for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {m.ndim} dimensions and {m.size} parameters."
            )
        mask[i] = m

    if scale is None:
        scale = [None for _ in range(n_transforms)]
    if not isinstance(scale, Sequence):
        raise TypeError(
            f"scale must be a sequence of arrays or None, got {type(scale)}"
        )
    if not len(scale) == n_transforms:
        raise ValueError(
            f"scale must have the same length as transforms, got {len(scale)} and "
            f"{n_transforms} respectively."
        )
    for i, (s, t) in enumerate(zip(scale, transforms)):
        if s is None:
            s = numpy.ones(t.n_params, dtype=numpy.float64)
        else:
            s = numpy.asarray(s, dtype=numpy.float64)
        if s.ndim != 1 or s.size != t.n_params:
            raise ValueError(
                f"Scale for transformation {i} must be a 1D array with {t.n_params} "
                f"parameters, got {s.ndim} dimensions and {s.size} parameters."
            )
        scale[i] = s

    if bounds is None:
        bounds = [None for _ in range(n_transforms)]
    if not isinstance(bounds, Sequence):
        raise TypeError(
            f"bounds must be a sequence of arrays or None, got {type(bounds)}"
        )
    if not len(bounds) == n_transforms:
        raise ValueError(
            f"bounds must have the same length as transforms, got {len(bounds)} and "
            f"{n_transforms} respectively."
        )
    for i, (b, t) in enumerate(zip(bounds, transforms)):
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
        if not all(b[0, j] <= guess[i][j] <= b[1, j] for j in range(t.n_params)):
            raise ValueError(
                f"Initial guess for transformation {i} must be within the bounds for each parameter. "
                f"Got guess {guess[i]} and bounds {b}."
            )
        bounds[i] = b

    if transform_kwargs is None:
        transform_kwargs = [None for _ in range(n_transforms)]
    if not isinstance(transform_kwargs, Sequence):
        raise TypeError(
            f"transform_kwargs must be a sequence of dictionaries or None, got "
            f"{type(transform_kwargs)}"
        )
    if not len(transform_kwargs) == n_transforms:
        raise ValueError(
            f"transform_kwargs must have the same length as transforms, got "
            f"{len(transform_kwargs)} and {n_transforms} respectively."
        )
    for i, tk in enumerate(transform_kwargs):
        if tk is None:
            tk = {}
        if not isinstance(tk, dict):
            raise TypeError(
                f"transform_kwargs for transformation {i} must be a dictionary or None, got {type(tk)}"
            )
        transform_kwargs[i] = tk

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
            "linear",
            "soft_l1",
            "huber",
            "cauchy",
            "arctan",
        ]:
            raise ValueError(
                f"loss must be one of 'linear', 'soft_l1', 'huber', 'cauchy', or 'arctan', got {loss}"
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
    if all(t.n_params == 0 for t in transforms):
        out = tuple(numpy.zeros((0,), dtype=numpy.float64) for _ in transforms)
        if return_result and return_history:
            return out, None, None
        elif return_result:
            return out, None
        elif return_history:
            return out, None
        else:
            return out

    if all(not any(m) for m in mask):
        out = tuple(guess[i].copy() for i in range(n_transforms))
        if return_result and return_history:
            return out, None, None
        elif return_result:
            return out, None
        elif return_history:
            return out, None
        else:
            return out

    # -------------
    # Optimization
    # -------------
    object_classes = tuple(copy.deepcopy(t) for t in transforms)

    residuals_func, jacobian_func, callback, get_history = (
        _build_optimize_chain_parameters_lsq_functions(
            object_classes,
            chains,
            input_points,
            output_points,
            mask,
            guess,
            transform_kwargs,
            return_history,
            max_iterations,
            max_time,
            filter_nans,
        )
    )

    # Crop the problem to the optimized parameters
    params_initial = numpy.concatenate(
        [guess[i][mask[i]] for i in range(n_transforms)]
    )  # shape (sum(n_reduced_parameters),)
    params_bounds = numpy.concatenate(
        [bounds[i][:, mask[i]] for i in range(n_transforms)], axis=1
    )  # shape (2, sum(n_reduced_parameters))
    params_scale = numpy.concatenate(
        [scale[i][mask[i]] for i in range(n_transforms)]
    )  # shape (sum(n_reduced_parameters),)

    if verbose_level >= 3:
        _pretext = ""
        n_p = [mask[i].sum() for i in range(n_transforms)]
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

        _study_jacobian_least_squares(
            residuals_func(params_initial),
            jacobian_func(params_initial).toarray(),
            params_initial,
            _pretext,
            _start=True,
        )

    # Run the least squares optimization
    result = scipy.optimize.least_squares(
        fun=residuals_func,
        x0=params_initial,
        jac=jacobian_func,
        bounds=params_bounds,
        x_scale=params_scale,
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
            residuals_func(result.x),
            jacobian_func(result.x).toarray(),
            result.x,
            _pretext,
            _start=False,
        )

    parameters = []
    index = 0
    for i in range(n_transforms):
        p = guess[i].copy()
        p[mask[i]] = result.x[index : index + numpy.sum(mask[i])]
        parameters.append(p)
        index += numpy.sum(mask[i])
    parameters = tuple(parameters)  # len n_transforms, each shape (n_params,)

    if return_result and return_history:
        return parameters, result, get_history()
    elif return_result:
        return parameters, result
    elif return_history:
        return parameters, get_history()
    else:
        return parameters
