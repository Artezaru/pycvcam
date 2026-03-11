from typing import Optional, Sequence, Tuple, Callable, Dict
from numbers import Real, Integral
from numpy.typing import ArrayLike

import time
import scipy
import numpy
from ..core.transform import Transform
from ..core.transform_composition import TransformComposition


def _solve_optimize_chains_gauss_newton(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[int]],
    seq_outputs: Sequence[numpy.ndarray],
    guess: numpy.ndarray,
    seq_transform_kwargs: Sequence[Dict],
    max_iterations: Optional[int],
    max_time: Optional[int],
    ftol: Optional[Real],
    xtol: Optional[Real],
    gtol: Optional[Real],
    eps: Optional[Real],
    verbose: bool,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Optimize the input points based of the result of chains of transformations using
    the Gauss-Newton optimization method.

    .. note::

        This method does not implement bounds or scaling of the input_points,
        and does not provide a robust optimization process for handling
        non-linear problems, which can lead to divergence or convergence to local minima.

    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a tuple of :math:`N_T`
    :class:`Transform` objects, and :math:`(C_0, C_1, ..., C_{N_C-1})` be a
    tuple of :math:`N_C` chains of transformations.

    A chain :math:`C_i` is defined as a tuple of indices corresponding to the
    transformations in the chain. For example:

    .. code-block:: console

        C_0 = (1, 4, 8) -----> C_0(X) = T_8 ∘ T_4 ∘ T_1(X)

    We search :math:`X_i` such that the transformed points :math:`C_i(X_i; p)` match the
    output points :math:`Y_i` for each chain :math:`C_i`, where :math:`p` is the vector
    of parameters of the transformations in the chain.

    The residual function for each chain :math:`C_i` is defined as:

    .. math::

        R_i(X_i) = Y_i - C_i(X_i; p)

    .. math::

        J_i(X_i) = -\frac{\partial C_i(X_i; p)}{\partial X_i} = \frac{\partial R_i(X_i)}{\partial X_i}

    Then we can build the full residual function :math:`R` and Jacobian function
    :math:`J` for the least squares optimization by assembling the residuals and
    Jacobians of each chain, while respecting the structure of the chains.

    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of transformations involved in the chains.

    seq_chains : Sequence[Sequence[int]]
        A sequence of chains, where each chain is a sequence of indices corresponding to
        the transformations in `seq_transforms`.

    seq_outputs : Sequence[numpy.ndarray]
        A sequence of output points corresponding to each chain, where each element has
        shape (n_points, output_dim).

    guess : numpy.ndarray
        The initial guess for the input points of the transformations with shape
        (n_points, input_dim).

    seq_transform_kwargs : Sequence[Dict]
        A sequence of dictionaries containing additional keyword arguments for each
        transformation in `seq_transforms`. Each dictionary should have the same keys as
        the parameters of the corresponding transformation, and the values should be the
        values of those parameters to be used during the optimization.

    max_iterations : Optional[int]
        The maximum number of iterations for the optimization.
        Default is None, which means no limit on the number of iterations.

    max_time : Optional[int]
        The maximum time in seconds for the optimization.
        Default is None, which means no limit on the time.

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

    eps : Optional[Real], optional
        The convergence threshold for the optimization. The optimization process is
        stopped when the change in the cost function is less than eps.
        Default is None, which means no convergence threshold is used.

    verbose : bool
        If True, print the optimization progress and diagnostics. Default is False.


    Returns
    -------
    numpy.ndarray
        The optimized input points of the transformations with shape (n_points, input_dim).

    numpy.ndarray
        The convergence status of each point, where

        - 0 means not converged,
        - 1 means converged by ftol criterion,
        - 2 means converged by xtol criterion,
        - 3 means converged by gtol criterion,
        - 4 means converged by eps criterion,
        - 5 means diverged by NaN values,
        - 6 means maximum number of iterations reached,
        - 7 means maximum time reached.
        - inf means warning not evaluated yet.

    """
    n_points = guess.shape[0]
    input_dim = guess.shape[1]

    c_output_dims = [seq_outputs[i].shape[1] for i in range(len(seq_chains))]
    c_tc = [TransformComposition([seq_transforms[i] for i in c]) for c in seq_chains]
    c_kwargs = [list(seq_transform_kwargs[i] for i in c) for c in seq_chains]
    c_start_idx = numpy.cumsum([0] + c_output_dims[:-1])  # shape (N_C,)
    c_end_idx = numpy.cumsum(c_output_dims)  # shape (N_C,)

    def _build_R_and_J(
        current_inputs: numpy.ndarray,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        R = numpy.empty(
            (current_inputs.shape[0], sum(c_output_dims)), dtype=numpy.float64
        )
        J = numpy.empty(
            (current_inputs.shape[0], sum(c_output_dims), input_dim),
            dtype=numpy.float64,
        )

        for ic, tc in enumerate(c_tc):
            list_kwargs = c_kwargs[ic]
            out, jac, _ = tc._transform(
                current_inputs, dx=True, dp=False, list_kwargs=list_kwargs
            )  # shape (Nopt, c_output_dim), (Nopt, c_output_dim, input_dim), None

            if J is None:
                raise ValueError(
                    f"Jacobian with respect to the input points is not available for chain {ic}. Please implement the _transform method to return the Jacobian with respect to the input points."
                )

            R[:, c_start_idx[ic] : c_end_idx[ic]] = seq_outputs[ic][active_idx, :] - out
            J[:, c_start_idx[ic] : c_end_idx[ic], :] = -jac

        return R, J

    start_time = time.time()
    iteration = 0
    input_points = guess.copy()
    input_conv = numpy.full((n_points,), numpy.inf, dtype=numpy.float64)
    active_idx = numpy.arange(n_points)
    still_active = numpy.ones(n_points, dtype=bool)
    cost = None
    optimality = None
    last_cost = None
    last_optimality = None
    R = None
    J = None
    JTR = None
    JTJ = None
    end = False

    if verbose:
        header = f" {'Iteration':^10}   {'Total nfev':^10}  {'N points':^10}  {'Cost /point':^15}  {'Step norm /points':^15}   {'Optimality /points':^15}"
        print(header)

    while not end:
        current_inputs = input_points[active_idx, :]  # shape (Nopt, dim)
        current_conv = input_conv[active_idx]  # shape (Nopt,)

        if R is None or J is None:
            R, J = _build_R_and_J(current_inputs)
        else:
            R = R[still_active, :]
            J = J[still_active, :, :]

        if JTR is None:
            JTR = (
                -J.transpose(0, 2, 1) @ R[..., None]
            )  # shape (Nopt, input_dim, 1) | warning (-) sign for the gradient
        else:
            JTR = JTR[still_active, :, :]

        if JTJ is None:
            JTJ = J.transpose(0, 2, 1) @ J  # shape (Nopt, input_dim, input_dim)
        else:
            JTJ = JTJ[still_active, :, :]

        delta = numpy.linalg.solve(JTJ, JTR)[..., 0]  # shape (Nopt, input_dim)
        input_points[active_idx, :] += delta

        if cost is None:
            cost = 0.5 * numpy.einsum("ij,ij->i", R, R)  # shape (Nopt,)
        else:
            cost = cost[still_active]  # shape (Nopt,)
        last_cost = cost

        if optimality is None:
            optimality = numpy.linalg.norm(
                JTR, axis=(1, 2), ord=numpy.inf
            )  # shape (Nopt,)
        else:
            optimality = optimality[still_active]  # shape (Nopt,)
        last_optimality = optimality

        R, J, JTR, JTJ, cost, optimality = (
            None,
            None,
            None,
            None,
            None,
            None,
        )  # Invalidate cached values

        if verbose and iteration == 0:
            cost_per_point = numpy.linalg.norm(last_cost) / numpy.sqrt(last_cost.size)
            optimality_per_point = numpy.linalg.norm(last_optimality) / numpy.sqrt(
                last_optimality.size
            )
            print(
                f" {iteration:^10}   {iteration+1:^10}   {active_idx.size:^10}   {cost_per_point:^15.3e}   {'':^15}   {optimality_per_point:^15.3e}"
            )

        if verbose or ftol is not None or eps is not None:
            R, J = _build_R_and_J(input_points[active_idx, :])
            cost = 0.5 * numpy.einsum("ij,ij->i", R, R)  # shape (Nopt,)
            cost_reduction = last_cost - cost  # shape (Nopt,)

        if verbose or gtol is not None:
            if R is None or J is None:
                R, J = _build_R_and_J(input_points[active_idx, :])
            JTR = (
                -J.transpose(0, 2, 1) @ R[..., None]
            )  # Warning (-) sign for the gradient
            optimality = numpy.linalg.norm(JTR, axis=(1, 2), ord=numpy.inf)

        if verbose or xtol is not None:
            step_norm = numpy.linalg.norm(delta, axis=1)  # shape (Nopt,)
            norm = numpy.linalg.norm(
                input_points[active_idx, :], axis=1
            )  # shape (Nopt,)

        if verbose:
            cost_per_point = numpy.linalg.norm(cost) / numpy.sqrt(cost.size)
            optimality_per_point = numpy.linalg.norm(optimality) / numpy.sqrt(
                optimality.size
            )
            step_norm_per_point = numpy.linalg.norm(step_norm) / numpy.sqrt(
                step_norm.size
            )
            print(
                f" {iteration+1:^10}   {iteration+1:^10}   {active_idx.size:^10}   {cost_per_point:^15.3e}   {step_norm_per_point:^15.3e}   {optimality_per_point:^15.3e}"
            )

        if ftol is not None:
            ftol_criterion = (cost_reduction < ftol * cost) & (cost_reduction >= 0)

        if xtol is not None:
            xtol_criterion = step_norm < xtol * (xtol + norm)

        if gtol is not None:
            gtol_criterion = optimality < gtol

        if eps is not None:
            eps_criterion = cost < eps

        nan_criterion = numpy.any(numpy.isnan(delta), axis=1)

        still_active = numpy.ones_like(active_idx, dtype=bool)
        if ftol is not None:
            still_active &= ~ftol_criterion
            current_conv[ftol_criterion] = numpy.minimum(
                current_conv[ftol_criterion], 1
            )
        if xtol is not None:
            still_active &= ~xtol_criterion
            current_conv[xtol_criterion] = numpy.minimum(
                current_conv[xtol_criterion], 2
            )
        if gtol is not None:
            still_active &= ~gtol_criterion
            current_conv[gtol_criterion] = numpy.minimum(
                current_conv[gtol_criterion], 3
            )
        if eps is not None:
            still_active &= ~eps_criterion
            current_conv[eps_criterion] = numpy.minimum(current_conv[eps_criterion], 4)
        still_active &= ~nan_criterion
        current_conv[nan_criterion] = 5

        if max_iterations is not None and iteration >= max_iterations - 1:
            current_conv = numpy.minimum(current_conv, 6)
            if verbose:
                print(
                    f"Maximum number of iterations {max_iterations} reached, stopping optimization."
                )
            end = True

        if max_time is not None and (time.time() - start_time) >= max_time:
            current_conv = numpy.minimum(current_conv, 7)
            if verbose:
                print(
                    f"Maximum time {max_time} seconds reached, stopping optimization."
                )
            end = True

        input_conv[active_idx] = current_conv
        active_idx = active_idx[still_active]

        if active_idx.size == 0:
            if verbose:
                print(
                    f"All points have converged, stopping optimization at iteration {iteration+1}."
                )
            end = True

        iteration += 1

    return input_points, input_conv


def optimize_chains_input_points_gn(
    seq_transforms: Sequence[Transform],
    seq_chains: Sequence[Sequence[Integral]],
    seq_outputs: Sequence[ArrayLike],
    guess: ArrayLike,
    *,
    transpose: bool = False,
    seq_transform_kwargs: Sequence[Dict] = None,
    max_iterations: Optional[int] = None,
    max_time: Optional[int] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    eps: Optional[Real] = None,
    verbose: bool = False,
    return_convergence: bool = False,
) -> numpy.ndarray:
    r"""
    Optimize the input points based of the result of chains of transformations using
    the Gauss-Newton optimization method.

    .. note::

        This method does not implement bounds or scaling of the input_points,
        and does not provide a robust optimization process for handling
        non-linear problems, which can lead to divergence or convergence to local minima.

    .. warning::

        This method can only be used if the dimensions check input_dim <= sum(output_dim).


    Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a tuple of :math:`N_T`
    :class:`Transform` objects, and :math:`(C_0, C_1, ..., C_{N_C-1})` be a
    tuple of :math:`N_C` chains of transformations.

    A chain :math:`C_i` is defined as a tuple of indices corresponding to the
    transformations in the chain. For example:

    .. code-block:: console

        C_0 = (1, 4, 8) -----> C_0(X) = T_8 ∘ T_4 ∘ T_1(X)

    We search :math:`X_i` such that the transformed points :math:`C_i(X_i; p)` match the
    output points :math:`Y_i` for each chain :math:`C_i`, where :math:`p` is the vector
    of parameters of the transformations in the chain.

    The residual function for each chain :math:`C_i` is defined as:

    .. math::

        R_i(X_i) = Y_i - C_i(X_i; p)

    .. math::

        J_i(X_i) = -\frac{\partial C_i(X_i; p)}{\partial X_i} = \frac{\partial R_i(X_i)}{\partial X_i}



    Parameters
    ----------
    seq_transforms : Sequence[Transform]
        A sequence of transformations involved in the chains.

    seq_chains : Sequence[Sequence[Integral]]
        A sequence of chains, where each chain is a sequence of indices corresponding to
        the transformations in `seq_transforms`.

    seq_outputs : Sequence[numpy.ndarray]
        A sequence of output points corresponding to each chain, where each element has
        shape (n_points, output_dim).

    guess : numpy.ndarray
        The initial guess for the input points of the transformations with shape
        (n_points, input_dim).

    transpose : bool, optional
        If True, the output points are transposed to shape (output_dim, n_points)
        before optimization, and the optimized input points are transposed back to
        shape (input_dim, n_points) before returning. Default is False.

    seq_transform_kwargs : Sequence[Dict]
        A sequence of dictionaries containing additional keyword arguments for each
        transformation in `seq_transforms`. Each dictionary should have the same keys as
        the parameters of the corresponding transformation, and the values should be the
        values of those parameters to be used during the optimization.

    max_iterations : Optional[int]
        The maximum number of iterations for the optimization.
        Default is None, which means no limit on the number of iterations.

    max_time : Optional[int]
        The maximum time in seconds for the optimization.
        Default is None, which means no limit on the time.

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
        be overridden by the user-specified value. Default is False.

    eps : Optional[Real], optional
        The convergence threshold for the optimization. The optimization process is
        stopped when the change in the cost function is less than eps.
        Default is None, which means no convergence threshold is used.

    verbose : bool
        If True, print the optimization progress and diagnostics. Default is False.

    return_convergence : bool
        If True, the function returns a tuple of (optimized_input_points, convergence_status), where
        convergence_status is a numpy array of shape (n_points,) indicating the convergence status of each point.


    Returns
    -------
    numpy.ndarray
        The optimized input points of the transformation with shape (..., dim).

    numpy.ndarray, optional
        The convergence status of each point, where:

        - 0 means not converged,
        - 1 means converged by ftol criterion,
        - 2 means converged by xtol criterion,
        - 3 means converged by gtol criterion,
        - 4 means converged by eps criterion,
        - 5 means diverged by NaN values,
        - 6 means maximum number of iterations reached,
        - 7 means maximum time reached.
        - inf means warning not evaluated yet.



    See Also
    --------
    optimize_input_points_gn :
        Optimize the input points of a single transformation using the Gauss-Newton optimization method.

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

    guess = numpy.asarray(guess, dtype=numpy.float64)

    if not all([seq_outputs.ndim >= 2 for seq_outputs in seq_outputs]):
        raise ValueError(
            f"All elements of seq_outputs must be at least 2D arrays with shape (..., dim), got {[p.shape for p in seq_outputs]}"
        )
    if not guess.ndim >= 2:
        raise ValueError(
            f"guess must be at least a 2D array with shape (..., dim), got {guess.shape}"
        )

    if not isinstance(transpose, bool):
        raise TypeError(f"transpose must be a boolean, got {type(transpose)}")

    if transpose:
        seq_outputs = [numpy.moveaxis(p, 0, -1) for p in seq_outputs]
        guess = numpy.moveaxis(guess, 0, -1)

    seq_outputs_dim = [p.shape[-1] for p in seq_outputs]
    shape = guess.shape
    seq_outputs = [p.reshape(-1, seq_outputs_dim[i]) for i, p in enumerate(seq_outputs)]
    guess = guess.reshape(-1, shape[-1])

    if not all(p.shape[0] == guess.shape[0] for p in seq_outputs):
        raise ValueError(
            f"All elements of seq_outputs must have the same number of points as guess. Got {[p.shape[0] for p in seq_outputs]} and {guess.shape[0]} respectively."
        )
    if not all(
        p.shape[1] == seq_transforms[c[-1]].output_dim
        for p, c in zip(seq_outputs, seq_chains)
    ):
        raise ValueError(
            f"All elements of seq_outputs must have the same output dimension as the output dimension of the last transformation in the corresponding chain. Got {[p.shape[1] for p in seq_outputs]} and {[seq_transforms[c[-1]].output_dim for c in seq_chains]} respectively."
        )
    if not all(guess.shape[1] == seq_transforms[c[0]].input_dim for c in seq_chains):
        raise ValueError(
            f"guess must have the same input dimension as the input dimension of the first transformation in each chain. Got {guess.shape[1]} and {[seq_transforms[c[0]].input_dim for c in seq_chains]} respectively."
        )

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

    if eps is not None:
        if not isinstance(eps, Real) or eps <= 0:
            raise TypeError(f"eps must be a positive float, got {eps}")
        eps = float(eps)

    if not isinstance(verbose, bool):
        raise TypeError(f"verbose must be a boolean, got {type(verbose)}")
    verbose = bool(verbose)

    if ftol is None and xtol is None and gtol is None and eps is None:
        raise ValueError(
            "At least one of ftol, xtol, gtol, or eps must be specified for stopping criteria."
        )

    if not isinstance(return_convergence, bool):
        raise TypeError(
            f"return_convergence must be a boolean, got {type(return_convergence)}"
        )

    if guess.shape[1] > sum(p.shape[1] for p in seq_outputs):
        raise ValueError(
            f"The input dimension of the transformations must be less than the sum of the output dimensions of the chains. Got {guess.shape[1]} and {sum(p.shape[1] for p in seq_outputs)} respectively."
        )

    out, conv = _solve_optimize_chains_gauss_newton(
        seq_transforms=seq_transforms,
        seq_chains=seq_chains,
        seq_outputs=seq_outputs,
        guess=guess,
        seq_transform_kwargs=seq_transform_kwargs,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        eps=eps,
        verbose=verbose,
    )

    if transpose:
        out = numpy.moveaxis(out, -1, 0)

    if return_convergence:
        return out.reshape(shape), conv
    return out.reshape(shape)


def optimize_input_points_gn(
    transform: Transform,
    output_points: ArrayLike,
    guess: ArrayLike,
    *,
    transpose: bool = False,
    max_iterations: Optional[Integral] = None,
    max_time: Optional[Real] = None,
    ftol: Optional[Real] = None,
    xtol: Optional[Real] = None,
    gtol: Optional[Real] = None,
    auto: bool = False,
    eps: Optional[Real] = None,
    verbose: bool = False,
    return_convergence: bool = False,
) -> numpy.ndarray:
    r"""
    Optimize the input points of the transformation using the given output points.

    Estimate the optimized input points of the transformation such that the transformed input points match the output points.

    .. warning::

        This method can only be used if the dimensions check input_dim <= output_dim.

    Lets consider a set of output points :math:`X_O` with shape (..., dim) and a set of input points :math:`\vec{X}_I` with shape (..., input_dim).
    We search :math:`\vec{X}_I = \vec{X}_{I_0} + \delta \vec{X}_I` such that:

    .. math::

        \vec{X}_O = \text{Transform}(\vec{X}_I, \lambda) = T(\vec{X}_{I_0} + \delta \vec{X}_I, \lambda)

    We have:

    .. math::

        \nabla_{X} T (\vec{X}_{I_0}, \lambda) \delta \vec{X}_I = \vec{X}_O - T(\vec{X}_{I_0}, \lambda)

    The corrections are computed using the following equations :

    .. math::

        J \delta \vec{X}_I = R

    Where :math:`J = \nabla_{X} T (\vec{X}_{I_0}, \lambda)` is the Jacobian matrix of the transformation with respect to the input points, and :math:`R = \vec{X}_O - T(\vec{X}_{I_0}, \lambda)` is the residual vector.
    :math:`\vec{X}_{I_0}` is the initial guess for the input points.

    .. note::

        The ``auto`` parameter sets the stopping criteria (``ftol``, ``xtol``, and ``gtol``) to ``1e-8``. If any of the stopping criteria is already specified, it will be overridden by the user-specified value.

    Parameters
    ----------
    transform : Transform
        The transformation object used for optimization.

    output_points : ArrayLike
        The output points to be matched. Shape (..., dim) (or (dim, ...) if `transpose` is True).
        tional[ArrayLike], optional

    guess : ArrayLike
        The initial guess for the input points of the transformation with shape (..., dim).

    transpose : bool, optional
        If True, the output points are transposed to shape (dim, ...) before optimization,
        and the optimized input points are transposed back to shape (input_dim, ...) before returning. Default is False.

    max_iterations : Optional[int]
        The maximum number of iterations for the optimization.
        Default is None, which means no limit on the number of iterations.

    max_time : Optional[int]
        The maximum time in seconds for the optimization.
        Default is None, which means no limit on the time.

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
        be overridden by the user-specified value. Default is False.

    eps : Optional[Real], optional
        The convergence threshold for the optimization. The optimization process is
        stopped when the change in the cost function is less than eps.
        Default is None, which means no convergence threshold is used.

    verbose : bool
        If True, print the optimization progress and diagnostics. Default is False.

    return_convergence : bool
        If True, the function returns a tuple of (optimized_input_points, convergence_status), where
        convergence_status is a numpy array of shape (n_points,) indicating the convergence status of each
        point.


    Returns
    -------
    numpy.ndarray
        The optimized input points of the transformation with shape (..., dim).

    numpy.ndarray, optional
        The convergence status of each point, where:

        - 0 means not converged,
        - 1 means converged by ftol criterion,
        - 2 means converged by xtol criterion,
        - 3 means converged by gtol criterion,
        - 4 means converged by eps criterion,
        - 5 means diverged by NaN values,
        - 6 means maximum number of iterations reached,
        - 7 means maximum time reached,
        - inf means warning not evaluated yet.


    See Also
    --------
    optimize_chains_input_points_gn :
        Optimize the input points based of the result of chains of transformations using the Gauss-Newton optimization method.

    """
    return optimize_chains_input_points_gn(
        seq_transforms=[transform],
        seq_chains=[(0,)],
        seq_outputs=[output_points],
        guess=guess,
        transpose=transpose,
        max_iterations=max_iterations,
        max_time=max_time,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        auto=auto,
        eps=eps,
        verbose=verbose,
        return_convergence=return_convergence,
    )


def optimize_input_points(
    transform: Transform,
    output_points: numpy.ndarray,
    guess: Optional[numpy.ndarray] = None,
    *,
    transpose: bool = False,
    max_iter: int = 10,
    eps: float = 1e-8,
    verbose: bool = False,
    _skip: bool = False,
) -> numpy.ndarray:
    r"""
    Optimize the input points of the transformation using the given output points.

    .. deprecated:: 2.1.8

        Consider using the :func:`optimize_input_points_gn` function instead,
        which provides a more robust optimization process for handling non-linear problems,
        and allows for more flexible stopping criteria.

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

    The corrections are computed using the following equations :

    .. math::

        J \delta \vec{X}_I = R

    Where :math:`J = \nabla_{X} T (\vec{X}_{I_0}, \lambda)` is the Jacobian matrix of the transformation with respect to the input points, and :math:`R = \vec{X}_O - T(\vec{X}_{I_0}, \lambda)` is the residual vector.

    :math:`\vec{X}_{I_0}` is the initial guess for the input points, if None, it use the output points as the initial guess.

    .. note::

        The ``_skip`` parameter is used to skip the checks for the transformation parameters and assume the output points are given in the (n_points, dim) float format.
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
        If True, skip the checks for the transformation parameters and assume the output points are given in the (n_points, dim) float format.
        The guess must be given in the (n_points, dim) float format.
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
        distortion = Cv2Distortion(parameters=numpy.array([1e-3, 2e-3, 1e-3, 1e-4, 2e-3]), n_params=5)

        # Generate some random distorted points
        distorted_points = numpy.random.rand(10, 2)  # Random 2D points

        # Optimize the input points to match the distorted points
        optimized_input_points = optimize_input_points(distortion, distorted_points) # shape (10, 2)
        print("Optimized Input Points:", optimized_input_points)

    """
    if not isinstance(transform, Transform):
        raise TypeError(
            f"transform must be an instance of Transform, got {type(transform)}"
        )

    if transform.input_dim != transform.output_dim:
        raise ValueError(
            f"Input dimension ({transform.input_dim}) must be equal to output dimension ({transform.output_dim}) for this method to work."
        )
    dim = transform.input_dim  # Since input_dim == output_dim

    if not _skip:
        # Check the boolean flags
        if not isinstance(transpose, bool):
            raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise TypeError(
                f"max_iter must be an integer greater than 0, got {max_iter}"
            )
        if not isinstance(eps, float) or eps <= 0:
            raise TypeError(f"eps must be a positive float, got {eps}")
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose must be a boolean, got {type(verbose)}")

        # Check if the transformation is set
        if not transform.is_set():
            raise ValueError(
                "Transformation parameters are not set. Please set the parameters before optimizing."
            )

        # Convert output points to float
        output_points = numpy.asarray(output_points, dtype=numpy.float64)

        # Check the guess
        if guess is not None:
            guess = numpy.asarray(guess, dtype=numpy.float64)
        else:
            # Use the output points as the initial guess
            guess = output_points.copy()  # shape (..., dim)

        # Check the shape of the output points
        if output_points.ndim < 2:
            raise ValueError(
                f"Output points must have at least 2 dimensions, got {output_points.ndim} dimensions."
            )
        if guess.ndim < 2:
            raise ValueError(
                f"Guess must have at least 2 dimensions, got {guess.ndim} dimensions."
            )

        # Transpose the output points if requested
        if transpose:
            output_points = numpy.moveaxis(
                output_points, 0, -1
            )  # (dim, ...) -> (..., dim)
            guess = numpy.moveaxis(guess, 0, -1)  # (dim, ...) -> (..., dim)

        # Flatten the output points to 2D for processing
        shape = output_points.shape  # (..., dim)
        output_points = output_points.reshape(-1, dim)  # (..., dim) -> (n_points, dim)
        guess = guess.reshape(-1, dim)  # (..., dim) -> (n_points, dim)

        # Check the number of points
        if output_points.shape[0] != guess.shape[0]:
            raise ValueError(
                f"Output points and guess must have the same number of points, got {output_points.shape[0]} and {guess.shape[0]} points respectively."
            )
        if output_points.shape[0] == 0:
            raise ValueError("Output points and guess must have at least one point.")

        if output_points.shape[-1] != dim:
            raise ValueError(
                f"Output points must have {dim} dimensions, got {output_points.shape[-1]} dimensions."
            )
        if guess.shape[-1] != dim:
            raise ValueError(
                f"Guess must have {dim} dimensions, got {guess.shape[-1]} dimensions."
            )

    # Initialize the guess for the input points
    n_points = output_points.shape[0]

    # Prepare the output array:
    input_points = guess
    active_idx = numpy.arange(n_points)  # shape (n_points,)

    # Run the iterative algorithm
    for it in range(max_iter):
        # Check if there are any points left in computation
        if active_idx.size == 0:
            if verbose:
                print(
                    f"All points are invalid. Stopping optimization at iteration {it}."
                )
            break

        current_inputs = input_points[active_idx, :]  # shape (Nopt, dim)

        # Compute the transformation of the input points and the Jacobian with respect to the input points
        out, J, _ = transform._transform(
            current_inputs, dx=True, dp=False
        )  # shape (Nopt, dim), (Nopt, dim, dim), None

        # Check if the jacobian_dx is None
        if J is None:
            raise ValueError(
                "Jacobian with respect to the input points is not available. Please implement the _transform method to return the Jacobian with respect to the input points."
            )

        R = output_points[active_idx, :] - out  # shape (Nopt, dim)
        diff = numpy.linalg.norm(R, axis=1)  # shape (Nopt,)
        still_active = diff > eps  # shape (Nopt,)

        if not numpy.any(still_active):
            if verbose:
                print(f"Optimization converged in {it} iterations.")
            break

        new_active_idx = active_idx[still_active]  # shape (NewNopt,)

        # Solve the linear system to find the delta
        # delta_itk = numpy.array(
        #     [scipy.linalg.solve(J[i], R[i]) for i in range(Nopt)], dtype=numpy.float64
        # )  # shape (Nopt, dim)
        delta = numpy.linalg.solve(J[still_active, :, :], R[still_active, :, None])[
            ..., 0
        ]  # shape (Nopt, dim)

        # Update the input points
        input_points[new_active_idx, :] += delta
        if verbose:
            print(
                f"Iteration {it+1}: {new_active_idx.size} valid points out of {n_points}. Mean residual: {numpy.mean(diff[still_active])}, Max residual: {numpy.max(diff[still_active])}"
            )

        # Update the mask and active indices for the next iteration
        active_idx = new_active_idx

    # Return the optimized input points
    if not _skip:
        input_points = input_points.reshape(
            *shape[:-1], dim
        )  # (n_points, dim) -> (..., dim)

        if transpose:
            input_points = numpy.moveaxis(
                input_points, -1, 0
            )  # (..., dim) -> (dim, ...)

    return input_points
