from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, ClassVar, Any
import numpy
from numpy.typing import ArrayLike
import datetime
import json

from .transform_result import TransformResult
from ..__version__ import __version__


class Transform(ABC):
    r"""
    Transform is the base class to manage transformations from
    :math:`\mathbb{R}^{\text{input_dim}}` to :math:`\mathbb{R}^{\text{output_dim}}`.

    A tranformation is a function that maps points from an input space to an output
    space. The transformation is defined by:

    - a set of ``n_params`` parameters :math:`\{\lambda_1, \lambda_2, \ldots, \lambda_N\}` that define the transformation.
    - a set of ``n_constants`` constants that are constant for the transformation. (Can not be optimized and no jacobian with respect to these constants).

    .. math::

        \vec{X}_O = T(\vec{X}_I, \lambda_1, \lambda_2, \ldots, \lambda_N)

    where :math:`\vec{X}_O` are the output points, :math:`\vec{X}_I` are the input
    points, and :math:`\{\lambda_1, \lambda_2, \ldots, \lambda_N\}` are the parameters
    of the transformation.

    This class provides the base for all transformations. It defines the interface for
    extrinsic, distortion, and intrinsic transformations.

    .. seealso::

        - :class:`pycvcam.core.Extrinsic` for extrinsic transformations.
        - :class:`pycvcam.core.Distortion` for distortion transformations.
        - :class:`pycvcam.core.Intrinsic` for intrinsic transformations.

    Each sub-classes must implement the following methods and properties:

    - ``_input_dim``: (class attribute) The dimension of the input points as integer (example: 2 for 2D points).
    - ``_output_dim``: (class attribute) The dimension of the output points as integer (example: 2 for 2D points).
    - ``_transform``: (method) Apply the transformation to the given points with shape (n_points, input_dim) and return the transformed points with shape (n_points, output_dim), and optionally the Jacobian matrices if requested.
    - ``_inverse_transform``: (method) Apply the inverse transformation to the given points with shape (n_points, output_dim) and return the transformed points with shape (n_points, input_dim), and optionally the Jacobian matrices if requested.

    The following properties are not required but can be overwritting to provide additional information about the transformation:

    - ``parameters`` (property and setter) The parameters of the transformation in a 1D numpy array of shape (``n_params``,) or None if the transformation does not have parameters or they are not set. Default only impose 1D array of floats or None.
    - ``constants`` (property and setter) The constants of the transformation in a 1D numpy array of shape (``n_constants``,) or None if the transformation does not have constants or they are not set. Default only impose 1D array of floats or None.
    - ``parameter_names`` (property) The names of the parameters as a list of strings or None if the transformation does not have parameters or they are not set. Default is None.
    - ``constant_names`` (property) The names of the constants as a list of strings or None if the transformation does not have constants or they are not set. Default is None.
    - ``is_set``: (method) Check if the transformation is set (i.e., if the parameters are initialized). Default is to return True if the parameters and constants are not None. Default is to return True if the parameters and constants are not None.
    - ``_result_class``: (class attribute) The class used for the result of the transformation (sub-class of ``TransformResult``). Default is :class:`pycvcam.core.TransformResult`.
    - ``_inverse_result_class``: (class attribute) The class used for the result of the inverse transformation (sub-class of ``TransformResult``). Default is :class:`pycvcam.core.TransformResult`.
    - ``_get_jacobian_shorthands``: (method) A dictionary of short-hand notation for the Jacobian matrices, which can be used to add custom views of the ``jacobian_dp`` matrix with respect to the parameters of the transformation. Default is an empty dictionary.
    - ``_get_transform_aliases``: (method) A dictionary of aliases for the transformation, which can be used to add custom names for the transformation parameters. Default is an empty list.
    - ``_get_inverse_transform_aliases``: (method) A dictionary of aliases for the inverse transformation, which can be used to add custom names for the inverse transformation parameters. Default is an empty list.

    More details on the transformation methods are provided in the ``transform`` and ``inverse_transform`` methods.

    .. seealso::

        - :meth:`pycvcam.core.Transform.transform` for applying the transformation to points.
        - :meth:`pycvcam.core.Transform.inverse_transform` for applying the inverse transformation to points.
        - :class:`pycvcam.core.TransformResult` for the result of the transformation.

    .. note::

        ``...`` in the shape of the attributes indicates that the shape can have any number of leading dimensions, which is useful for batch processing of points.

    """

    __slots__ = ["_parameters", "_constants"]

    _input_dim: ClassVar[Optional[int]] = None
    _output_dim: ClassVar[Optional[int]] = None
    _result_class: ClassVar[type] = TransformResult
    _inverse_result_class: ClassVar[type] = TransformResult

    @abstractmethod
    def __init__(
        self,
        parameters: Optional[ArrayLike] = None,
        constants: Optional[ArrayLike] = None,
    ):
        self.parameters = parameters
        self.constants = constants

    # =============================================
    # Properties for Transform Class
    # =============================================
    @property
    def result_class(self) -> type:
        r"""
        [Get] the class used for the result of the transformation.

        Returns
        -------
        type
            The class used for the result of the transformation.

        """
        if not issubclass(self._result_class, TransformResult):
            raise TypeError(
                f"result_class must be a subclass of TransformResult, got {self._result_class}"
            )
        return self._result_class

    @property
    def inverse_result_class(self) -> type:
        r"""
        [Get] the class used for the result of the inverse transformation.

        Returns
        -------
        type
            The class used for the result of the inverse transformation.
        """
        if not issubclass(self._inverse_result_class, TransformResult):
            raise TypeError(
                f"inverse_result_class must be a subclass of TransformResult, got {self._inverse_result_class}"
            )
        return self._inverse_result_class

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension of the transformation.

        Returns
        -------
        int
            The number of dimensions of the input points.

        Raises
        -------
        NotImplementedError
            If the input dimension is not defined in the subclass.
        """
        if self._input_dim is None:
            raise NotImplementedError(
                "Subclasses must define the ``_input_dim`` class attribute."
            )
        return self._input_dim

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension of the transformation.

        Returns
        -------
        int
            The number of dimensions of the output points.

        Raises
        -------
        NotImplementedError
            If the output dimension is not defined in the subclass.
        """
        if self._output_dim is None:
            raise NotImplementedError(
                "Subclasses must define the ``_output_dim`` class attribute."
            )
        return self._output_dim

    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the parameters of the transformation.

        The parameters must be a 1-D numpy array of shape (``n_params``,) where ``n_params`` is the number of parameters of the transformation.

        If the transformation does not have parameters or they are not set, this property should return None.

        .. note::

            The given value is converted to a numpy array of ``dtype=numpy.float64`` if it is not None.

        Parameters
        ----------
        value : Optional[ArrayLike]
            The parameters of the transformation as a 1-D numpy array.


        Returns
        -------
        Optional[numpy.ndarray]
            The parameters of the transformation.


        Raises
        -------
        ValueError
            If the parameters are not a 1-D numpy array.

        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: Optional[ArrayLike]) -> None:
        parameters = (
            numpy.asarray(value, dtype=numpy.float64) if value is not None else None
        )
        if parameters is not None and parameters.ndim != 1:
            raise ValueError(
                f"Parameters must be a 1-D numpy array, got shape {parameters.shape}"
            )
        self._parameters = parameters

    @property
    def constants(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] to return the constants of the transformation.

        The constants must be a 1-D float numpy array of shape (``n_constants``,) where ``n_constants`` is the number of constants of the transformation.

        If the transformation does not have constants or they are not set, this property should return None.

        .. note::

            The given value is converted to a numpy array of ``dtype=numpy.float64`` if it is not None.

        Parameters
        ----------
        value : Optional[ArrayLike]
            The constants of the transformation as a 1-D numpy array.


        Returns
        -------
        Optional[numpy.ndarray]
            The constants of the transformation.


        Raises
        -------
        ValueError
            If the constants are not a 1-D numpy array.
        """
        return self._constants

    @constants.setter
    def constants(self, value: Optional[ArrayLike]) -> None:
        constants = (
            numpy.asarray(value, dtype=numpy.float64) if value is not None else None
        )
        if constants is not None and constants.ndim != 1:
            raise ValueError(
                f"Constants must be a 1-D numpy array, got shape {constants.shape}"
            )
        self._constants = constants

    @property
    def n_params(self) -> int:
        r"""
        [Get] the number of parameters of the transformation.

        .. note::

            For retro-compatibility, this property can also be accessed using the name ``Nparams``.

        Returns
        -------
        int
            The number of parameters of the transformation.
        """
        return self.parameters.size if self.parameters is not None else 0

    @property
    def Nparams(self) -> int:
        return self.n_params

    @property
    def n_constants(self) -> int:
        r"""
        [Get] the number of constants of the transformation.

        .. note::

            For retro-compatibility, this property can also be accessed using the name ``Nconstants``.

        Returns
        -------
        int
            The number of constants of the transformation.
        """
        return self.constants.size if self.constants is not None else 0

    @property
    def Nconstants(self) -> int:
        return self.n_constants

    @property
    def parameter_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters of the transformation.

        The names is a list of strings of length ``n_params`` where ``n_params`` is the number of parameters of the transformation.

        If the transformation does not have parameters should return an empty list.

        By default, the parameter names are generated as "p_0", "p_1", ..., "p_{n_params-1}". See sub-classes for more specific names.

        Returns
        -------
        List[str]
            The names of the parameters of the transformation.
        """
        return [f"p_{i}" for i in range(self.n_params)]

    @property
    def constant_names(self) -> List[str]:
        r"""
        [Get] the names of the constants of the transformation.

        The names is a list of strings of length ``n_constants`` where ``n_constants`` is the number of constants of the transformation.

        If the transformation does not have constants should return an empty list.

        By default, the constant names are generated as "c_0", "c_1", ..., "c_{n_constants-1}". See sub-classes for more specific names.

        Returns
        -------
        List[str]
            The names of the constants of the transformation.
        """
        return [f"c_{i}" for i in range(self.n_constants)]

    def copy(self) -> Transform:
        r"""
        Return a deep copy of the transformation.

        Returns
        -------
        Transform
            A deep copy of the transformation,  with the same parameters and constants.

        """
        parameters = self.parameters.copy() if self.parameters is not None else None
        constants = self.constants.copy() if self.constants is not None else None
        return self.__class__(parameters=parameters, constants=constants)

    # =============================================
    # Methods for Transform Class
    # =============================================
    def _get_jacobian_shorthands(self) -> Dict[str, Tuple[int, int, Optional[str]]]:
        r"""
        Return a dictionary of short-hand notation for the Jacobian matrices.

        This dictionary can be used to add custom views of the ``jacobian_dp`` matrix with respect to the parameters of the transformation.

        .. code-block:: python

            {
                "dk": (0, 2, "Custom Jacobian view for two first parameters related to k1 and k2"),
                "dother": (2, 4, "Custom Jacobian view for other parameters related to k3 and k4"),
            }

        Returns
        -------
        Dict[str, Tuple[int, int, Optional[str]]]
            A dictionary where keys are names of the custom Jacobian views and values are tuples containing:

            - start index (int): The starting index of the parameters to include in the custom Jacobian view.
            - end index (int): The ending index of the parameters to include in the custom Jacobian view.
            - doc (Optional[str]): A documentation string for the custom Jacobian view.
        """
        return {}

    def _get_transform_aliases(self) -> List[str]:
        r"""
        Return a list of aliases for the transformed points.

        Returns
        -------
        List[str]
            A list of aliases for the transformed points.
        """
        return []

    def _get_inverse_transform_aliases(self) -> List[str]:
        r"""
        Return a list of aliases for the inverse transformed points.

        Returns
        -------
        List[str]
            A list of aliases for the inverse transformed points.
        """
        return []

    def is_set(self) -> bool:
        r"""
        Method to check if the transformation parameters and constants are set.

        .. note::

            By default return :obj:`True`. Sub-classes can override this method to implement specific checks for the transformation parameters and constants.

        Returns
        -------
        bool
            :obj:`True` if the transformation parameters and constants are set, otherwise :obj:`False`.
        """
        return True

    def __repr__(self) -> str:
        r"""
        String representation of the Transform class.

        .. code-block:: console

            {class name} with {``n_params``} parameters and {``n_constants``} constants.
            Parameters: {parameters}
            Constants: {constants}

        Returns
        -------
        str
            A string representation of the transformation.
        """
        return f"{self.__class__.__name__} with {self.n_params} parameters and {self.n_constants} constants.\nParameters: {self.parameters}\nConstants: {self.constants}"

    def _return_transform_result(
        self, transform_result: TransformResult
    ) -> TransformResult:
        r"""
        Return the result of the transformation as a ``TransformResult`` object.

        This method is used to return the result of the transformation, including the transformed points and the Jacobian matrices if requested.

        This method also adds the custom Jacobian views to the `TransformResult` object using the `add_jacobian` method and the custom aliases using the `add_alias` method.

        Parameters
        ----------
        transform_result : :class:`TransformResult`
            The result of the transformation containing the transformed points and the Jacobian matrices.

        Returns
        -------
        :class:`TransformResult`
            The result of the transformation.
        """
        if not isinstance(transform_result, TransformResult):
            raise TypeError(
                f"transform_result must be an instance of TransformResult, got {type(transform_result)}"
            )

        # Add custom Jacobian views to the TransformResult object
        for name, (start, end, doc) in self._get_jacobian_shorthands().items():
            transform_result.add_jacobian(name, start, end, doc=doc)

        # Add custom aliases to the TransformResult object
        for alias in self._get_transform_aliases():
            transform_result.add_alias(alias)

        return transform_result

    def _return_inverse_transform_result(
        self, transform_result: TransformResult
    ) -> TransformResult:
        r"""
        Return the result of the inverse transformation as a ``TransformResult`` object.

        This method is used to return the result of the inverse transformation, including the transformed points and the Jacobian matrices if requested.

        This method also adds the custom Jacobian views to the `TransformResult` object using the `add_jacobian` method and the custom aliases using the `add_alias` method.

        Parameters
        ----------
        transform_result : :class:`TransformResult`
            The result of the inverse transformation containing the transformed points and the Jacobian matrices.

        Returns
        -------
        :class:`TransformResult`
            The result of the inverse transformation.
        """
        if not isinstance(transform_result, TransformResult):
            raise TypeError(
                f"transform_result must be an instance of TransformResult, got {type(transform_result)}"
            )

        # Add custom Jacobian views to the TransformResult object
        for name, (start, end, doc) in self._get_jacobian_shorthands().items():
            transform_result.add_jacobian(name, start, end, doc=doc)

        # Add custom aliases to the TransformResult object
        for alias in self._get_inverse_transform_aliases():
            transform_result.add_alias(alias)

        return transform_result

    # =============================================
    # To be implemented by subclasses
    # =============================================
    @abstractmethod
    def _transform(
        self, points: ArrayLike, *, dx: bool = False, dp: bool = False, **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the transformation to the given points.

        This method must be implemented by subclasses to apply the transformation to the input points.

        Parameters
        ----------
        points : ArrayLike
            The input points to be transformed. Shape (n_points, input_dim).

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:

            - :obj:`transformed_points`: The transformed points of shape (n_points, output_dim).
            - :obj:`jacobian_dx`: The Jacobian matrix with respect to the input points of shape (n_points, output_dim, input_dim) if :obj:`dx` is True, otherwise None.
            - :obj:`jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (n_points, output_dim, n_params) if :obj:`dp` is True, otherwise None.
        """
        raise NotImplementedError("Subclasses must implement the _transform method.")

    @abstractmethod
    def _inverse_transform(
        self, points: ArrayLike, *, dx: bool = False, dp: bool = False, **kwargs
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the inverse transformation to the given points.

        This method must be implemented by subclasses to apply the inverse transformation to the input points.

        Parameters
        ----------
        points : ArrayLike
            The input points to be transformed. Shape (n_points, output_dim).

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:

            - :obj:`transformed_points`: The transformed points of shape (n_points, input_dim).
            - :obj:`jacobian_dx`: The Jacobian matrix with respect to the input points of shape (n_points, input_dim, output_dim) if :obj:`dx` is True, otherwise None.
            - :obj:`jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (n_points, input_dim, n_params) if :obj:`dp` is True, otherwise None.
        """
        raise NotImplementedError(
            "Subclasses must implement the _inverse_transform method."
        )

    # =============================================
    # Transformation Methods
    # =============================================
    def transform(
        self,
        points: ArrayLike,
        *,
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> numpy.ndarray:
        r"""
        Transform the given points using the transformation from :math:`\mathbb{R}^{\text{input_dim}}` to :math:`\mathbb{R}^{\text{output_dim}}`.

        The given points :obj:`points` are assumed to be with shape (..., input_dim) or (input_dim, ...), depending on the value of :obj:`transpose`.

        The output :obj:`transformed_points` will have shape (..., output_dim) if :obj:`transpose` is :obj:`False`, or (output_dim, ...) if :obj:`transpose` is :obj:`True`.

        .. note::

            The :obj:`points`  is converted to a numpy array of ``dtype=numpy.float64``.

        The method also computes 2 Jacobian matrices if requested:

        - :obj:`dx`: Jacobian of the transformed points with respect to the input points.
        - :obj:`dp`: Jacobian of the transformed points with respect to the parameters of the transformation.

        The jacobian matrice with respect to the input points is a (..., output_dim, input_dim) matrix where:

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂X_o/∂X_i -> Jacobian of the coordinates X_o with respect to the coordinates X_i.
            jacobian_dx[..., 0, 1]  # ∂X_o/∂Y_i
            ...

            jacobian_dx[..., 1, 0]  # ∂Y_o/∂X_i -> Jacobian of the coordinates Y_o with respect to the coordinates X_i.
            jacobian_dx[..., 1, 1]  # ∂Y_o/∂Y_i
            ...

        The Jacobian matrice with respect to the parameters is a (..., output_dim, n_params) matrix where:

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂X_o/∂λ_1 -> Jacobian of the coordinates X_o with respect to the first parameter λ_1.
            jacobian_dp[..., 0, 1]  # ∂X_o/∂λ_2
            ...

            jacobian_dp[..., 1, 0]  # ∂Y_o/∂λ_1 -> Jacobian of the coordinates Y_o with respect to the first parameter λ_1.
            jacobian_dp[..., 1, 1]  # ∂Y_o/∂λ_2
            ...

        The Jacobian matrices are computed only if :obj:`dx` or :obj:`dp` are set to True, respectively.

        The output will be a :class:`TransformResult` object containing the transformed points and the Jacobian matrices if requested.

        Parameters
        ----------
        points : ArrayLike
            The input points to be transformed. Shape (..., input_dim) (or (input_dim, ...) if :obj:`transpose` is :obj:`True`).

        transpose : bool, optional
            If True, the input points are transposed to shape (input_dim, ...). Default is False.

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        :class:`TransformResult`
            An object containing the transformed points and the Jacobian matrices if requested.


        Developer Notes
        ----------------
        The subclasses must implement the ``_transform`` method to apply the transformation to the input points.

        The ``_transform`` method should:

        - take the input points as a numpy array of shape (n_points, input_dim)
        - return 3 numpy arrays:

            - ``transformed_points``: The transformed points of shape (n_points, output_dim).
            - ``jacobian_dx``: The Jacobian matrix with respect to the input points of shape (n_points, output_dim, input_dim) if ``dx`` is True, otherwise None.
            - ``jacobian_dp``: The Jacobian matrix with respect to the parameters of the transformation of shape (n_points, output_dim, n_params) if ``dp`` is True, otherwise None.

        """
        # Check the boolean flags
        if not isinstance(dx, bool):
            raise TypeError(f"dx must be a boolean, got {type(dx)}")
        if not isinstance(dp, bool):
            raise TypeError(f"dp must be a boolean, got {type(dp)}")
        if not isinstance(transpose, bool):
            raise TypeError(f"transpose must be a boolean, got {type(transpose)}")

        # Check if the transformation is set
        if not self.is_set():
            raise ValueError(
                "Transformation parameters are not set. Please set the parameters before transforming points."
            )

        # Convert input points to float
        points = numpy.asarray(points, dtype=numpy.float64)

        # Check the shape of the input points
        if points.ndim < 2:
            raise ValueError(
                f"Input points must have at least 2 dimensions, got {points.ndim} dimensions."
            )

        # Transpose the input points if requested
        if transpose:
            points = numpy.moveaxis(
                points, 0, -1
            )  # (input_dim, ...) -> (..., input_dim)

        # Save the shape of the input points
        shape = points.shape  # (..., input_dim)

        # Check the last dimension of the input points
        if shape[-1] != self.input_dim:
            raise ValueError(
                f"Input points must have {self.input_dim} dimensions, got {shape[-1]} dimensions."
            )

        # Flatten the input points to 2D for processing
        points = points.reshape(
            -1, self.input_dim
        )  # (..., input_dim) -> (Npoints, input_dim)

        # Apply the transformation
        transformed_points, jacobian_dx, jacobian_dp = self._transform(
            points, dx=dx, dp=dp, **kwargs
        )  # (Npoints, output_dim), (Npoints, output_dim, input_dim), (Npoints, output_dim, Nparams)

        # Reshape the transformed points to the original shape
        transformed_points = transformed_points.reshape(
            *shape[:-1], self.output_dim
        )  # (Npoints, output_dim) -> (..., output_dim)
        jacobian_dx = (
            jacobian_dx.reshape(*shape[:-1], self.output_dim, self.input_dim)
            if jacobian_dx is not None
            else None
        )  # (Npoints, output_dim, input_dim) -> (..., output_dim, input_dim)
        jacobian_dp = (
            jacobian_dp.reshape(*shape[:-1], self.output_dim, self.n_params)
            if jacobian_dp is not None
            else None
        )  # (Npoints, output_dim, n_params) -> (..., output_dim, n_params)

        # Transpose the transformed points if requested
        if transpose:
            transformed_points = numpy.moveaxis(
                transformed_points, -1, 0
            )  # (..., output_dim) -> (output_dim, ...)
            jacobian_dx = (
                numpy.moveaxis(jacobian_dx, -2, 0) if jacobian_dx is not None else None
            )  # (..., output_dim, input_dim) -> (output_dim, ..., input_dim)
            jacobian_dp = (
                numpy.moveaxis(jacobian_dp, -2, 0) if jacobian_dp is not None else None
            )  # (..., output_dim, n_params) -> (output_dim, ..., n_params)

        # Return the result as a TransformResult object
        return self._return_transform_result(
            self.result_class(
                transformed_points=transformed_points,
                jacobian_dx=jacobian_dx,
                jacobian_dp=jacobian_dp,
                transpose=transpose,
            )
        )

    def inverse_transform(
        self,
        points: ArrayLike,
        *,
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        **kwargs,
    ) -> numpy.ndarray:
        r"""
        Apply the inverse transformation to the given points using the transformation from :math:`\mathbb{R}^{\text{output_dim}}` to :math:`\mathbb{R}^{\text{input_dim}}`.

        The given points :obj:`points` are assumed to be with shape (..., output_dim) or (output_dim, ...), depending on the value of :obj:`transpose`.

        The output :obj:`transformed_points` will have shape (..., input_dim) if :obj:`transpose` is :obj:`False`, or (input_dim, ...) if :obj:`transpose` is :obj:`True`.

        .. note::

            The :obj:`points`  is converted to a numpy array of ``dtype=numpy.float64``.

        The method also computes 2 Jacobian matrices if requested:

        - :obj:`dx`: Jacobian of the transformed points with respect to the input points.
        - :obj:`dp`: Jacobian of the transformed points with respect to the parameters of the transformation.

        The jacobian matrice with respect to the input points is a (..., input_dim, output_dim) matrix where:

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂X_i/∂X_o -> Jacobian of the coordinates X_i with respect to the coordinates X_o.
            jacobian_dx[..., 0, 1]  # ∂X_i/∂Y_o
            ...

            jacobian_dx[..., 1, 0]  # ∂Y_i/∂X_o -> Jacobian of the coordinates Y_i with respect to the coordinates X_o.
            jacobian_dx[..., 1, 1]  # ∂Y_i/∂Y_o
            ...

        The Jacobian matrice with respect to the parameters is a (..., input_dim, n_params) matrix where:

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂X_i/∂λ_1 -> Jacobian of the coordinates X_i with respect to the first parameter λ_1.
            jacobian_dp[..., 0, 1]  # ∂X_i/∂λ_2
            ...

            jacobian_dp[..., 1, 0]  # ∂Y_i/∂λ_1 -> Jacobian of the coordinates Y_i with respect to the first parameter λ_1.
            jacobian_dp[..., 1, 1]  # ∂Y_i/∂λ_2
            ...

        The Jacobian matrices are computed only if :obj:`dx` or :obj:`dp` are set to :obj:`True`, respectively.

        The output will be a :class:`TransformResult` object containing the transformed points and the Jacobian matrices if requested.

        Parameters
        ----------
        points : ArrayLike
            The input points to be transformed. Shape (..., output_dim) (or (output_dim, ...) if :obj:`transpose` is :obj:`True`).

        transpose : bool, optional
            If True, the input points are transposed to shape (output_dim, ...). Default is False.

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        :class:`TransformResult`
            An object containing the transformed points and the Jacobian matrices if requested.


        Developer Notes
        ----------------
        The subclasses must implement the `_inverse_transform` method to apply the inverse transformation to the input points.

        The ``_inverse_transform`` method should:

        - take the input points as a numpy array of shape (n_points, output_dim)
        - return 3 numpy arrays:

            - ``transformed_points``: The transformed points of shape (n_points, input_dim).
            - ``jacobian_dx``: The Jacobian matrix with respect to the input points of shape (n_points, input_dim, output_dim) if ``dx`` is :obj:`True`, otherwise None.
            - ``jacobian_dp``: The Jacobian matrix with respect to the parameters of the transformation of shape (n_points, input_dim, n_params) if ``dp`` is :obj:`True`, otherwise None.
        """
        # Check the boolean flags
        if not isinstance(dx, bool):
            raise TypeError(f"dx must be a boolean, got {type(dx)}")
        if not isinstance(dp, bool):
            raise TypeError(f"dp must be a boolean, got {type(dp)}")
        if not isinstance(transpose, bool):
            raise TypeError(f"transpose must be a boolean, got {type(transpose)}")

        # Check if the transformation is set
        if not self.is_set():
            raise ValueError(
                "Transformation parameters are not set. Please set the parameters before transforming points."
            )

        # Convert input points to float
        points = numpy.asarray(points, dtype=numpy.float64)

        # Check the shape of the input points
        if points.ndim < 2:
            raise ValueError(
                f"Input points must have at least 2 dimensions, got {points.ndim} dimensions."
            )

        # Transpose the input points if requested
        if transpose:
            points = numpy.moveaxis(
                points, 0, -1
            )  # (output_dim, ...) -> (..., output_dim)

        # Save the shape of the input points
        shape = points.shape  # (..., output_dim)

        # Check the last dimension of the input points
        if shape[-1] != self.output_dim:
            raise ValueError(
                f"Input points must have {self.output_dim} dimensions, got {shape[-1]} dimensions."
            )

        # Flatten the input points to 2D for processing
        points = points.reshape(-1, self.output_dim)  # (Npoints, output_dim)

        # Apply the inverse transformation
        transformed_points, jacobian_dx, jacobian_dp = self._inverse_transform(
            points, dx=dx, dp=dp, **kwargs
        )  # (Npoints, input_dim), (Npoints, input_dim, output_dim), (Npoints, input_dim, Nparams)

        # Reshape the transformed points to the original shape
        transformed_points = transformed_points.reshape(
            *shape[:-1], self.input_dim
        )  # (Npoints, input_dim) -> (..., input_dim)
        jacobian_dx = (
            jacobian_dx.reshape(*shape[:-1], self.input_dim, self.output_dim)
            if jacobian_dx is not None
            else None
        )  # (..., input_dim, output_dim)
        jacobian_dp = (
            jacobian_dp.reshape(*shape[:-1], self.input_dim, self.n_params)
            if jacobian_dp is not None
            else None
        )  # (..., input_dim, n_params)

        # Transpose the transformed points if requested
        if transpose:
            transformed_points = numpy.moveaxis(
                transformed_points, -1, 0
            )  # (..., input_dim) -> (input_dim, ...)
            jacobian_dx = (
                numpy.moveaxis(jacobian_dx, -2, 0) if jacobian_dx is not None else None
            )  # (..., input_dim, output_dim) -> (input_dim, ..., output_dim)
            jacobian_dp = (
                numpy.moveaxis(jacobian_dp, -2, 0) if jacobian_dp is not None else None
            )  # (..., input_dim, n_params) -> (input_dim, ..., n_params)

        # Return the result as a InverseTransformResult object
        return self._return_inverse_transform_result(
            self.inverse_result_class(
                transformed_points=transformed_points,
                jacobian_dx=jacobian_dx,
                jacobian_dp=jacobian_dp,
                transpose=transpose,
            )
        )

    # =============================================
    # I/O Methods
    # =============================================
    def to_dict(self) -> Dict[str, Any]:
        r"""
        Serialize the transformation to a dictionary.

        Here is an example of the dictionary structure:

        .. code-block:: python

            from pycvcam import Cv2Distortion
            from pycvcam import write_transform

            distortion = Cv2Distortion(...)
            distortion.parameters = numpy.array([0.1, 0.2, 0.3, 0.01, 0.5])
            disto_dict = distortion.to_dict()

        The content of the dict will be similar to:

        .. code-block:: json

            {
                "type": "Cv2Distortion",
                "version": "1.3.0",
                "date": "2023-01-01T00:00:00",
                "parameters": [0.1, 0.2, 0.3, 0.01, 0.5],
                "constants": null
            }

        .. seealso::

            - :meth:`from_dict` : Method to create a Transform object from a dictionary.
            - :meth:`to_json` : Method to write the Transform object to a JSON file.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the transformation.

        """
        # Create a dict containing the Transform object's data
        transform_data = {}
        transform_data["type"] = type(self).__name__
        transform_data["version"] = __version__
        transform_data["date"] = datetime.datetime.now().isoformat()
        transform_data["parameters"] = (
            list(self.parameters) if self.parameters is not None else None
        )
        transform_data["constants"] = (
            list(self.constants) if self.constants is not None else None
        )

        return transform_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Transform:
        r"""
        Create a Transform object from a dictionary.

        The input dictionary should have the following structure:

        .. code-block:: json

            {
                "type": "Cv2Distortion",
                "version": "1.3.0",
                "date": "2023-01-01T00:00:00",
                "parameters": [0.1, 0.2, 0.3, 0.01, 0.5],
                "constants": null
            }

        .. seealso::

            - :meth:`to_dict` : Method to serialize the Transform object to a dictionary.
            - :meth:`from_json` : Method to read a Transform object from a JSON file.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary representation of the transformation.

        Returns
        -------
        :class:`Transform`
            A Transform object created from the input dictionary.

        Raises
        -------
        ValueError
            If the input dictionary does not contain the required keys or has invalid values.
        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dictionary, got {type(data)}")

        if not "parameters" in data:
            raise ValueError("Missing 'parameters' key in transform data.")

        if not "constants" in data:
            raise ValueError("Missing 'constants' key in transform data.")

        if not "type" in data:
            print(
                "[pycvcam] Missing 'type' key in transform data. Loading without type verification."
            )

        if "type" in data and not data["type"] == cls.__name__:
            raise ValueError(
                f"Transform type mismatch, expected {cls.__name__} but got {data['type']}"
            )

        # Create an instance of the Transform subclass
        transform = cls()
        transform.parameters = data.get("parameters", None)
        transform.constants = data.get("constants", None)

        return transform

    def to_json(self, filepath: str) -> None:
        r"""
        Write the transformation to a JSON file.

        .. code-block:: python

            from pycvcam import Cv2Distortion
            from pycvcam import write_transform

            distortion = Cv2Distortion(...)
            distortion.parameters = numpy.array([0.1, 0.2, 0.3, 0.01, 0.5])
            distortion.to_json("distortion.json")

        The content of the JSON file will be similar to:

        .. code-block:: json

            {
                "type": "Cv2Distortion",
                "version": "1.3.0",
                "date": "2023-01-01T00:00:00",
                "parameters": [0.1, 0.2, 0.3, 0.01, 0.5],
                "constants": null
            }

        .. seealso::

            - :meth:`from_dict` : Method to create a Transform object from a dictionary.
            - :meth:`from_json` : Method to read a Transform object from a JSON file.

        Parameters
        ----------
        filepath : str
            The path to the JSON file where the transformation will be saved.

        """
        transform_data = self.to_dict()

        # Write the transform data to a JSON file
        with open(filepath, "w") as json_file:
            json.dump(transform_data, json_file, indent=4)

    @classmethod
    def from_json(cls, filepath: str) -> Transform:
        r"""
        Read a Transform object from a JSON file.

        The input JSON file should have the following structure:

        .. code-block:: json

            {
                "type": "Cv2Distortion",
                "version": "1.3.0",
                "date": "2023-01-01T00:00:00",
                "parameters": [0.1, 0.2, 0.3, 0.01, 0.5],
                "constants": null
            }

        .. seealso::

            - :meth:`to_dict` : Method to serialize the Transform object to a dictionary.
            - :meth:`to_json` : Method to write the Transform object to a JSON file.

        Parameters
        ----------
        filepath : str
            The path to the JSON file containing the transformation data.

        Returns
        -------
        :class:`Transform`
            A Transform object created from the input JSON file.

        Raises
        -------
        ValueError
            If the JSON file does not contain the required keys or has invalid values.
        """
        # Read the transform data from the JSON file
        with open(filepath, "r") as json_file:
            transform_data = json.load(json_file)

        return cls.from_dict(transform_data)
