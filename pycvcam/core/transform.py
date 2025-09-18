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

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, ClassVar
import numpy

from .transform_result import TransformResult
from .package import Package


class Transform(ABC):
    r"""
    Transform is the base class to manage transformations from :math:`\mathbb{R}^{input\_dim}` to :math:`\mathbb{R}^{output\_dim}`.

    A tranformation is a function that maps points from an input space to an output space.
    The transformation is defined by:
    
    - a set of ``Nparams`` parameters :math:`\{\lambda_1, \lambda_2, \ldots, \lambda_N\}` that define the transformation.
    - a set of ``Nconstants`` constants that are constant for the transformation. (Can not be optimized and no jacobian with respect to these constants).

    .. math::

        \vec{X}_O = T(\vec{X}_I, \lambda_1, \lambda_2, \ldots, \lambda_N)

    where :math:`\vec{X}_O` are the output points, :math:`\vec{X}_I` are the input points, and :math:`\{\lambda_1, \lambda_2, \ldots, \lambda_N\}` are the parameters of the transformation.

    This class provides the base for all transformations. It defines the interface for extrinsic, distortion, and intrinsic transformations.

    .. seealso::

        - :class:`pycvcam.core.Extrinsic` for extrinsic transformations.
        - :class:`pycvcam.core.Distortion` for distortion transformations.
        - :class:`pycvcam.core.Intrinsic` for intrinsic transformations.

    Each sub-classes must implement the following methods and properties:

    - ``_input_dim``: (class attribute) The dimension of the input points as integer (example: 2 for 2D points).
    - ``_output_dim``: (class attribute) The dimension of the output points as integer (example: 2 for 2D points).
    - ``_transform``: (method) Apply the transformation to the given points with shape (Npoints, input_dim) and return the transformed points with shape (Npoints, output_dim), and optionally the Jacobian matrices if requested.
    - ``_inverse_transform``: (method) Apply the inverse transformation to the given points with shape (Npoints, output_dim) and return the transformed points with shape (Npoints, input_dim), and optionally the Jacobian matrices if requested.

    The following properties are not required but can be overwritting to provide additional information about the transformation:

    - ``parameters`` (property and setter) The parameters of the transformation in a 1D numpy array of shape (Nparams,) or None if the transformation does not have parameters or they are not set. Default only impose 1D array of floats or None.
    - ``constants`` (property and setter) The constants of the transformation in a 1D numpy array of shape (Nconsts,) or None if the transformation does not have constants or they are not set. Default only impose 1D array of floats or None.
    - ``parameter_names`` (property) The names of the parameters as a list of strings or None if the transformation does not have parameters or they are not set. Default is None.
    - ``constant_names`` (property) The names of the constants as a list of strings or None if the transformation does not have constants or they are not set. Default is None.
    - ``is_set``: (method) Check if the transformation is set (i.e., if the parameters are initialized). Default is to return True if the parameters and constants are not None. Default is to return True if the parameters and constants are not None.
    - ``_result_class``: (class attribute) The class used for the result of the transformation (sub-class of ``TransformResult``). Default is :class:`pycvcam.core.TransformResult`.
    - ``_inverse_result_class``: (class attribute) The class used for the result of the inverse transformation (sub-class of ``TransformResult``). Default is :class:`pycvcam.core.TransformResult`.
    - ``_get_jacobian_short_hand``: (method) A dictionary of short-hand notation for the Jacobian matrices, which can be used to add custom views of the ``jacobian_dp`` matrix with respect to the parameters of the transformation. Default is an empty dictionary.
    - ``_get_transform_aliases``: (method) A dictionary of aliases for the transformation, which can be used to add custom names for the transformation parameters. Default is an empty list.
    - ``_get_inverse_transform_aliases``: (method) A dictionary of aliases for the inverse transformation, which can be used to add custom names for the inverse transformation parameters. Default is an empty list.

    More details on the transformation methods are provided in the `transform` and `inverse_transform` methods. 

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
    def __init__(self, parameters: Optional[numpy.ndarray] = None, constants: Optional[numpy.ndarray] = None):
        self.parameters = parameters
        self.constants = constants


    # =============================================
    # Properties for Transform Class
    # =============================================
    @property
    def result_class(self) -> type:
        r"""
        Property to return the class used for the result of the transformation.
        
        Returns
        -------
        type
            The class used for the result of the transformation.
        """
        if not issubclass(self._result_class, TransformResult):
            raise TypeError(f"result_class must be a subclass of TransformResult, got {self._result_class}")
        return self._result_class

    @property
    def inverse_result_class(self) -> type:
        r"""
        Property to return the class used for the result of the inverse transformation.
                
        Returns
        -------
        type
            The class used for the result of the inverse transformation.
        """
        if not issubclass(self._inverse_result_class, TransformResult):
            raise TypeError(f"inverse_result_class must be a subclass of TransformResult, got {self._inverse_result_class}")
        return self._inverse_result_class
    
    @property
    def input_dim(self) -> int:
        r"""
        Property to return the input dimension of the transformation.
        
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
            raise NotImplementedError("Subclasses must define the ``_input_dim`` class attribute.")
        return self._input_dim

    @property
    def output_dim(self) -> int:
        r"""
        Property to return the output dimension of the transformation.

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
            raise NotImplementedError("Subclasses must define the ``_output_dim`` class attribute.")
        return self._output_dim

    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Property to return the parameters of the transformation.
        
        The parameters must be a 1-D numpy array of shape (Nparams,) where Nparams is the number of parameters of the transformation.

        If the transformation does not have parameters or they are not set, this property should return None.
        
        Returns
        -------
        Optional[numpy.ndarray]
            The parameters of the transformation.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, value: Optional[numpy.ndarray]) -> None:
        r"""
        Setter for the parameters of the transformation.
        
        The parameters must be a 1-D float numpy array of shape (Nparams,) where Nparams is the number of parameters of the transformation.

        If the transformation does not have parameters or they are not set, this setter should set the parameters to None.
        
        Parameters
        ----------
        value : Optional[numpy.ndarray]
            The parameters of the transformation.
        """
        parameters = numpy.asarray(value, dtype=Package.get_float_dtype()) if value is not None else None
        if parameters is not None and parameters.ndim != 1:
            raise ValueError(f"Parameters must be a 1-D numpy array, got shape {parameters.shape}")
        self._parameters = parameters

    @property
    def constants(self) -> Optional[numpy.ndarray]:
        r"""
        Property to return the constants of the transformation.

        The constants must be a 1-D float numpy array of shape (Nconstants,) where Nconstants is the number of constants of the transformation.

        If the transformation does not have constants or they are not set, this property should return None.

        Returns
        -------
        Optional[numpy.ndarray]
            The constants of the transformation.
        """
        return self._constants

    @constants.setter
    def constants(self, value: Optional[numpy.ndarray]) -> None:
        r"""
        Setter for the constants of the transformation.

        The constants must be a 1-D float numpy array of shape (Nconstants,) where Nconstants is the number of constants of the transformation.

        If the transformation does not have constants or they are not set, this setter should set the constants to None.

        Parameters
        ----------
        value : Optional[numpy.ndarray]
            The constants of the transformation.
        """
        constants = numpy.asarray(value, dtype=Package.get_float_dtype()) if value is not None else None
        if constants is not None and constants.ndim != 1:
            raise ValueError(f"Constants must be a 1-D numpy array, got shape {constants.shape}")
        self._constants = constants

    @property
    def Nparams(self) -> int:
        r"""
        Property to return the number of parameters of the transformation.
        
        The number of parameters must be a non-negative integer representing the number of parameters of the transformation.
        
        Returns
        -------
        int
            The number of parameters of the transformation.
        """
        return self.parameters.size if self.parameters is not None else 0
    
    @property
    def Nconstants(self) -> int:
        r"""
        Property to return the number of coefficients of the transformation.
        
        The number of coefficients must be a non-negative integer representing the number of coefficients of the transformation.
        
        Returns
        -------
        int
            The number of constants of the transformation.
        """
        return self.constants.size if self.constants is not None else 0
    
    @property
    def parameter_names(self) -> List[str]:
        r"""
        Property to return the names of the parameters of the transformation.

        The names must be a list of strings of length Nparams where Nparams is the number of parameters of the transformation.

        If the transformation does not have parameters should return an empty list.

        By default, the parameter names are generated as "p_0", "p_1", ..., "p_{Nparams-1}".

        Returns
        -------
        List[str]
            The names of the parameters of the transformation.
        """
        return [f"p_{i}" for i in range(self.Nparams)]

    @property
    def constant_names(self) -> List[str]:
        r"""
        Property to return the names of the constants of the transformation.

        The names must be a list of strings of length Nconstants where Nconstants is the number of constants of the transformation.

        If the transformation does not have constants, this property should return an empty list.

        By default, the constant names are generated as "c_0", "c_1", ..., "c_{Nconstants-1}".

        Returns
        -------
        List[str]
            The names of the constants of the transformation.
        """
        return [f"c_{i}" for i in range(self.Nconstants)]

    # =============================================
    # Methods for Transform Class
    # =============================================
    def _get_jacobian_short_hand(self) -> Dict[str, Tuple[int, int, Optional[str]]]:
        r"""
        Property to return a dictionary of short-hand notation for the Jacobian matrices.
        
        This dictionary can be used to add custom views of the `jacobian_dp` matrix with respect to the parameters of the transformation.

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
        Property to return a list of aliases for the transformed points.
        
        Returns
        -------
        List[str]
            A list of aliases for the transformed points.
        """
        return []
    
    def _get_inverse_transform_aliases(self) -> List[str]:
        r"""
        Property to return a list of aliases for the inverse transformed points.

        Returns
        -------
        List[str]
            A list of aliases for the inverse transformed points.
        """
        return []
    
    def is_set(self) -> bool:
        r"""
        Method to check if the transformation parameters and constants are set.

        This method returns True if the parameters and constants are not None, otherwise False.

        Returns
        -------
        bool
            True if the transformation parameters and constants are set, otherwise False.
        """
        return self.parameters is not None and self.constants is not None

    def __repr__(self) -> str:
        r"""
        String representation of the Transform class.

        .. code-block:: console

            {class name} with {Nparams} parameters and {Nconstants} constants.
            Parameters: {parameters}
            Constants: {constants}

        Returns
        -------
        str
            A string representation of the transformation.
        """
        return f"{self.__class__.__name__} with {self.Nparams} parameters and {self.Nconstants} constants.\nParameters: {self.parameters}\nConstants: {self.constants}"

    def _return_transform_result(self, transform_result: TransformResult) -> TransformResult:
        r"""
        Return the result of the transformation as a ``TransformResult`` object.

        This method is used to return the result of the transformation, including the transformed points and the Jacobian matrices if requested.

        This method also adds the custom Jacobian views to the `TransformResult` object using the `add_jacobian` method and the custom aliases using the `add_alias` method.

        Parameters
        ----------
        transform_result : TransformResult
            The result of the transformation containing the transformed points and the Jacobian matrices.

        Returns
        -------
        TransformResult
            The result of the transformation.
        """
        if not isinstance(transform_result, TransformResult):
            raise TypeError(f"transform_result must be an instance of TransformResult, got {type(transform_result)}")
        
        # Add custom Jacobian views to the TransformResult object
        for name, (start, end, doc) in self._get_jacobian_short_hand().items():
            transform_result.add_jacobian(name, start, end, doc=doc)

        # Add custom aliases to the TransformResult object
        for alias in self._get_transform_aliases():
            transform_result.add_alias(alias)
        
        return transform_result

    def _return_inverse_transform_result(self, transform_result: TransformResult) -> TransformResult:
        r"""
        Return the result of the inverse transformation as a ``TransformResult`` object.

        This method is used to return the result of the inverse transformation, including the transformed points and the Jacobian matrices if requested.

        This method also adds the custom Jacobian views to the `TransformResult` object using the `add_jacobian` method and the custom aliases using the `add_alias` method.

        Parameters
        ----------
        transform_result : TransformResult
            The result of the inverse transformation containing the transformed points and the Jacobian matrices.

        Returns
        -------
        TransformResult
            The result of the inverse transformation.
        """
        if not isinstance(transform_result, TransformResult):
            raise TypeError(f"transform_result must be an instance of TransformResult, got {type(transform_result)}")
        
        # Add custom Jacobian views to the TransformResult object
        for name, (start, end, doc) in self._get_jacobian_short_hand().items():
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
        self,
        points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs
        ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the transformation to the given points.

        This method must be implemented by subclasses to apply the transformation to the input points.

        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (Npoints, input_dim).

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

            - `transformed_points`: The transformed points of shape (Npoints, output_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, output_dim, input_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, output_dim, Nparams) if `dp` is True, otherwise None.
        """
        raise NotImplementedError("Subclasses must implement the _transform method.")
    
    @abstractmethod
    def _inverse_transform(
        self,
        points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        **kwargs
        ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the inverse transformation to the given points.

        This method must be implemented by subclasses to apply the inverse transformation to the input points.

        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (Npoints, output_dim).

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

            - `transformed_points`: The transformed points of shape (Npoints, input_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, input_dim, output_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, input_dim, Nparams) if `dp` is True, otherwise None.
        """
        raise NotImplementedError("Subclasses must implement the _inverse_transform method.")

    
    # =============================================
    # Transformation Methods
    # =============================================
    def transform(
        self,
        points: numpy.ndarray,
        *,
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        _skip: bool = False,
        **kwargs
        ) -> numpy.ndarray:
        r"""
        The given points ``points`` are assumed to be with shape (..., input_dim) or (input_dim, ...), depending on the value of ``transpose``.

        The output ``transformed_points`` will have shape (..., output_dim) if ``transpose`` is False, or (output_dim, ...) if ``transpose`` is True.

        .. warning::

            The points are converting to float before applying the transformation.
            See :class:`pycvcam.core.Package` for more details on the default data types used in the package.

        The method also computes 2 Jacobian matrices if requested:

        - ``dx``: Jacobian of the transformed points with respect to the input points.
        - ``dp``: Jacobian of the transformed points with respect to the parameters of the transformation.

        The jacobian matrice with respect to the input points is a (..., output_dim, input_dim) matrix where:

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂X_o/∂X_i -> Jacobian of the coordinates X_o with respect to the coordinates X_i.
            jacobian_dx[..., 0, 1]  # ∂X_o/∂Y_i
            ...

            jacobian_dx[..., 1, 0]  # ∂Y_o/∂X_i -> Jacobian of the coordinates Y_o with respect to the coordinates X_i.
            jacobian_dx[..., 1, 1]  # ∂Y_o/∂Y_i
            ...

        The Jacobian matrice with respect to the parameters is a (..., output_dim, Nparams) matrix where:

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂X_o/∂λ_1 -> Jacobian of the coordinates X_o with respect to the first parameter λ_1.
            jacobian_dp[..., 0, 1]  # ∂X_o/∂λ_2
            ...

            jacobian_dp[..., 1, 0]  # ∂Y_o/∂λ_1 -> Jacobian of the coordinates Y_o with respect to the first parameter λ_1.
            jacobian_dp[..., 1, 1]  # ∂Y_o/∂λ_2
            ...

        The Jacobian matrices are computed only if ``dx`` or ``dp`` are set to True, respectively.

        The output will be a `TransformResult` object containing the transformed points and the Jacobian matrices if requested.

        .. note::

            The _skip parameter is used to skip the checks for the transformation parameters and assume the points are given in the (Npoints, input_dim) float format.
            Please use this parameter with caution, as it may lead to unexpected results if the transformation parameters are not set correctly.
        
        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (..., input_dim) (or (input_dim, ...) if `transpose` is True).

        transpose : bool, optional
            If True, the input points are transposed to shape (input_dim, ...). Default is False.

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        _skip : bool, optional
            [INTERNAL USE], If True, skip the checks for the transformation parameters and assume the points are given in the (Npoints, input_dim) float format.
            `transpose` is ignored if this parameter is set to True.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        TransformResult
            An object containing the transformed points and the Jacobian matrices if requested.

            
        Developer Notes
        ----------------
        The subclasses must implement the `_transform` method to apply the transformation to the input points.
        
        The `_transform` method should:

        - take the input points as a numpy array of shape (Npoints, input_dim)
        - return 3 numpy arrays:

            - `transformed_points`: The transformed points of shape (Npoints, output_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, output_dim, input_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, output_dim, Nparams) if `dp` is True, otherwise None.

        """
        if not _skip:
            # Check the boolean flags
            if not isinstance(dx, bool):
                raise TypeError(f"dx must be a boolean, got {type(dx)}")
            if not isinstance(dp, bool):
                raise TypeError(f"dp must be a boolean, got {type(dp)}")
            if not isinstance(transpose, bool):
                raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
            
            # Check if the transformation is set
            if not self.is_set():
                raise ValueError("Transformation parameters are not set. Please set the parameters before transforming points.")
        
            # Convert input points to float
            points = numpy.asarray(points, dtype=Package.get_float_dtype())

            # Check the shape of the input points
            if points.ndim < 2:
                raise ValueError(f"Input points must have at least 2 dimensions, got {points.ndim} dimensions.")

            # Transpose the input points if requested
            if transpose:
                points = numpy.moveaxis(points, 0, -1)  # (input_dim, ...) -> (..., input_dim)

            # Save the shape of the input points
            shape = points.shape # (..., input_dim)

            # Check the last dimension of the input points
            if shape[-1] != self.input_dim:
                raise ValueError(f"Input points must have {self.input_dim} dimensions, got {shape[-1]} dimensions.")

            # Flatten the input points to 2D for processing
            points = points.reshape(-1, self.input_dim) # (..., input_dim) -> (Npoints, input_dim)

        # Apply the transformation
        transformed_points, jacobian_dx, jacobian_dp = self._transform(points, dx=dx, dp=dp, **kwargs) # (Npoints, output_dim), (Npoints, output_dim, input_dim), (Npoints, output_dim, Nparams)

        if not _skip:
            # Reshape the transformed points to the original shape
            transformed_points = transformed_points.reshape(*shape[:-1], self.output_dim)  # (Npoints, output_dim) -> (..., output_dim)
            jacobian_dx = jacobian_dx.reshape(*shape[:-1], self.output_dim, self.input_dim) if jacobian_dx is not None else None  # (Npoints, output_dim, input_dim) -> (..., output_dim, input_dim)
            jacobian_dp = jacobian_dp.reshape(*shape[:-1], self.output_dim, self.Nparams) if jacobian_dp is not None else None # (Npoints, output_dim, Nparams) -> (..., output_dim, Nparams)

            # Transpose the transformed points if requested
            if transpose:
                transformed_points = numpy.moveaxis(transformed_points, -1, 0) # (..., output_dim) -> (output_dim, ...)
                jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if jacobian_dx is not None else None # (..., output_dim, input_dim) -> (output_dim, ..., input_dim)
                jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if jacobian_dp is not None else None # (..., output_dim, Nparams) -> (output_dim, ..., Nparams)

        # Return the result as a TransformResult object
        return self._return_transform_result(self.result_class(
            transformed_points=transformed_points,
            jacobian_dx=jacobian_dx,
            jacobian_dp=jacobian_dp,
            transpose=transpose,
        ))
    

    def inverse_transform(
        self,
        points: numpy.ndarray,
        *,
        transpose: bool = False,
        dx: bool = False,
        dp: bool = False,
        _skip: bool = False,
        **kwargs
        ) -> numpy.ndarray:
        r"""
        The given points ``points`` are assumed to be with shape (..., output_dim) or (output_dim, ...), depending on the value of ``transpose``.

        The output ``transformed_points`` will have shape (..., input_dim) if ``transpose`` is False, or (input_dim, ...) if ``transpose`` is True.

        .. warning::

            The points are converting to float before applying the inverse transformation.
            See :class:`pycvcam.core.Package` for more details on the default data types used in the package.

        The method also computes 2 Jacobian matrices if requested:

        - ``dx``: Jacobian of the transformed points with respect to the input points.
        - ``dp``: Jacobian of the transformed points with respect to the parameters of the transformation.

        The jacobian matrice with respect to the input points is a (..., input_dim, output_dim) matrix where:

        .. code-block:: python

            jacobian_dx[..., 0, 0]  # ∂X_i/∂X_o -> Jacobian of the coordinates X_i with respect to the coordinates X_o.
            jacobian_dx[..., 0, 1]  # ∂X_i/∂Y_o
            ...

            jacobian_dx[..., 1, 0]  # ∂Y_i/∂X_o -> Jacobian of the coordinates Y_i with respect to the coordinates X_o.
            jacobian_dx[..., 1, 1]  # ∂Y_i/∂Y_o
            ...

        The Jacobian matrice with respect to the parameters is a (..., input_dim, Nparams) matrix where:

        .. code-block:: python

            jacobian_dp[..., 0, 0]  # ∂X_i/∂λ_1 -> Jacobian of the coordinates X_i with respect to the first parameter λ_1.
            jacobian_dp[..., 0, 1]  # ∂X_i/∂λ_2
            ...

            jacobian_dp[..., 1, 0]  # ∂Y_i/∂λ_1 -> Jacobian of the coordinates Y_i with respect to the first parameter λ_1.
            jacobian_dp[..., 1, 1]  # ∂Y_i/∂λ_2
            ...

        The Jacobian matrices are computed only if ``dx`` or ``dp`` are set to True, respectively.

        The output will be a `TransformResult` object containing the transformed points and the Jacobian matrices if requested.

        .. note::

            The _skip parameter is used to skip the checks for the transformation parameters and assume the points are given in the (Npoints, output_dim) float format.
            Please use this parameter with caution, as it may lead to unexpected results if the transformation parameters are not set correctly.

        Parameters
        ----------
        points : numpy.ndarray
            The input points to be transformed. Shape (..., output_dim) (or (output_dim, ...) if `transpose` is True).

        transpose : bool, optional
            If True, the input points are transposed to shape (output_dim, ...). Default is False.

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        _skip : bool, optional
            [INTERNAL USE], If True, skip the checks for the transformation parameters and assume the points are given in the (Npoints, output_dim) float format.
            `transpose` is ignored if this parameter is set to True.

        **kwargs
            Additional keyword arguments for the transformation.

        Returns
        -------
        TransformResult
            An object containing the transformed points and the Jacobian matrices if requested.

            
        Developer Notes
        ----------------
        The subclasses must implement the `_inverse_transform` method to apply the inverse transformation to the input points.

        The `_inverse_transform` method should:

        - take the input points as a numpy array of shape (Npoints, output_dim)
        - return 3 numpy arrays:

            - `transformed_points`: The transformed points of shape (Npoints, input_dim).
            - `jacobian_dx`: The Jacobian matrix with respect to the input points of shape (Npoints, input_dim, output_dim) if `dx` is True, otherwise None.
            - `jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (Npoints, input_dim, Nparams) if `dp` is True, otherwise None.
        """
        if not _skip:
            # Check the boolean flags
            if not isinstance(dx, bool):
                raise TypeError(f"dx must be a boolean, got {type(dx)}")
            if not isinstance(dp, bool):
                raise TypeError(f"dp must be a boolean, got {type(dp)}")
            if not isinstance(transpose, bool):
                raise TypeError(f"transpose must be a boolean, got {type(transpose)}")
            
            # Check if the transformation is set
            if not self.is_set():
                raise ValueError("Transformation parameters are not set. Please set the parameters before transforming points.")
            
            # Convert input points to float
            points = numpy.asarray(points, dtype=Package.get_float_dtype())

            # Check the shape of the input points
            if points.ndim < 2:
                raise ValueError(f"Input points must have at least 2 dimensions, got {points.ndim} dimensions.")
            
            # Transpose the input points if requested
            if transpose:
                points = numpy.moveaxis(points, 0, -1) # (output_dim, ...) -> (..., output_dim)
            
            # Save the shape of the input points
            shape = points.shape # (..., output_dim)

            # Check the last dimension of the input points
            if shape[-1] != self.output_dim:
                raise ValueError(f"Input points must have {self.output_dim} dimensions, got {shape[-1]} dimensions.")
            
            # Flatten the input points to 2D for processing
            points = points.reshape(-1, self.output_dim) # (Npoints, output_dim)

        # Apply the inverse transformation
        transformed_points, jacobian_dx, jacobian_dp = self._inverse_transform(points, dx=dx, dp=dp, **kwargs) # (Npoints, input_dim), (Npoints, input_dim, output_dim), (Npoints, input_dim, Nparams)

        if not _skip:
            # Reshape the transformed points to the original shape
            transformed_points = transformed_points.reshape(*shape[:-1], self.input_dim)  # (Npoints, input_dim) -> (..., input_dim)
            jacobian_dx = jacobian_dx.reshape(*shape[:-1], self.input_dim, self.output_dim) if jacobian_dx is not None else None  # (..., input_dim, output_dim)
            jacobian_dp = jacobian_dp.reshape(*shape[:-1], self.input_dim, self.Nparams) if jacobian_dp is not None else None # (..., input_dim, Nparams)

            # Transpose the transformed points if requested
            if transpose:
                transformed_points = numpy.moveaxis(transformed_points, -1, 0) # (..., input_dim) -> (input_dim, ...)
                jacobian_dx = numpy.moveaxis(jacobian_dx, -2, 0) if jacobian_dx is not None else None # (..., input_dim, output_dim) -> (input_dim, ..., output_dim)
                jacobian_dp = numpy.moveaxis(jacobian_dp, -2, 0) if jacobian_dp is not None else None # (..., input_dim, Nparams) -> (input_dim, ..., Nparams)

        # Return the result as a InverseTransformResult object
        return self._return_inverse_transform_result(self.inverse_result_class(
            transformed_points=transformed_points,
            jacobian_dx=jacobian_dx,
            jacobian_dp=jacobian_dp,
            transpose=transpose
        ))