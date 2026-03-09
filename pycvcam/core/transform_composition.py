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

from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Any
import numpy

from numpy.typing import ArrayLike
from .transform import Transform
from ..__version__ import __version__


class TransformComposition(Transform):
    r"""
    A class to represent the composition of multiple transformations.

    This class allows to compose multiple transformations together, applying them sequentially to the input points.

    The composition of transformations is defined as follows:

    .. math::

        T_{\text{composed}} = T_n \circ T_{n-1} \circ ... \circ T_2 \circ T_1

    where :math:`T_i` are the individual transformations and :math:`T_{\text{composed}}` is the resulting composed transformation.

    The parameters and constants of the composed transformation are the concatenation of the parameters and constants of the individual transformations.

    The Jacobian matrices of the composed transformation can be computed using the chain rule of differentiation, taking into account the Jacobian matrices of the individual transformations.

    Parameters
    ----------
    transforms : List[:class:`Transform`]
        A list of Transform objects to be composed together. The transformations will be applied in the order they are listed.

    """

    def __init__(self, transforms: List[Transform]):
        if not isinstance(transforms, list):
            raise TypeError(f"transforms must be a list, got {type(transforms)}")
        if len(transforms) == 0:
            raise ValueError("transforms list cannot be empty")
        if not all(isinstance(t, Transform) for t in transforms):
            raise TypeError("All elements in transforms must be instances of Transform")

        self.transforms = transforms

    @property
    def input_dim(self) -> int:
        r"""
        The input dimension of the composed transformation, which is the input dimension of the first transformation in the composition.

        Returns
        -------
        int
            The input dimension of the composed transformation.
        """
        return self.transforms[0].input_dim

    @property
    def output_dim(self) -> int:
        r"""
        The output dimension of the composed transformation, which is the output dimension of the last transformation in the composition.

        Returns
        -------
        int
            The output dimension of the composed transformation.
        """
        return self.transforms[-1].output_dim

    @property
    def n_params(self) -> int:
        r"""
        The total number of parameters of the composed transformation, which is the sum of the number of parameters of the individual transformations.

        Returns
        -------
        int
            The total number of parameters of the composed transformation.
        """
        return sum(t.n_params for t in self.transforms)

    @property
    def n_constants(self) -> int:
        r"""
        The total number of constants of the composed transformation, which is the sum of the number of constants of the individual transformations.

        Returns
        -------
        int
            The total number of constants of the composed transformation.
        """
        return sum(t.n_constants for t in self.transforms)

    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        The parameters of the composed transformation, which is the concatenation of the parameters of the individual transformations.

        Parameters
        ----------
        value : Optional[ArrayLike]
            The parameters to set for the composed transformation. This should be a 1D array containing the concatenated parameters of the individual transformations.

        Returns
        -------
        Optional[numpy.ndarray]
            The parameters of the composed transformation as a 1D numpy array, or None if all of the individual transformations has parameters set to None.
        """
        if all(t.parameters is None for t in self.transforms):
            return None
        parameters = []
        for t in self.transforms:
            if t.parameters is not None:
                parameters.append(t.parameters)
        return numpy.concatenate(parameters)

    @parameters.setter
    def parameters(self, value: Optional[ArrayLike]) -> None:
        if value is None:
            for t in self.transforms:
                t.parameters = None
            return
        value = numpy.asarray(value, dtype=numpy.float64)
        if value.ndim != 1:
            raise ValueError(f"parameters must be a 1D array, got {value.ndim}D array")
        if value.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {value.size}")
        index = 0
        for t in self.transforms:
            n_params = t.n_params
            if n_params > 0:
                t.parameters = value[index : index + n_params]
                index += n_params

    @property
    def constants(self) -> Optional[numpy.ndarray]:
        r"""
        The constants of the composed transformation, which is the concatenation of the constants of the individual transformations.

        Parameters
        ----------
        value : Optional[ArrayLike]
            The constants to set for the composed transformation. This should be a 1D array containing the concatenated constants of the individual transformations.

        Returns
        -------
        Optional[numpy.ndarray]
            The constants of the composed transformation as a 1D numpy array, or None if all of the individual transformations has constants set to None.
        """
        if all(t.constants is None for t in self.transforms):
            return None
        constants = []
        for t in self.transforms:
            if t.constants is not None:
                constants.append(t.constants)
        return numpy.concatenate(constants)

    @constants.setter
    def constants(self, value: Optional[ArrayLike]) -> None:
        if value is None:
            for t in self.transforms:
                t.constants = None
            return
        value = numpy.asarray(value, dtype=numpy.float64)
        if value.ndim != 1:
            raise ValueError(f"constants must be a 1D array, got {value.ndim}D array")
        if value.size != self.n_constants:
            raise ValueError(f"Expected {self.n_constants} constants, got {value.size}")
        index = 0
        for t in self.transforms:
            n_constants = t.n_constants
            if n_constants > 0:
                t.constants = value[index : index + n_constants]
                index += n_constants

    def is_set(self) -> bool:
        r"""
        Check if all the parameters of the individual transformations in the composition are set.

        Returns
        -------
        bool
            True if all the parameters of the individual transformations are set, False otherwise.
        """
        return all(t.is_set() for t in self.transforms)

    def _transform(
        self,
        points: ArrayLike,
        *,
        dx: bool = False,
        dp: bool = False,
        list_kwargs: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the composed transformation to the given points.

        This method applies the individual transformations sequentially to the input points, and computes the Jacobian matrices using the chain rule of differentiation.

        Parameters
        ----------
        points : ArrayLike
            The input points to be transformed. Shape (n_points, input_dim).

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        list_kwargs : Optional[List[Dict[str, Any]]], optional
            A list of dictionaries containing additional keyword arguments for each individual transformation.
            Default is None. If provided, the length of the list must be equal to the number of transformations in the composition, and each dictionary will be passed as keyword arguments to the corresponding transformation in the sequence.

        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:

            - :obj:`transformed_points`: The transformed points of shape (n_points, output_dim).
            - :obj:`jacobian_dx`: The Jacobian matrix with respect to the input points of shape (n_points, output_dim, input_dim) if :obj:`dx` is True, otherwise None.
            - :obj:`jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (n_points, output_dim, n_params) if :obj:`dp` is True, otherwise None.
        """
        output_points = points
        jacobian_dx = None
        jacobian_dp = None
        is_first = True

        for i, t in enumerate(self.transforms):
            kwargs = list_kwargs[i] if list_kwargs is not None else {}
            output_points, jacobian_dx_i, jacobian_dp_i = t._transform(
                output_points, dx=dx or (dp and not is_first), dp=dp, **kwargs
            )
            if dx:
                if is_first:
                    jacobian_dx = jacobian_dx_i
                else:
                    jacobian_dx = numpy.einsum(
                        "...ik,...kj->...ij", jacobian_dx_i, jacobian_dx
                    )
            if dp:
                if is_first:
                    jacobian_dp = jacobian_dp_i
                else:
                    # Chain rule for parameters: dT/dp = dT/dT_prev * dT_prev/dp + dT/dp (if T has its own parameters)
                    jacobian_dp = numpy.einsum(
                        "...ik,...kj->...ij", jacobian_dx_i, jacobian_dp
                    )
                    if jacobian_dp_i is not None:
                        # If the current transformation has its own parameters, concatenate them to the existing Jacobian with respect to parameters
                        if jacobian_dp is not None:
                            jacobian_dp = numpy.concatenate(
                                [jacobian_dp, jacobian_dp_i], axis=-1
                            )
                        else:
                            jacobian_dp = jacobian_dp_i
            is_first = False

        return output_points, jacobian_dx, jacobian_dp

    def _inverse_transform(
        self,
        points: ArrayLike,
        *,
        dx: bool = False,
        dp: bool = False,
        list_kwargs: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the inverse of the composed transformation to the given points.

        This method applies the inverse of the individual transformations sequentially to the input points in reverse order, and computes the Jacobian matrices using the chain rule of differentiation.

        Parameters
        ----------
        points : ArrayLike
            The input points to be transformed. Shape (n_points, output_dim).

        dx : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the input points. Default is False.

        dp : bool, optional
            If True, compute the Jacobian of the transformed points with respect to the parameters of the transformation. Default is False.

        list_kwargs : Optional[List[Dict[str, Any]]], optional
            A list of dictionaries containing additional keyword arguments for each individual transformation.
            Default is None. If provided, the length of the list must be equal to the number of transformations in the composition, and each dictionary will be passed as keyword arguments to the corresponding transformation in reverse order in the sequence.

        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:

            - :obj:`transformed_points`: The transformed points of shape (n_points, input_dim).
            - :obj:`jacobian_dx`: The Jacobian matrix with respect to the input points of shape (n_points, input_dim, output_dim) if :obj:`dx` is True, otherwise None.
            - :obj:`jacobian_dp`: The Jacobian matrix with respect to the parameters of the transformation of shape (n_points, input_dim, n_params) if :obj:`dp` is True, otherwise None.
        """
        output_points = points
        jacobian_dx = None
        jacobian_dp = None
        is_first = True

        for i, t in enumerate(reversed(self.transforms)):
            kwargs = list_kwargs[-(i + 1)] if list_kwargs is not None else {}
            output_points, jacobian_dx_i, jacobian_dp_i = t._inverse_transform(
                output_points, dx=dx or (dp and not is_first), dp=dp, **kwargs
            )
            if dx:
                if is_first:
                    jacobian_dx = jacobian_dx_i
                else:
                    jacobian_dx = numpy.einsum(
                        "...ik,...kj->...ij", jacobian_dx_i, jacobian_dx
                    )
            if dp:
                if is_first:
                    jacobian_dp = jacobian_dp_i
                else:
                    # Chain rule for parameters: dT_inv/dp = dT_inv/dT_prev * dT_prev/dp + dT_inv/dp (if T has its own parameters)
                    jacobian_dp = numpy.einsum(
                        "...ik,...kj->...ij", jacobian_dx_i, jacobian_dp
                    )
                    if jacobian_dp_i is not None:
                        # If the current transformation has its own parameters, concatenate them to the existing Jacobian with respect to parameters
                        if jacobian_dp is not None:
                            jacobian_dp = numpy.concatenate(
                                [jacobian_dp, jacobian_dp_i], axis=-1
                            )
                        else:
                            jacobian_dp = jacobian_dp_i
            is_first = False

        return output_points, jacobian_dx, jacobian_dp
