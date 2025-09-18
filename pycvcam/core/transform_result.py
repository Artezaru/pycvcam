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

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import numpy

@dataclass(slots=True)
class TransformResult:
    r"""
    A class to represent the result of a transformation.

    This class is used to store the results of a transformation, including the transformed points and the Jacobian matrices.

    .. seealso::

        - :class:`pycvcam.core.Transform` for the base class of all transformations.
        - :meth:`pycvcam.core.Transform.transform` for applying the transformation to points.
        - :meth:`pycvcam.core.Transform.inverse_transform` for applying the inverse transformation to points (`output_dim` and `input_dim` are swapped).

    For a transformation from :math:`\mathbb{R}^{input\_dim}` to :math:`\mathbb{R}^{output\_dim}`, the input points are assumed to have shape (..., input_dim) and the output points will have shape (..., output_dim).
    
    The Jacobian matrices are computed with respect to the input points and the parameters of the transformation:

    - The Jacobian with respect to the input points has shape (..., output_dim, input_dim).
    - The Jacobian with respect to the parameters has shape (..., output_dim, Nparams), where Nparams is the number of parameters of the transformation.

    .. note::

        If ``transpose`` is set to True during the transformation, the output points will have shape (output_dim, ...) instead of (..., output_dim), same for the Jacobian matrices (ie. shape (output_dim, ..., input_dim) and (output_dim, ..., Nparams) respectively).

    Attributes
    ----------
    transformed_points : numpy.ndarray
        The transformed points after applying the transformation.
        Shape (..., output_dim).

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the input points.
        Shape (..., output_dim, input_dim).

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian matrix with respect to the parameters of the transformation.
        Shape (..., output_dim, Nparams).

    transpose : bool
        If True, the output points and Jacobian matrices will have shape (output_dim, ...) instead of (..., output_dim). True if set during the transformation, otherwise False.

        
    Jacobians Short-hand Notations
    -------------------------------
    Short-hand notations for the Jacobian matrices can be added to the `TransformResult` class using the `add_jacobian` method. This allows adding custom views of the ``jacobian_dp`` matrix with respect to the parameters of the transformation.

    .. code-block:: python

        result = TransformResult(transformed_points, jacobian_dx, jacobian_dp)
        result.add_jacobian("dk", start=0, end=2, doc="Custom Jacobian view for the first two parameters related to k1 and k2")

        result.jacobian_dk  # Returns a view of the jacobian_dp matrix for parameters k1 and k2, i.e., jacobian_dp[..., 0:2]


    Aliases for Transformed Points
    -------------------------------
    Aliases can be added to the `TransformResult` class to provide more convenient access to the transformed points, depending on the context of the transformation, via the `add_alias` method.

    .. code-block:: python

        result = TransformResult(transformed_points, jacobian_dx, jacobian_dp)
        result.add_alias("image_points")

        result.image_points  # Returns the transformed_points array

    """
    transformed_points: numpy.ndarray
    jacobian_dx: Optional[numpy.ndarray] = None
    jacobian_dp: Optional[numpy.ndarray] = None
    transpose: bool = False
    _custom_jacobians: Dict[str, Tuple[int, int, Optional[str]]] = field(default_factory=dict, init=False, repr=False) # To avoid mutability issues, we use field with default_factory
    _custom_aliases: List[str] = field(default_factory=list, init=False, repr=False)

    def add_jacobian(self, name: str, start: int, end: int, doc: Optional[str] = None) -> None:
        r"""
        Add a custom view of the `jacobian_dp` matrix to the `TransformResult` object.

        This method allows to add custom views of the `jacobian_dp` matrix with respect to the parameters of the transformation.
        The custom Jacobian can be accessed using the `name` attribute.

        Parameters
        ----------
        name : str
            The name of the custom Jacobian view.
        
        start : int
            The starting index of the parameters to include in the custom Jacobian view.
        
        end : int
            The ending index of the parameters to include in the custom Jacobian view.
        
        doc : Optional[str], optional
            A documentation string for the custom Jacobian view. Default is None.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name)}")
        if not name.isidentifier():
            raise ValueError(f"name must be a valid identifier to add as jacobian_name, got {name=}")
        if not isinstance(start, int):
            raise TypeError(f"start must be an integer, got {type(start)}")
        if not isinstance(end, int):
            raise TypeError(f"end must be an integer, got {type(end)}")
        if not doc is None and not isinstance(doc, str):
            raise TypeError(f"doc must be a string, got {type(doc)}")
        
        if self.jacobian_dp is not None:
            
            if start < 0 or end < 0 or start > end or end > self.jacobian_dp.shape[-1]:
                raise ValueError(f"Invalid range for custom Jacobian view: start={start}, end={end}, Nparams={self.jacobian_dp.shape[-1]}")
            
            self._custom_jacobians[name] = (start, end, doc)

    def add_alias(self, name: str) -> None:
        r"""
        Add an alias for the transformed points in the `TransformResult` object.

        This method allows to add an alias for the transformed points, which can be used to access the transformed points using a more convenient name.

        Parameters
        ----------
        name : str
            The name of the alias to add.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name)}")
        if not name.isidentifier():
            raise ValueError(f"name must be a valid identifier, got {name}")
        
        if name in self._custom_aliases:
            raise ValueError(f"Alias '{name}' already exists.")
        if name in self._custom_jacobians:
            raise ValueError(f"Alias '{name}' conflicts with an existing Jacobian view.")
        if hasattr(self, name):
            raise ValueError(f"Alias '{name}' conflicts with an existing attribute of TransformResult.")
        
        self._custom_aliases.append(name)

    def __getattr__(self, key):
        if key in self._custom_aliases:
            return self.transformed_points
        if key.startswith("jacobian_"):
            name = key[len("jacobian_"):]
            if name in self._custom_jacobians:
                if self.jacobian_dp is None:
                    return None
                start, end, _ = self._custom_jacobians[name]
                return self.jacobian_dp[..., start:end]
        raise AttributeError(f"'TransformResult' object has no attribute '{key}'")

    def print_help(self):
        r"""
        Print the descriptions of the properties and custom Jacobian views.

        This method prints the names and documentation strings of the custom Jacobian views added to the `TransformResult` object.
        """
        print("transformed_points: The transformed points after applying the transformation with shape (..., output_dim)")
        for alias in self._custom_aliases:
            print(f"{alias}: Alias for transformed_points, same shape (..., output_dim)")
        print("jacobian_dx: The Jacobian matrix with respect to the input points with shape (..., output_dim, input_dim) [or None if not computed]")
        print("jacobian_dp: The Jacobian matrix with respect to the parameters of the transformation with shape (..., output_dim, Nparams) [or None if not computed]")
        for name, (start, end, doc) in self._custom_jacobians.items():
            print(f"jacobian_{name}: {doc if doc is not None else 'No description provided'} with shape (..., output_dim, {end - start}) [or None if not computed], view of jacobian_dp[..., {start}:{end}]")






