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

from typing import Optional, Tuple, List
import numpy
from numbers import Number, Integral
import cv2

from ..core import Distortion
from ..core.package import Package
from ..optimize import optimize_input_points

class FisheyeDistortion(Distortion):
    r"""

    Subclass of the :class:`pycvcam.core.Distortion` class that represents the fisheye distortion model.

    .. note::

        This class represents the distortion transformation, which is the middle step of the process from the ``world_points`` to the ``image_points``.

    The ``FisheyeDistortion`` model in the using a polynomial model on the angle :math:`\theta` between the optical axis and the incoming ray.

    Lets consider ``normalized_points`` in the camera normalized coordinate system :math:`\vec{x}_n = (x_n, y_n)`, the corresponding ``distorted_points`` in the camera normalized coordinate system are given :math:`\vec{x}_d` can be obtained by :

    .. math::

        \vec{x}_d = \text{distort}(\vec{x}_n, \lambda_1, \lambda_2, \lambda_3, \ldots)

    The model of distortion is given by:

    .. math::

        \begin{bmatrix}
        x_d \\
        y_d
        \end{bmatrix}
        =
        \begin{bmatrix}
        x(r, \theta_d) \\
        y(r, \theta_d)
        \end{bmatrix}

    where :math:`r^2 = x_n^2 + y_n^2` and :math:`\theta_d` is the distorted angle given by:

    .. math::

        \theta_d = \theta(1 + d_1\theta^2 + d_2\theta^4 + d_3\theta^6 + \ldots)

    The number of parameters is variable and depends on the number of distortion coefficients :math:`d_i` used in the model.

    .. note::

        If the number of parameters is ``Nparams``, the maximum order of the polynomial is :math:`2*Nparams + 1`. and only odd powers of :math:`\theta` are used in the polynomial.
    
    .. warning::

        If the number of parameters ``Nparams`` is given during instantiation, the given parameters are truncated or extended to the given number of parameters.

    Parameters
    ----------
    parameters : Optional[numpy.ndarray], optional
        The parameters of the distortion transformation. It should be a numpy array of shape (Nparams,) containing the distortion coefficients ordered as described above. Default is None, which means no distortion is setted.

    Nparams : Optional[Integral], optional
        The number of parameters for the distortion model. If not specified, it will be inferred from the shape of the `parameters` array.

    Examples
    --------
    Create an distortion object with a specific number of parameters:

    .. code-block:: python

        import numpy
        from pycvcam import FisheyeDistortion

        parameters = numpy.array([0.1, 0.01, 0.02, 0.03, 0.001])

        distortion = FisheyeDistortion(parameters=parameters)

    Then you can use the distortion object to transform ``normalized_points`` to ``distorted_points``:

    .. code-block:: python

        normalized_points = numpy.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]) # shape (Npoints, 2)

        result = distortion.transform(normalized_points)
        distorted_points = result.distorted_points # Shape (Npoints, 2)
        print(distorted_points)

    You can also access to the jacobian of the distortion transformation:

    .. code-block:: python

        result = distortion.transform(normalized_points, dx=True, dp=True)
        distorted_points_dx = result.jacobian_dx  # Shape (Npoints, 2, 2)
        distorted_points_dp = result.jacobian_dp  # Shape (Npoints, 2, Nparams = 5)
        print(distorted_points_dx) 
        print(distorted_points_dp)

    The inverse transformation can be computed using the `inverse_transform` method:

    .. code-block:: python

        inverse_result = distortion.inverse_transform(distorted_points, dx=True, dp=True)
        normalized_points = inverse_result.normalized_points  # Shape (Npoints, 2)
        print(normalized_points)

    .. note::

        The jacobian with respect to the depth is not computed.
    
    .. seealso::

        For more information about the transformation process, see:

        - :meth:`pycvcam.FisheyeDistortion._transform` to transform the ``normalized_points`` to ``distorted_points``.
        - :meth:`pycvcam.FisheyeDistortion._inverse_transform` to transform the ``distorted_points`` back to ``normalized_points``.

    """
    def __init__(self, parameters: Optional[numpy.ndarray] = None, Nparams: Optional[Integral] = None) -> None:
        # Initialize the Transform base class
        super().__init__(parameters=parameters, constants=None)
        if Nparams is not None:
            self.Nparams = Nparams

    # =============================================
    # Overwrite some properties from the base class
    # =============================================
    @property
    def parameters(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the parameters of the distortion model.

        If None, no distortion is applied.

        .. note::

            If the number of parameters is ``Nparams``, the maximum order of the polynomial is :math:`2*Nparams + 1`. and only odd powers of :math:`\theta` are used in the polynomial.

        Parameters
        ----------
        parameters : numpy.ndarray, optional
            The parameters of the distortion model. If None, no distortion is applied. The default is None.

        Raises
        -------
        ValueError
            If the parameters is not a 1D numpy array.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Optional[numpy.ndarray]) -> None:
        if parameters is not None:
            parameters = numpy.asarray(parameters, dtype=Package.get_float_dtype())
            if parameters.ndim != 1:
                raise ValueError("The parameters should be a 1D numpy array.")
        self._parameters = parameters

    @property
    def constants(self) -> Optional[numpy.ndarray]:
        r"""
        Always returns None for the FisheyeDistortion class, as it does not have any constants.
        """
        return None
    
    @constants.setter
    def constants(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            raise ValueError("FisheyeDistortion model has no constants, must be set to None.")
        self._constants = None

    @property
    def Nparams(self) -> int:
        r"""
        Get or set the number of parameters of the distortion model.

        .. note::

            If the number of parameters is ``Nparams``, the maximum order of the polynomial is :math:`2*Nparams + 1`. and only odd powers of :math:`\theta` are used in the polynomial.

        If the given number of parameters is less than the current number of parameters, the parameters are truncated.
        If the given number of parameters is greater than the current number of parameters, the parameters are extended with zeros.

        Returns
        -------
        int
            The number of parameters of the distortion model.
        """
        if self.parameters is None:
            return 0
        else:
            return self.parameters.size
        
    @Nparams.setter
    def Nparams(self, value: Integral) -> None:
        if not isinstance(value, Integral):
            raise TypeError("The number of parameters should be an integer.")
        if value < 0:
            raise ValueError("The number of parameters should be a non-negative integer.")
        
        # If parameters is None, create a new array of zeros
        if self.parameters is None:
            self.parameters = numpy.zeros(value)
            return
        
        # Update the number of parameters instead of creating a new array
        if value == 0:
            self.parameters = None
        elif value < self.Nparams:
            self.parameters = self.parameters[:value]
        elif value > self.Nparams:
            self.parameters = numpy.concatenate((self.parameters, numpy.zeros(value - self.Nparams)))

    @property
    def parameter_names(self) -> List[str]:
        r"""
        Get the names of the parameters of the distortion transformation : ["d_1", "d_2", "d_3", "d_4", ...]

        Returns
        -------
        List[str]
            The names of the parameters of the distortion transformation.
        """
        params = [f"d_{i+1}" for i in range(self.Nparams)]
        return params
    
    @property
    def constant_names(self) -> List[str]:
        r"""
        Always returns an empty list for the FisheyeDistortion class, as it does not have any constants.
        """
        return []

    def is_set(self) -> bool:
        r"""
        Check if the distortion parameters are set (always True for FisheyeDistortion).

        Returns
        -------
        bool
            True if the distortion parameters are set, False otherwise.
        """
        return True

    
    # =================================================================
    # Distortion Model Coefficients
    # =================================================================
    def set_di(self, i: Integral, value: Number) -> None:
        r"""
        Set the coefficient for the i-th power of the polynomial decomposition.

        .. math::

            \theta_d = \theta(1 + d_1\theta^2 + d_2\theta^4 + d_3\theta^6 + \ldots)

        For i=3 set the coefficient :math:`d_3` associated to the :math:`\theta^7 = 2 * i + 1` term.

        Parameters
        ----------
        i : int
            The index of the distortion coefficient.
        value : float
            The value of the distortion coefficient.
        """
        if not isinstance(i, Integral):
            raise TypeError("The index of the distortion coefficient should be an integer.")
        i_min = 1
        i_max = self.Nparams
        if i < i_min or i > i_max:
            raise ValueError(f"The index of the distortion coefficient should be between {i_min} and {i_max}.")
        self.parameters[i - 1] = value

    def get_di(self, i: Integral) -> Number:
        r"""
        Get the coefficient for the i-th power of the polynomial decomposition.

        .. math::

            \theta_d = \theta(1 + d_1\theta^2 + d_2\theta^4 + d_3\theta^6 + \ldots)

        For i=3 return the coefficient :math:`d_3` associated to the :math:`\theta^7 = 2 * i + 1` term.

        Parameters
        ----------
        i : int
            The index of the distortion coefficient.

        Returns
        -------
        float
            The value of the distortion coefficient.
        """
        if not isinstance(i, Integral):
            raise TypeError("The index of the distortion coefficient should be an integer.")
        i_min = 1
        i_max = self.Nparams
        if i < i_min or i > i_max: 
            raise ValueError(f"The index of the distortion coefficient should be between {i_min} and {i_max}.")
        return self.parameters[i - 1]

    def make_empty(self) -> None:
        r"""
        Set to zero the parameters of the distortion model.
        """
        self.parameters = numpy.zeros((self.Nparams, ), dtype=Package.get_float_dtype())

    # =================================================================
    # Internal methods to compute the distortion
    # =================================================================
    def _cartesian_to_polar(self, cartesian: numpy.ndarray, dx: bool = True) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Convert cartesian coordinates to polar coordinates.
        
        The input ``cartesian`` points are given in the form :math:`(x, y)` with shape (Npoints, 2).
        The output ``polar`` points are given in the form :math:`(r, \theta)` with shape (Npoints, 2), where :math:`r = \sqrt{x^2 + y^2}` and :math:`\theta = \arctan2(y, x)`.

        .. math::

            \begin{bmatrix}
            r \\
            \theta \\
            \end{bmatrix}
            =
            \begin{bmatrix}
            \sqrt{x^2 + y^2} \\
            \arctan2(y, x) \\
            \end{bmatrix}

        The jacobian with respect to the cartesian points is an array with shape (Npoints, 2, 2).

        .. math::

            J = \begin{bmatrix}
            \frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} \\
            \frac{\partial \theta}{\partial x} & \frac{\partial \theta}{\partial y} \\
            \end{bmatrix}
            =
            \begin{bmatrix}
            \frac{x}{\sqrt{x^2 + y^2}} & \frac{y}{\sqrt{x^2 + y^2}} \\
            -\frac{y}{x^2 + y^2} & \frac{x}{x^2 + y^2} \\
            \end{bmatrix}

        .. warning::
            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``cartesian`` is (Npoints, 2) before calling this method.

        Parameters
        ----------
        cartesian : numpy.ndarray
            The cartesian points to be converted. Shape (Npoints, 2).
        
        dx : bool, optional
            If True, the jacobian with respect to the cartesian points is computed. Default is True.

        Returns
        -------
        polar : numpy.ndarray
            The polar points. Shape (Npoints, 2).
        
        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the polar points with respect to the cartesian points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        """
        if not isinstance(dx, bool):
            raise TypeError("The dx parameter must be a boolean.")

        # Extract the cartesian coordinates
        x = cartesian[:, 0] # shape (Npoints,)
        y = cartesian[:, 1] # shape (Npoints,)
        
        # Compute the polar coordinates
        r = numpy.sqrt(x**2 + y**2) # shape (Npoints,)
        theta = numpy.arctan2(y, x) # shape (Npoints,)

        polar = numpy.empty((cartesian.shape[0], 2), dtype=Package.get_float_dtype()) # shape (Npoints, 2)
        polar[:, 0] = r
        polar[:, 1] = theta

        # Compute the jacobian with respect to the cartesian points
        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.empty((cartesian.shape[0], 2, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 2)
            r_safe = numpy.where(r == 0, 1e-8, r) # Avoid division by zero
            r2 = r_safe ** 2 # shape (Npoints,)
            jacobian_dx[:, 0, 0] = x / r_safe # dr/dx
            jacobian_dx[:, 0, 1] = y / r_safe # dr/dy
            jacobian_dx[:, 1, 0] = -y / r2 # dtheta/dx
            jacobian_dx[:, 1, 1] = x / r2 # dtheta/dy

        return polar, jacobian_dx
    

    def _polar_to_cartesian(self, polar: numpy.ndarray, dx: bool = True) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Convert polar coordinates to cartesian coordinates.
        
        The input ``polar`` points are given in the form :math:`(r, \theta)` with shape (Npoints, 2), where :math:`r = \sqrt{x^2 + y^2}` and :math:`\theta = \arctan2(y, x)`.
        The output ``cartesian`` points are given in the form :math:`(x, y)` with shape (Npoints, 2).

        .. math::

            \begin{bmatrix}
            x \\
            y \\
            \end{bmatrix}
            =
            \begin{bmatrix}
            r \cos(\theta) \\
            r \sin(\theta) \\
            \end{bmatrix}

        The jacobian with respect to the polar points is an array with shape (Npoints, 2, 2).

        .. math::

            J = \begin{bmatrix}
            \frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta} \\
            \frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta} \\
            \end{bmatrix}
            =
            \begin{bmatrix}
            \cos(\theta) & -r\sin(\theta) \\
            \sin(\theta) & r\cos(\theta) \\
            \end{bmatrix}

        .. warning::
            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``polar`` is (Npoints, 2) before calling this method.

        Parameters
        ----------
        polar : numpy.ndarray
            The polar points to be converted. Shape (Npoints, 2).
        
        dx : bool, optional
            If True, the jacobian with respect to the polar points is computed. Default is True.

        Returns
        -------
        cartesian : numpy.ndarray
            The cartesian points. Shape (Npoints, 2).
        
        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the cartesian points with respect to the polar points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        """
        if not isinstance(dx, bool):
            raise TypeError("The dx parameter must be a boolean.")
        
        # Extract the polar coordinates
        r = polar[:, 0] # shape (Npoints,)
        theta = polar[:, 1] # shape (Npoints,)

        # Compute the cartesian coordinates
        x = r * numpy.cos(theta) # shape (Npoints,)
        y = r * numpy.sin(theta) # shape (Npoints,)

        cartesian = numpy.empty((polar.shape[0], 2), dtype=Package.get_float_dtype()) # shape (Npoints, 2)
        cartesian[:, 0] = x
        cartesian[:, 1] = y

        # Compute the jacobian with respect to the polar points
        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.empty((polar.shape[0], 2, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 2)
            jacobian_dx[:, 0, 0] = numpy.cos(theta) # dx/dr
            jacobian_dx[:, 0, 1] = -r * numpy.sin(theta) # dx/dtheta
            jacobian_dx[:, 1, 0] = numpy.sin(theta) # dy/dr
            jacobian_dx[:, 1, 1] = r * numpy.cos(theta) # dy/dtheta

        return cartesian, jacobian_dx


    # =================================================================
    # Implementation of the transform method
    # =================================================================
    def _transform(self, normalized_points: numpy.ndarray, *, dx: bool = False, dp: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the ``normalized_points`` to the ``distorted_points``.

        Lets consider ``normalized_points`` in the camera normalized coordinate system :math:`\vec{x}_n = (x_n, y_n)`, the corresponding ``distorted_points`` in the camera normalized coordinate system are given :math:`\vec{x}_d` can be obtained by :

        .. math::

            \vec{x}_d = \text{distort}(\vec{x}_n, \lambda_1, \lambda_2, \lambda_3, \ldots)

        The model of distortion is given by:

        .. math::

            \begin{bmatrix}
            x_d \\
            y_d
            \end{bmatrix}
            =
            \begin{bmatrix}
            x(r, \theta_d) \\
            y(r, \theta_d)
            \end{bmatrix}

        where :math:`r^2 = x_n^2 + y_n^2` and :math:`\theta_d` is the distorted angle given by:

        .. math::

            \theta_d = \theta(1 + d_1\theta^2 + d_2\theta^4 + d_3\theta^6 + \ldots)

        The jacobians with respect to the distortion parameters is an array with shape (Npoints, 2, Nparams), where the last dimension represents the parameters in the order of the class attributes (d1, d2, d3, ...).
        The jacobian with respect to the normalized points is an array with shape (Npoints, 2, 2).

        .. warning::

            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``normalized_points`` is (Npoints, 2) before calling this method.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points in camera normalized coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the normalized points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the distortion parameters is computed. Default is False

        opencv : bool, optional
            If True, the distortion transformation is achieved using the OpenCV function ``projectPoints``.
            If False, the distortion transformation is achieved using the internal method.
            Default is False.

        Returns
        -------
        distorted_points : numpy.ndarray
            The distorted points in camera normalized coordinates. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the normalized points. Shape (Npoints, 2, 2) if dx is True, otherwise None.

        jacobian_dp : Optional[numpy.ndarray]
            The jacobian of the distorted points with respect to the distortion parameters. Shape (Npoints, 2, Nparams) if dp is True, otherwise None.
        """
        # Prepare the inputs data for distortion
        x_N = normalized_points[:, 0] # shape (Npoints,)
        y_N = normalized_points[:, 1] # shape (Npoints,)
        Npoints = normalized_points.shape[0]
        Nparams = self.Nparams

        # Convert to polar coordinates
        polar, jacobian_dx_cart2pol = self._cartesian_to_polar(normalized_points, dx=dx) # shape (Npoints, 2), (Npoints, 2, 2) or None

        r = polar[:, 0] # shape (Npoints,)
        theta = polar[:, 1] # shape (Npoints,)

        # Apply the distortion model
        theta_powers = numpy.power(theta[:, numpy.newaxis], 2 * numpy.arange(1, Nparams + 1)) # shape (Npoints, Nparams) # theta^2, theta^4, theta^6, ...
        theta_d = theta * (1 + numpy.dot(theta_powers, self.parameters)) # shape (Npoints,)

        if dx:
            theta_d_dx = numpy.empty((Npoints, 1, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 1, 2)
            # dtheta_d/dtheta = 1 + 3 * d1 * theta^2 + 5 * d2 * theta^4 + 7 * d3 * theta^6 + ...
            coefficients = numpy.array([2*i + 1 for i in range(1, Nparams + 1)], dtype=Package.get_float_dtype()) # shape (Nparams,)
            dtheta_d_dtheta = 1 + numpy.dot(theta_powers, self.parameters * coefficients) # shape (Npoints,)
            theta_d_dx[:, 0, :] = dtheta_d_dtheta[:, numpy.newaxis] * jacobian_dx_cart2pol[:, 1, :] # shape (Npoints, 2)
    
        if dp and Nparams > 0:
            theta_d_dp = numpy.empty((Npoints, 1, Nparams), dtype=Package.get_float_dtype()) # shape (Npoints, 1, Nparams)
            theta_d_dp[:, 0, :] = theta[:, numpy.newaxis] * theta_powers # shape (Npoints, Nparams)
        
        # Convert back to cartesian coordinates
        distorted_points, jacobian_dx_pol2cart = self._polar_to_cartesian(numpy.column_stack((r, theta_d)), dx=dx) # shape (Npoints, 2), (Npoints, 2, 2) or None

        # Compute the jacobians
        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.empty((Npoints, 2, 2), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 2)
            jacobian_dx[:, 0, 0] = jacobian_dx_pol2cart[:, 0, 0] * jacobian_dx_cart2pol[:, 0, 0] + jacobian_dx_pol2cart[:, 0, 1] * theta_d_dx[:, 0, 0] # dx/dx_N
            jacobian_dx[:, 0, 1] = jacobian_dx_pol2cart[:, 0, 0] * jacobian_dx_cart2pol[:, 0, 1] + jacobian_dx_pol2cart[:, 0, 1] * theta_d_dx[:, 0, 1] # dx/dy_N
            jacobian_dx[:, 1, 0] = jacobian_dx_pol2cart[:, 1, 0] * jacobian_dx_cart2pol[:, 0, 0] + jacobian_dx_pol2cart[:, 1, 1] * theta_d_dx[:, 0, 0] # dy/dx_N
            jacobian_dx[:, 1, 1] = jacobian_dx_pol2cart[:, 1, 0] * jacobian_dx_cart2pol[:, 0, 1] + jacobian_dx_pol2cart[:, 1, 1] * theta_d_dx[:, 0, 1] # dy/dy_N

        jacobian_dp = None
        if dp and Nparams > 0:
            jacobian_dp = numpy.empty((Npoints, 2, Nparams), dtype=Package.get_float_dtype()) # shape (Npoints, 2, Nparams)
            jacobian_dp[:, 0, :] = jacobian_dx_pol2cart[:, 0, 1][:, numpy.newaxis] * theta_d_dp[:, 0, :] # dx/dp
            jacobian_dp[:, 1, :] = jacobian_dx_pol2cart[:, 1, 1][:, numpy.newaxis] * theta_d_dp[:, 0, :] # dy/dp
        
        if dp and Nparams == 0:
            jacobian_dp = numpy.empty((Npoints, 2, 0), dtype=Package.get_float_dtype()) # shape (Npoints, 2, 0)
        
        return distorted_points, jacobian_dx, jacobian_dp
    
    def _inverse_transform(self, distorted_points: numpy.ndarray, *, dx: bool = False, dp: bool = False, **kwargs) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from the ``distorted_points`` to the ``normalized_points``.

        Lets consider ``distorted_points`` in the camera normalized coordinate system :math:`\vec{x}_d = (x_d, y_d)`, the corresponding ``normalized_points`` in the camera normalized coordinate system are obtained by an ``iterative`` algorithm that finds the points :math:`\vec{x}_n`.

        .. seealso::

            - :func:`pycvcam.optimize.optimize_input_points` for the implementation of the iterative algorithm to find the inverse distortion points.

        The initial guess is setted to :math:`\vec{x}_{n} = \vec{x}_{d} - U(\vec{x}_{d})``, where :math:`U(\vec{x}_{d})` is the distortion field applied to the distorted points.

        .. warning::

            This method is not intended to be used directly, but rather through the :meth:`pycvcam.core.Transform.transform` method.
            Please ensure, the shape of the input ``image_points`` is (Npoints, 2) before calling this method.

            The jacobians with respect to the distortion parameters and the distorted points are always None, since it is an iterative algorithm.

        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points in camera normalized coordinates to be transformed. Shape (Npoints, 2).

        dx : bool, optional
            If True, the jacobian with respect to the distorted points is computed. Default is False

        dp : bool, optional
            If True, the jacobian with respect to the distortion parameters is computed. Default is False

        **kwargs : dict, optional
            Additional keyword arguments to pass to the iterative algorithm. Not used in this implementation.

        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points in camera normalized coordinates, which are equal to the x and y components of the image points. Shape (Npoints, 2).

        jacobian_dx : Optional[numpy.ndarray]
            Always None, since the jacobian with respect to the distorted points is not computed by an iterative algorithm.

        jacobian_dp : Optional[numpy.ndarray]
            Always None, since the jacobian with respect to the distortion parameters is not computed by an iterative algorithm.
        """
        if dx or dp:
            print("\n[WARNING]: Undistortion with dx=True or dp=True. The jacobians cannot be computed with this method. They are always None.\n")

        normalized_points = optimize_input_points(
            self,
            distorted_points,
            guess = 2 * distorted_points - self._transform(distorted_points, dx=False, dp=False)[0],
            _skip = True,  # Skip the checks on the input points
            **kwargs
        )

        return normalized_points, None, None