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
import pyzernike

from ..core import Distortion
from ..optimize import optimize_input_points
from ..core.package import Package


class ZernikeDistortion(Distortion):
    r"""

    Subclass of the :class:`pycvcam.core.Distortion` class that represents a Zernike decomposition distortion model.

    .. note::

        This class represents the distortion transformation, which is the middle step of the process from the ``world_points`` to the ``image_points``.

    The ``ZernikeDistortion`` model consists of a Zernike polynomial decomposition of the distortion along the x and y axes. The distortion is represented as a sum of Zernike polynomials, which are orthogonal polynomials defined on the unit disk.

    Lets consider ``normalized_points`` in the camera normalized coordinate system :math:`\vec{x}_n = (x_n, y_n)`, the corresponding ``distorted_points`` in the camera normalized coordinate system are given :math:`\vec{x}_d` can be obtained by :

    .. math::

        x_{d} = x_{n} + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} Z_{nm}(\rho, \theta)

    .. math::

        y_{d} = y_{n} + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{y}_{n,m} Z_{nm}(\rho, \theta)

    where :math:`Z_{nm}(\rho, \theta)` are the Zernike polynomials, :math:`C^{x}_{n,m}` and :math:`C^{y}_{n,m}` are the
    Zernike coefficients for the x and y coordinates, respectively, and :math:`\rho` and :math:`\theta` are the
    polar coordinates of the normalized points in the defined unit ellipse with radius :math:`R_x, R_y` and center :math:`(x_0, y_0)`:

    .. math::

        \rho = \sqrt{\left(\frac{x_{n} - x_{0}}{R_x}\right)^2 + \left(\frac{y_{n} - y_{0}}{R_y}\right)^2}

    .. math::

        \theta = \arctan2(y_{n} - y_{0}, x_{n} - x_{0})

    .. note::

        For more informations about Zernike polynomials, see the package pyzernike (https://github.com/Artezaru/pyzernike).

    This transformation is caracterized by Nparams parameters and 4 constants:

    - Nparams parameters :math:`C^{x}_{n,m}` and :math:`C^{y}_{n,m}` for the Zernike coefficients.
    - 4 constants :math:`(R_x, R_y, x_0, y_0)` for the radius and center of the ellipse of definition of the Zernike polynomials.

    Only coefficients for :math:`n \leq N_{zer}` and :math:`m \in [-n, n]` and :math:`n-m \equiv 0 \mod 2` are stored.

    The coefficients are storred in a ``parameters`` 1D-array with the OSA/ANSI standard indices but with a x/y separation:

    - C^{x}_{0,0}, parameters[0] for the x coordinate :math:`n=0, m=0`
    - C^{y}_{0,0}, parameters[1] for the y coordinate :math:`n=0, m=0`
    - C^{x}_{1,-1}, parameters[2] for the x coordinate :math:`n=1, m=-1`
    - C^{y}_{1,-1}, parameters[3] for the y coordinate :math:`n=1, m=-1`
    - C^{x}_{1,1}, parameters[4] for the x coordinate :math:`n=1, m=1`
    - C^{y}_{1,1}, parameters[5] for the y coordinate :math:`n=1, m=1`
    - C^{x}_{2,-2}, parameters[6] for the x coordinate :math:`n=2, m=-2`
    - C^{y}_{2,-2}, parameters[7] for the y coordinate :math:`n=2, m=-2`
    - C^{x}_{2,0}, parameters[8] for the x coordinate :math:`n=2, m=0`
    - C^{y}_{2,0}, parameters[9] for the y coordinate :math:`n=2, m=0`
    - C^{x}_{2,2}, parameters[10] for the x coordinate :math:`n=2, m=2`
    - C^{y}_{2,2}, parameters[11] for the y coordinate :math:`n=2, m=2`
    - ...

    If the number of input parameters is not equal to the number of parameters required by the model, the other parameters are set to 0.

    The number of parameters is given by the formula:

    .. math::

        N_{params} = (N_{zer}+1)(N_{zer}+2)

    +---------------------------+---------------------------------+-------------------------------------+
    | Ordre of Zernike ``Nzer`` | Nparameters for X or Y          | Nparameters in model ``Nparams``    |
    +===========================+=================================+=====================================+
    | None                      | 0                               | 0                                   |
    +---------------------------+---------------------------------+-------------------------------------+
    | 0                         | 1                               | 2                                   |
    +---------------------------+---------------------------------+-------------------------------------+
    | 1                         | 3                               | 6                                   |
    +---------------------------+---------------------------------+-------------------------------------+
    | 2                         | 6                               | 12                                  |
    +---------------------------+---------------------------------+-------------------------------------+
    | 3                         | 10                              | 20                                  |
    +---------------------------+---------------------------------+-------------------------------------+
    | 4                         | 15                              | 30                                  |
    +---------------------------+---------------------------------+-------------------------------------+

    .. warning::

        If the ordre of the zernike polynomials ``Nzer`` is given during instantiation, the given parameters are truncated or extended to the given number of parameters. Same for the number of parameters ``Nparams``.

    To compute the Distortion, the user must define the unit circle in which the normalized points are defined.
    The unit circle is defined by the radius :math:`R` and the center :math:`(x_{0}, y_{0})`.

    Parameters
    ----------
    parameters : Optional[numpy.ndarray], optional
        The parameters of the distortion transformation. It should be a numpy array of shape (Nparams,) containing the distortion coefficients ordered as described above. Default is None, which means no distortion is setted.

    constants : numpy.ndarray, optional
        The constants of the distortion transformation. It should be a numpy array of shape (4,) containing the radius and center of the ellipse of definition of the Zernike polynomials in the order: (R_x, R_y, x_0, y_0). Default is [1.0, 1.0, 0.0, 0.0], which means the unit circle is defined with radius 1 and center (0, 0).

    Nparams : Optional[Integral], optional
        The number of parameters for the distortion model. If not specified, it will be inferred from the shape of the `parameters` array.

    Nzer : int, optional
        The order of the Zernike polynomials. If None, the order is set according to the number of parameters. The default is None.
        Only use ``Nzer`` or ``Nparams``, not both.

    Examples
    --------
    Create an distortion object with a specific order of Zernike polynomials and parameters:

    .. code-block:: python

        import numpy
        from pycvcam import Cv2Distortion

        # Create a distortion object with 6 parameters
        distortion = ZernikeDistortion(numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])) # Model with Nzer=1, -> Nparams=6

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

        - :meth:`pycvcam.Cv2Distortion._transform` to transform the ``normalized_points`` to ``distorted_points``.
        - :meth:`pycvcam.Cv2Distortion._inverse_transform` to transform the ``distorted_points`` back to ``normalized_points``.

    If you want to define the Zernike unit disk to encapsulate an image, with a centered distortion and a circular distortion in the image plane, you can use the `constants` parameter:

    .. code-block:: python

        import numpy
        import cv2
        from pycvcam import Cv2Distortion

        # Load the image
        image = cv2.imread('image.jpg')
        image_height, image_width = image.shape[:2]

        # Compute the center and radius of the unit disk in the image plane
        x0 = (image_width - 1) / 2
        y0 = (image_height - 1) / 2
        R_x = R_y = numpy.sqrt(((image_width - 1) / 2) ** 2 + ((image_height - 1) / 2) ** 2)

        # Extract the intrinsic focal length (fx, fy) from the camera calibration and the principal point (cx, cy) form the intrinsic transformation
        x0 = (x0 - cx) / fx
        y0 = (y0 - cy) / fy
        R_x /= fx
        R_y /= fy        

        # Create a distortion object with a specific unit disk
        constants = numpy.array([R_x, R_y, x0, y0])
        distortion = ZernikeDistortion(
            parameters=numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            constants=constants
        )

    """
    def __init__(self, parameters: Optional[numpy.ndarray] = None, constants: Optional[numpy.ndarray] = None, Nparams: Optional[Integral] = None, Nzer: Optional[int] = None) -> None:
        # Initialize the Transform base class
        super().__init__(parameters=parameters, constants=constants)
        if Nparams is not None and Nzer is not None:
            raise ValueError("You can only use one of Nparams or Nzer, not both.")
        if Nzer is not None:
            self.Nzer = Nzer
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

        The number of parameters should be a 1D numpy array with a size in [None, 2, 6, 12, 20, 30, ...].

        .. math::

            N_{params} = (N_{zer}+1)(N_{zer}+2)

        If the number of input parameters is not equal to the number of parameters required by the model, the other parameters are set to 0.

        The parameters are set in the following order:

        - N = 0 parameters : similar than None
        - N = 2 parameters : :math:`C^{x}_{0,0}, C^{y}_{0,0}` : Zernike coefficients for the x and y coordinates with order 0
        - N = 6 parameters : :math:`C^{x}_{0,0}, C^{y}_{0,0}, C^{x}_{1,-1}, C^{y}_{1,-1}, C^{x}_{1,1}, C^{y}_{1,1}` : Zernike coefficients for the x and y coordinates with order 0 and 1
        - N = 12 parameters : :math:`C^{x}_{0,0}, C^{y}_{0,0}, C^{x}_{1,-1}, C^{y}_{1,-1}, C^{x}_{1,1}, C^{y}_{1,1}, C^{x}_{2,-2}, C^{y}_{2,-2}, C^{x}_{2,0}, C^{y}_{2,0}, C^{x}_{2,2}, C^{y}_{2,2}` : Zernike coefficients for the x and y coordinates with order 0, 1 and 2
        - N = 20 parameters : ...

        To easily use the parameters, you can use the methods ``get_Cx``, ``set_Cx``, ``get_Cy`` and ``set_Cy`` to get and set the Zernike coefficients for the x and y coordinates.

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
            # Extend the number of parameters to a valid number
            Nzer = 0
            while (Nzer + 1) * (Nzer + 2) < parameters.size:
                Nzer += 1
            Nparams = (Nzer + 1) * (Nzer + 2)
            # Extend the parameters to the next valid size
            if Nparams > parameters.size:
                parameters = numpy.concatenate((parameters, numpy.zeros(Nparams - parameters.size)))
            # Set to None if the number of parameters is 0
            if parameters.size == 0:
                parameters = None
        self._parameters = parameters

    @property
    def constants(self) -> numpy.ndarray:
        r"""
        Get or set the constants of the distortion model.

        The constants are a numpy array of shape (4,) containing the radius and center of the ellipse of definition of the Zernike polynomials in the order: (R_x, R_y, x_0, y_0).
        
        If None, the default constants are set to [1.0, 1.0, 0.0, 0.0], which means the unit circle is defined with radius 1 and center (0, 0).

        Returns
        -------
        numpy.ndarray
            The constants of the distortion model. An array of shape (4,) containing the radius and center of the ellipse of definition of the Zernike polynomials in the order: (R_x, R_y, x_0, y_0).

        Raises
        -------
        ValueError
            If the constants is not a 1D numpy array of shape (4,).
        """
        return self._constants
    
    @constants.setter
    def constants(self, value: Optional[numpy.ndarray]) -> None:
        if value is None:
            self._constants = numpy.array([1.0, 1.0, 0.0, 0.0], dtype=Package.get_float_dtype())
        else:
            value = numpy.asarray(value, dtype=Package.get_float_dtype())
            if value.ndim != 1 or value.size != 4:
                raise ValueError("The constants should be a 1D numpy array of shape (4,).")
            if not numpy.isfinite(value).all():
                raise ValueError("The constants should be finite values.")
            if not numpy.all(value[:2] > 0):
                raise ValueError("The radius constants R_x and R_y should be positive values.")
            self._constants = value

    @property
    def Nparams(self) -> int:
        r"""
        Get or set the number of parameters of the distortion model.

        The given number of parameters must be in [0, 2, 6, 12, 20, 30, ...].

        .. math::

            N_{params} = (N_{zer}+1)(N_{zer}+2)

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
        if value == 0:
            self.parameters = None
            return
        
        Nzer = 0
        while (Nzer + 1) * (Nzer + 2) < value:
            Nzer += 1

        # Check if the number of parameters is valid
        if (Nzer + 1) * (Nzer + 2) != value:
            raise ValueError("The number of parameters should be in [0, 2, 6, 12, 20, 30, ...].")
        
        # If parameters is None, create a new array of zeros
        if self.parameters is None:
            self.parameters = numpy.zeros(value)
            return
        
        if value < self.Nparams:
            self.parameters = self.parameters[:value]
        elif value > self.Nparams:
            self.parameters = numpy.concatenate((self.parameters, numpy.zeros(value - self.Nparams)))

    @property
    def Nzer(self) -> Optional[int]:
        r"""
        Get or set the order of the Zernike polynomials.

        The given order must be in [0, 1, 2, 3, 4, ...].

        If the given order is less than the current order, the parameters are truncated.
        If the given order is greater than the current order, the parameters are extended with zeros.

        None means no distortion is applied.

        Returns
        -------
        Optional[int]
            The order of the Zernike polynomials.
        """
        if self.parameters is None:
            return None
        
        Nparams = self.Nparams
        Nzer = 0
        while (Nzer + 1) * (Nzer + 2) < Nparams:
            Nzer += 1
        return Nzer
    
    @Nzer.setter
    def Nzer(self, value: Optional[int]) -> None:
        if not isinstance(value, Integral):
            raise TypeError("The order of the Zernike polynomials should be an integer.")
        if value < 0:
            raise ValueError("The order of the Zernike polynomials should be a non-negative integer.")
        
        value = int(value)
        self.Nparams = (value + 1) * (value + 2)

    @property
    def parameter_names(self) -> List[str]:
        r"""
        Get the names of the parameters of the distortion transformation : ["Cx(0, 0), Cy(0, 0), Cx(1, -1), Cy(1, -1), Cx(1, 1), Cy(1, 1), ...]

        Returns
        -------
        List[str]
            The names of the parameters of the distortion transformation.
        """
        return [val for pair in zip(self.parameter_x_names, self.parameter_y_names) for val in pair]
    
    @property
    def parameter_x_names(self) -> List[str]:
        r"""
        Get the names of the parameters along the x-axis of the distortion transformation : ["Cx(0, 0), Cx(1, -1), Cx(1, 1), ...]

        Returns
        -------
        List[str]
            The names of the parameters of the distortion transformation.
        """
        return [f"Cx({n}, {m})" for n in range(self.Nzer + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0]

    @property
    def parameter_y_names(self) -> List[str]:
        r"""
        Get the names of the parameters along the y-axis of the distortion transformation : ["Cy(0, 0), Cy(1, -1), Cy(1, 1), ...]

        Returns
        -------
        List[str]
            The names of the parameters of the distortion transformation.
        """
        return [f"Cy({n}, {m})" for n in range(self.Nzer + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0]

    @property
    def constant_names(self) -> List[str]:
        r"""
        Get the names of the constants of the distortion transformation : ["R_x", "R_y", "x_0", "y_0"]

        Returns
        -------
        List[str]
            The names of the constants of the distortion transformation.
        """
        return ["R_x", "R_y", "x_0", "y_0"]

    # =================================================================
    # Radius and center of the unit disk
    # =================================================================
    @property
    def radius(self) -> float:
        r"""
        Get or set the radius of the unit circle in which the normalized points are defined.

        The value will be propagated to both `radius_x` and `radius_y`.
        The radius should be a positive number.

        Returns
        -------
        float
            The radius of the unit circle.
        """
        if abs(self.radius_x - self.radius_y) > 1e-6:
            raise ValueError("The radius_x and radius_y should be the same for the unit circle. Please set the same value for both or use radius_x and radius_y separately for ellipses.")
        return self.radius_x
    
    @radius.setter
    def radius(self, value: Number) -> None:
        if not isinstance(value, Number):
            raise TypeError("The radius should be a float or an integer.")
        if value <= 0:
            raise ValueError("The radius should be a positive number.")
        self._constants[0] = float(value)
        self._constants[1] = float(value)  # Set radius_y to the same value as radius_x

    @property
    def radius_x(self) -> float:
        r"""
        Get or set the radius of the unit circle in the x direction.

        The radius should be a positive number.

        .. note::

           An alias for the radius of the unit circle in the x direction is ``rx``.

        Returns
        -------
        float
            The radius of the unit circle in the x direction.
        """
        return self._constants[0]
    
    @radius_x.setter
    def radius_x(self, value: Number) -> None:
        if not isinstance(value, Number):
            raise TypeError("The radius_x should be a float or an integer.")
        if value <= 0:
            raise ValueError("The radius_x should be a positive number.")
        self._constants[0] = float(value)

    @property
    def rx(self) -> float:
        return self.radius_x
    
    @rx.setter
    def rx(self, value: Number) -> None:
        self.radius_x = value

    @property
    def radius_y(self) -> float:
        r"""
        Get or set the radius of the unit circle in the y direction.

        The radius should be a positive number.

        .. note::

              An alias for the radius of the unit circle in the y direction is ``ry``.

        Returns
        -------
        float
            The radius of the unit circle in the y direction.
        """
        return self._constants[1]
    
    @radius_y.setter
    def radius_y(self, value: Number) -> None:
        if not isinstance(value, Number):
            raise TypeError("The radius_y should be a float or an integer.")
        if value <= 0:
            raise ValueError("The radius_y should be a positive number.")
        self._constants[1] = float(value)

    @property
    def ry(self) -> float:
        return self.radius_y

    @ry.setter
    def ry(self, value: Number) -> None:
        self.radius_y = value

    @property
    def center(self) -> numpy.ndarray:
        r"""
        Get or set the center of the unit circle in which the normalized points are defined.

        The center is a numpy array of shape (2,) containing the x and y coordinates of the center.

        Returns
        -------
        numpy.ndarray
            The center of the unit circle.
        """
        return self._constants[2:4]
    
    @center.setter
    def center(self, value: numpy.ndarray) -> None:
        value = numpy.asarray(value, dtype=Package.get_float_dtype())
        if value.ndim != 1 or value.size != 2:
            raise ValueError("The center should be a 1D numpy array of shape (2,).")
        if not numpy.isfinite(value).all():
            raise ValueError("The center should be finite values.")
        self._constants[2:4] = value

    @property
    def center_x(self) -> float:
        r"""
        Get or set the x coordinate of the center of the unit circle in which the normalized points are defined.

        The center_x should be a finite number.

        .. note::

            An alias for the x coordinate of the center of the unit circle is ``x0``.

        Returns
        -------
        float
            The x coordinate of the center of the unit circle.
        """
        return self._constants[2]
    
    @center_x.setter
    def center_x(self, value: Number) -> None:
        if not isinstance(value, Number):
            raise TypeError("The center_x should be a float or an integer.")
        if not numpy.isfinite(value):
            raise ValueError("The center_x should be a finite number.")
        self._constants[2] = float(value)

    @property
    def x0(self) -> float:
        return self.center_x

    @x0.setter
    def x0(self, value: Number) -> None:
        self.center_x = value

    @property
    def center_y(self) -> float:
        r"""
        Get or set the y coordinate of the center of the unit circle in which the normalized points are defined.

        The center_y should be a finite number.

        .. note::

            An alias for the y coordinate of the center of the unit circle is ``y0``.

        Returns
        -------
        float
            The y coordinate of the center of the unit circle.
        """
        return self._constants[3]
    
    @center_y.setter
    def center_y(self, value: Number) -> None:
        if not isinstance(value, Number):
            raise TypeError("The center_y should be a float or an integer.")
        if not numpy.isfinite(value):
            raise ValueError("The center_y should be a finite number.")
        self._constants[3] = float(value)

    @property
    def y0(self) -> float:
        return self.center_y

    @y0.setter
    def y0(self, value: Number) -> None:
        self.center_y = value

    def is_set(self) -> bool:
        r"""
        Check if the distortion parameters are set (always True for ZernikeDistortion).

        Returns
        -------
        bool
            True if the distortion parameters are set, False otherwise.
        """
        return True

    # =================================================================
    # Distortion Model Coefficients
    # =================================================================
    @property
    def parameters_x(self) -> Optional[numpy.ndarray]:
        r"""
        Get  or set the Zernike coefficients for the x coordinate.

        .. warning::

            The value must be a 1D numpy array with the same number of elements as the number of parameters requested by the model.
            Use ``parameters``, ``Nzer`` or ``Nparams`` to change the model !

        Returns
        -------
        Optional[numpy.ndarray]
            The Zernike coefficients for the x coordinate. If no distortion is applied, returns None.
        """
        if self.parameters is None:
            return None
        return self.parameters[0::2]
    
    @parameters_x.setter
    def parameters_x(self, value: numpy.ndarray) -> None:
        if self.parameters is None:
            raise ValueError("No distortion model is defined. Set the parameters first.")
        value = numpy.asarray(value, dtype=Package.get_float_dtype())
        if not value.ndim == 1:
            raise ValueError("The Zernike coefficients for the x coordinate should be a 1D numpy array.")
        if not value.size == self.Nparams // 2:
            raise ValueError(f"The number of Zernike coefficients for the x coordinate should be {self.Nparams // 2}.")
        if not numpy.all(numpy.isfinite(value)):
            raise ValueError("The Zernike coefficients for the x coordinate should be finite numbers.")
        self.parameters[0::2] = value

    @property
    def parameters_y(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the Zernike coefficients for the y coordinate.

        .. warning::

            The value must be a 1D numpy array with the same number of elements as the number of parameters requested by the model.
            Use ``parameters``, ``Nzer`` or ``Nparams`` to change the model !

        Returns
        -------
        Optional[numpy.ndarray]
            The Zernike coefficients for the y coordinate. If no distortion is applied, returns None.
        """
        if self.parameters is None:
            return None
        return self.parameters[1::2]
    
    @parameters_y.setter
    def parameters_y(self, value: numpy.ndarray) -> None:
        if self.parameters is None:
            raise ValueError("No distortion model is defined. Set the parameters first.")
        value = numpy.asarray(value, dtype=Package.get_float_dtype())
        if not value.ndim == 1:
            raise ValueError("The Zernike coefficients for the y coordinate should be a 1D numpy array.")
        if not value.size == self.Nparams // 2:
            raise ValueError(f"The number of Zernike coefficients for the y coordinate should be {self.Nparams // 2}.")
        if not numpy.all(numpy.isfinite(value)):
            raise ValueError("The Zernike coefficients for the y coordinate should be finite numbers.")
        self.parameters[1::2] = value

    def get_index(self, n: Integral, m: Integral, coord: str) -> int:
        r"""
        Get the index of the Zernike coefficient for the given order and azimuthal frequency.

        .. math::

            j = n(n+2) + m + (0 \text{ if } coord = 'x' \text{ else } 1)

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.
        coord : str
            The coordinate ('x' or 'y') for which to get the index.

        Returns
        -------
        int
            The index of the Zernike coefficient in the parameters array.
        """
        if coord not in ['x', 'y']:
            raise ValueError("The coordinate should be 'x' or 'y'.")
        if not isinstance(n, Integral) or not isinstance(m, Integral):
            raise TypeError("The order and azimuthal frequency should be integers.")
        if n < 0 or abs(m) > n or (n - m) % 2 != 0:
            raise ValueError("Invalid order or azimuthal frequency for Zernike polynomial.")
        if self.parameters is None:
            raise ValueError("No distortion model is defined.")
        if n > self.Nzer:
            raise ValueError(f"The order of the Zernike polynomial {n} is greater than the defined order {self.Nzer}.")
        
        index = n * (n + 2) + m
        if coord == 'y':
            index += 1
        return index
        
    def set_Cx(self, n: Integral, m: Integral, value: Number) -> None:
        r"""
        Set the Zernike coefficient for the x coordinate.

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.
        value : float
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, 'x')
        self.parameters[index] = value

    def get_Cx(self, n: Integral, m: Integral) -> Number:
        r"""
        Get the Zernike coefficient for the x coordinate.

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.

        Returns
        -------
        float
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, 'x')
        return self.parameters[index]
        
    def set_Cy(self, n: Integral, m: Integral, value: Number) -> None:
        r"""
        Set the Zernike coefficient for the y coordinate.

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.
        value : float
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, 'y')
        self.parameters[index] = value

    def get_Cy(self, n: Integral, m: Integral) -> Number:
        r"""
        Get the Zernike coefficient for the y coordinate.

        Parameters
        ----------
        n : int
            The order of the Zernike polynomial.
        m : int
            The azimuthal frequency of the Zernike polynomial.

        Returns
        -------
        float
            The value of the Zernike coefficient.
        """
        index = self.get_index(n, m, 'y')
        return self.parameters[index]
    
    # =================================================================
    # Display the distortion model
    # =================================================================
    def __repr__(self) -> str:
        r"""
        Return a string representation of the distortion model.

        Returns
        -------
        str
            The string representation of the distortion model.
        """
        if self.parameters is None:
            return "ZernikeDistortion: No distortion model"
        
        Nzer = self.Nzer
        Nparams = self.Nparams
        parameters_str = f"ZernikeDistortion: {Nparams} parameters (Nzer={Nzer})\n"
        for n in range(Nzer + 1):
            for m in range(-n, n + 1, 2):
                parameters_str += f"  Cx[n={n}, m={m}] = {self.parameters[self.get_index(n, m, 'x')]:.6f}\n"
                parameters_str += f"  Cy[n={n}, m={m}] = {self.parameters[self.get_index(n, m, 'y')]:.6f}\n"
        return parameters_str
    
    # =================================================================
    # Implementation of the transformation methods
    # =================================================================
    def _transform(self, normalized_points: numpy.ndarray, *, dx: bool = False, dp: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the ``normalized_points`` to the ``distorted_points``.

        Lets consider ``normalized_points`` in the camera normalized coordinate system :math:`\vec{x}_n = (x_n, y_n)`, the corresponding ``distorted_points`` in the camera normalized coordinate system are given :math:`\vec{x}_d` can be obtained by :

        .. math::

            x_{d} = x_{n} + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} Z_{nm}(\rho, \theta)

        .. math::

            y_{d} = y_{n} + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{y}_{n,m} Z_{nm}(\rho, \theta)

        The jacobians with respect to the distortion parameters is an array with shape (Npoints, 2, Nparams), where the last dimension represents the parameters in the order of the class attributes (Cx[0,0], Cy[0,0], Cx[1,-1], Cy[1,-1], Cx[1,1], Cy[1,1], ...).
        The jacobian with respect to the normalized points is an array with shape (Npoints, 2, 2).

        The derivative of the distorted points with respect to the normalized points is given by:

        .. math::

            \frac{\partial x_{d}}{\partial x_{n}} = 1 + \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} \frac{\partial Z_{nm}}{\partial x_{n}}

        .. math::

            \frac{\partial x_{d}}{\partial y_{n}} = \sum_{n=0}^{N_{zer}} \sum_{m=-n}^{n} C^{x}_{n,m} \frac{\partial Z_{nm}}{\partial y_{n}}

        Where:

        .. math::

            \frac{\partial Z_{nm}}{\partial x_{n}} = \frac{\partial Z_{nm}}{\partial \rho} \cdot \frac{\partial \rho}{\partial x_{n}} + \frac{\partial Z_{nm}}{\partial \theta} \cdot \frac{\partial \theta}{\partial x_{n}}

        .. math::

            \frac{\partial Z_{nm}}{\partial y_{n}} = \frac{\partial Z_{nm}}{\partial \rho} \cdot \frac{\partial \rho}{\partial y_{n}} + \frac{\partial Z_{nm}}{\partial \theta} \cdot \frac{\partial \theta}{\partial y_{n}}

        .. seealso::

            Package ``pyzernike`` (https://github.com/Artezaru/pyzernike) for the implementation of the Zernike polynomials and their derivatives.

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
        Nparams = self.Nparams

        # Prepare the output jacobian arrays
        if dx:
            jacobian_dx = numpy.tile(numpy.eye(2, dtype=Package.get_float_dtype()), (x_N.size, 1, 1))  # shape (Npoints, 2, 2)
        else:
            jacobian_dx = None

        if dp:
            jacobian_dp = numpy.empty((x_N.size, 2, Nparams), dtype=Package.get_float_dtype())
        else:
            jacobian_dp = None

        # If no distortion model is defined, return the normalized points
        if self.parameters is None:
            return normalized_points.copy(), jacobian_dx, jacobian_dp
        
        # Initialize the distorted points
        x_D = x_N.copy()
        y_D = y_N.copy()

        # Construct the derivatives to compute the Jacobian
        if dx:
            list_dx = [0, 1, 0]
            list_dy = [0, 0, 1]
        else:
            list_dx = [0]
            list_dy = [0]
                
        # Construct the zernike polynomial values and their derivatives
        zernike_results = pyzernike.xy_zernike_polynomial_up_to_order(x_N, y_N, order=self.Nzer, Rx=self.radius_x, Ry=self.radius_y, x0=self.center[0], y0=self.center[1], x_derivative=list_dx, y_derivative=list_dy)

        # Initialize the distorted points and jacobians
        for n in range(self.Nzer + 1):
            for m in range(-n, n + 1, 2):
                zernike_index = pyzernike.zernike_order_to_index(n=[n], m=[m])[0]
                # Get the Zernike polynomial value
                Z_nm = zernike_results[0][zernike_index]
                
                # Get the dérivatives of the Zernike polynomial if requested
                if dx:
                    Z_nm_dx = zernike_results[1][zernike_index]
                    Z_nm_dy = zernike_results[2][zernike_index]

                # Extract the coefficients for the x and y coordinates
                index_x = self.get_index(n, m, 'x')
                Cx = self.parameters[index_x]
                index_y = self.get_index(n, m, 'y')
                Cy = self.parameters[index_y]

                # Update the distorted points
                x_D += Cx * Z_nm
                y_D += Cy * Z_nm

                if dx:
                    # Compute the Jacobian with respect to the normalized points
                    jacobian_dx[:, 0, 0] += Cx * Z_nm_dx
                    jacobian_dx[:, 0, 1] += Cx * Z_nm_dy
                    jacobian_dx[:, 1, 0] += Cy * Z_nm_dx
                    jacobian_dx[:, 1, 1] += Cy * Z_nm_dy

                if dp:
                    # Compute the Jacobian with respect to the distortion parameters
                    jacobian_dp[:, 0, index_x] = Z_nm
                    jacobian_dp[:, 1, index_y] = Z_nm

        # Convert the distorted points back to the original coordinates
        distorted_points = numpy.empty_like(normalized_points, dtype=Package.get_float_dtype())
        distorted_points[:, 0] = x_D
        distorted_points[:, 1] = y_D

        # Return the distorted points and the jacobians if requested
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