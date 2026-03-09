.. currentmodule:: pycvcam

pycvcam.ZernikeDistortion
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

ZernikeDistortion Class
------------------------

.. autoclass:: ZernikeDistortion


Accessing the parameters of ZernikeDistortion objects
-------------------------------------------------------

The ``parameters`` and ``constants`` properties can be accessing using :class:`pycvcam.core.Transform` methods.
Some additional convenience methods are provided to access commonly used parameters of the Cv2Distortion model:

.. seealso::

    - :meth:`pycvcam.core.Transform.parameters`
    - :meth:`pycvcam.core.Transform.constants`

.. autosummary::
   :toctree: ../_autosummary/

   ZernikeDistortion.center
   ZernikeDistortion.center_x
   ZernikeDistortion.center_y
   ZernikeDistortion.n_zer
   ZernikeDistortion.parameters_x
   ZernikeDistortion.parameter_x_names
   ZernikeDistortion.parameters_y
   ZernikeDistortion.parameter_y_names
   ZernikeDistortion.radius
   ZernikeDistortion.radius_x
   ZernikeDistortion.radius_y

Optionnaly, the parameters of the ZernikeDistortion can be accessed using set 
using the convenience methods based on Zernike polynomials indexing:

.. autosummary::
   :toctree: ../_autosummary/

   ZernikeDistortion.get_index
   ZernikeDistortion.get_Cx
   ZernikeDistortion.get_Cy
   ZernikeDistortion.set_Cx
   ZernikeDistortion.set_Cy



Performing distortion with ZernikeDistortion objects
-----------------------------------------------------

The ``transform`` and ``inverse_transform`` methods can be used to perform distortion and undistortion using the Cv2Distortion model (as described in the :class:`pycvcam.core.Transform` documentation).

.. seealso::

    - :meth:`pycvcam.core.Transform.transform`
    - :meth:`pycvcam.core.Transform.inverse_transform`

The implementation of theses transformations and more details on the options available can be found in the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   ZernikeDistortion._transform
   ZernikeDistortion._inverse_transform


Examples
--------
Create an distortion object with a specific order of Zernike polynomials and parameters:

.. code-block:: python

   import numpy
   from pycvcam import ZernikeDistortion

   # Create a distortion object with 6 parameters
   distortion = ZernikeDistortion(numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])) # Model with n_zer=1, -> n_params=6

Then you can use the distortion object to transform ``normalized_points`` to ``distorted_points``:

.. code-block:: python

   normalized_points = numpy.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]) # shape (n_points, 2)

   result = distortion.transform(normalized_points)
   distorted_points = result.distorted_points # Shape (n_points, 2)
   print(distorted_points)

You can also access to the jacobian of the distortion transformation:

.. code-block:: python

   result = distortion.transform(normalized_points, dx=True, dp=True)
   distorted_points_dx = result.jacobian_dx  # Shape (n_points, 2, 2)
   distorted_points_dp = result.jacobian_dp  # Shape (n_points, 2, n_params = 5)
   print(distorted_points_dx) 
   print(distorted_points_dp)

The inverse transformation can be computed using the `inverse_transform` method:

.. code-block:: python

   inverse_result = distortion.inverse_transform(distorted_points, dx=True, dp=True)
   normalized_points = inverse_result.normalized_points  # Shape (n_points, 2)
   print(normalized_points)

.. note::

   The jacobian with respect to the depth is not computed.

.. seealso::

   For more information about the transformation process, see:

   - :meth:`pycvcam.ZernikeDistortion._transform` to transform the ``normalized_points`` to ``distorted_points``.
   - :meth:`pycvcam.ZernikeDistortion._inverse_transform` to transform the ``distorted_points`` back to ``normalized_points``.

If you want to define the Zernike unit disk to encapsulate an image, with a centered distortion and a circular distortion in the image plane, you can use the `constants` parameter:

.. code-block:: python

   import numpy
   import cv2
   from pycvcam import ZernikeDistortion

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