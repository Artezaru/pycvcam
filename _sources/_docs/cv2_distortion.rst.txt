.. currentmodule:: pycvcam

pycvcam.Cv2Distortion
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

Cv2Distortion Class
---------------------

.. autoclass:: Cv2Distortion


Accessing the parameters of Cv2Distortion objects
-------------------------------------------------

The ``parameters`` and ``constants`` properties can be accessing using :class:`pycvcam.core.Transform` methods.
Some additional convenience methods are provided to access commonly used parameters of the Cv2Distortion model:

.. seealso::

    - :meth:`pycvcam.core.Transform.parameters`
    - :meth:`pycvcam.core.Transform.constants`

.. autosummary::
   :toctree: ../_autosummary/

   Cv2Distortion.k1
   Cv2Distortion.k2
   Cv2Distortion.k3
   Cv2Distortion.k4
   Cv2Distortion.k5
   Cv2Distortion.k6
   Cv2Distortion.p1
   Cv2Distortion.p2
   Cv2Distortion.s1
   Cv2Distortion.s2
   Cv2Distortion.s3
   Cv2Distortion.s4
   Cv2Distortion.tau_x
   Cv2Distortion.tau_y


Performing distortion with Cv2Distortion objects
-------------------------------------------------

The ``transform`` and ``inverse_transform`` methods can be used to perform distortion and undistortion using the Cv2Distortion model (as described in the :class:`pycvcam.core.Transform` documentation).

.. seealso::

    - :meth:`pycvcam.core.Transform.transform`
    - :meth:`pycvcam.core.Transform.inverse_transform`

The implementation of theses transformations and more details on the options available can be found in the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   Cv2Distortion._compute_tilt_matrix
   Cv2Distortion._transform
   Cv2Distortion._transform_opencv
   Cv2Distortion._inverse_transform
   Cv2Distortion._inverse_transform_opencv


Examples
--------
Create an distortion object with a specific number of parameters:

.. code-block:: python

   import numpy
   from pycvcam import Cv2Distortion

   parameters = numpy.array([0.1, 0.01, 0.02, 0.03, 0.001])

   distortion = Cv2Distortion(parameters=parameters)

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

   - :meth:`pycvcam.Cv2Distortion._transform` to transform the ``normalized_points`` to ``distorted_points``.
   - :meth:`pycvcam.Cv2Distortion._inverse_transform` to transform the ``distorted_points`` back to ``normalized_points``.