.. currentmodule:: pycvcam

pycvcam.Cv2Intrinsic
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

Cv2Intrinsic Class
---------------------

.. autoclass:: Cv2Intrinsic


Instantiate a Cv2Intrinsic object
-----------------------------------

The :class:`pycvcam.Cv2Intrinsic` class can be instantiated using :

- a :math:`3 \times 3` intrinsic matrix.

.. autosummary::
   :toctree: ../_autosummary/

   Cv2Intrinsic.from_matrix



Accessing the parameters of Cv2Intrinsic objects
-------------------------------------------------

The ``parameters`` and ``constants`` properties can be accessing using :class:`pycvcam.core.Transform` methods.
Some additional convenience methods are provided to access commonly used parameters of the Cv2Intrinsic model:

.. seealso::

    - :meth:`pycvcam.core.Transform.parameters`
    - :meth:`pycvcam.core.Transform.constants`

.. autosummary::
   :toctree: ../_autosummary/

   Cv2Intrinsic.focal_length_x
   Cv2Intrinsic.focal_length_y
   Cv2Intrinsic.intrinsic_matrix
   Cv2Intrinsic.intrinsic_vector
   Cv2Intrinsic.principal_point_x
   Cv2Intrinsic.principal_point_y


Performing projections with Cv2Intrinsic objects
-------------------------------------------------

The ``transform`` and ``inverse_transform`` methods can be used to perform intrinsic transformations using the Cv2Intrinsic model (as described in the :class:`pycvcam.core.Transform` documentation).

.. seealso::

    - :meth:`pycvcam.core.Transform.transform`
    - :meth:`pycvcam.core.Transform.inverse_transform`

The implementation of theses transformations and more details on the options available can be found in the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   Cv2Intrinsic._transform
   Cv2Intrinsic._inverse_transform



Examples
--------
Create an intrinsic object with a given intrinsic matrix:

.. code-block:: python

   import numpy
   from pycvcam import Cv2Intrinsic

   intrinsic_matrix = numpy.array([[1000, 0, 320],
                                 [0, 1000, 240],
                                 [0, 0, 1]])
   intrinsic = Cv2Intrinsic.from_matrix(intrinsic_matrix)

Then you can use the intrinsic object to transform ``distorted_points`` to ``image_points``:

.. code-block:: python

   distorted_points = numpy.array([[100, 200],
                                 [150, 250],
                                 [200, 300]]) # Shape (n_points, 2)
   result = intrinsic.transform(distorted_points)
   image_points = result.image_points # Shape (n_points, 2)
   print(image_points)

You can also access to the jacobian of the intrinsic transformation:

.. code-block:: python

   result = intrinsic.transform(distorted_points, dx=True, dp=True)
   image_points_dx = result.jacobian_dx  # Jacobian of the image points with respect to the distorted points
   image_points_dp = result.jacobian_dp  # Jacobian of the image points with respect to the intrinsic parameters
   print(image_points_dx)

The inverse transformation can be computed using the `inverse_transform` method:

.. code-block:: python

   inverse_result = intrinsic.inverse_transform(image_points, dx=True, dp=True)
   distorted_points = inverse_result.distorted_points  # Shape (n_points, 2)
   print(distorted_points)

.. seealso::

   For more information about the transformation process, see:

   - :meth:`pycvcam.Cv2Intrinsic._transform` to transform the ``distorted_points`` to ``image_points``.
   - :meth:`pycvcam.Cv2Intrinsic._inverse_transform` to transform the ``image_points`` back to ``distorted_points``.