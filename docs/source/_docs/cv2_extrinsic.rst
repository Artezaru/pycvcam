.. currentmodule:: pycvcam

pycvcam.Cv2Extrinsic
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

Cv2Extrinsic Class
---------------------

.. autoclass:: Cv2Extrinsic


Instantiate a Cv2Extrinsic object
-----------------------------------

The :class:`pycvcam.Cv2Extrinsic` class can be instantiated using :

- a rotation and translation vector (``rvec``, ``tvec``).
- a :math:`4 \times 4` transformation matrix.
- a frame of reference associated with the extrinsic parameters.

.. autosummary::
   :toctree: ../_autosummary/

   Cv2Extrinsic.from_frame
   Cv2Extrinsic.from_rt
   Cv2Extrinsic.from_tmatrix

Accessing the parameters of Cv2Extrinsic objects
-------------------------------------------------

The ``parameters`` and ``constants`` properties can be accessing using :class:`pycvcam.core.Transform` methods.
Some additional convenience methods are provided to access commonly used parameters of the Cv2Extrinsic model:

.. seealso::

    - :meth:`pycvcam.core.Transform.parameters`
    - :meth:`pycvcam.core.Transform.constants`

.. autosummary::
   :toctree: ../_autosummary/

   Cv2Extrinsic.frame
   Cv2Extrinsic.rotation_vector
   Cv2Extrinsic.transformation_matrix
   Cv2Extrinsic.translation_vector


Performing projections with Cv2Extrinsic objects
-------------------------------------------------

The ``transform`` and ``inverse_transform`` methods can be used to perform projections and unprojections using the Cv2Extrinsic model (as described in the :class:`pycvcam.core.Transform` documentation).

.. seealso::

    - :meth:`pycvcam.core.Transform.transform`
    - :meth:`pycvcam.core.Transform.inverse_transform`

The implementation of theses transformations and more details on the options available can be found in the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   Cv2Extrinsic._transform
   Cv2Extrinsic._inverse_transform
   Cv2Extrinsic._compute_rays


Examples
--------
Create an extrinsic object with a rotation vector and a translation vector:

.. code-block:: python

   import numpy
   from pycvcam import Cv2Extrinsic

   rvec = numpy.array([0.1, 0.2, 0.3])
   tvec = numpy.array([0.5, 0.5, 0.5])

   extrinsic = Cv2Extrinsic.from_rt(rvec, tvec)

Then you can use the extrinsic object to transform ``world_points`` to ``normalized_points``:

.. code-block:: python

   world_points = numpy.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9],
                              [10, 11, 12]]) # shape (n_points, 3)

   result = extrinsic.transform(world_points)
   normalized_points = result.normalized_points # shape (n_points, 2)
   print(normalized_points)

You can also access to the jacobian of the extrinsic transformation:

.. code-block:: python

   result = extrinsic.transform(world_points, dx=True, dp=True)
   normalized_points_dx = result.jacobian_dx  # Shape (n_points, 2, 3)
   normalized_points_dp = result.jacobian_dp  # Shape (n_points, 2, 6)
   print(normalized_points_dx)
   print(normalized_points_dp)

The inverse transformation can be computed using the `inverse_transform` method:
By default, the depth is assumed to be 1.0 for all points, but you can provide a specific depth for each point with shape (...,).

.. code-block:: python

   depth = numpy.array([1.0, 2.0, 3.0, 4.0])  # Example depth values for each point

   inverse_result = extrinsic.inverse_transform(normalized_points, dx=True, dp=True, depth=depth)
   world_points = inverse_result.world_points  # Shape (n_points, 3)
   print(world_points)

.. note::

   The jacobian with respect to the depth is not computed.

.. seealso::

   For more information about the transformation process, see:

   - :meth:`pycvcam.Cv2Extrinsic._transform` to transform the ``world_points`` to ``normalized_points``.
   - :meth:`pycvcam.Cv2Extrinsic._inverse_transform` to transform the ``normalized_points`` back to ``world_points``.