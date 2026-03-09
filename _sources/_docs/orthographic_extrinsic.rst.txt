.. currentmodule:: pycvcam

pycvcam.OrthographicExtrinsic
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

OrthographicExtrinsic Class
----------------------------

.. autoclass:: OrthographicExtrinsic


Instantiate a OrthographicExtrinsic object
------------------------------------------------

The :class:`pycvcam.OrthographicExtrinsic` class can be instantiated using :

- a rotation and translation vector (``rvec``, ``tvec``).
- a :math:`4 \times 4` transformation matrix.
- a frame of reference associated with the extrinsic parameters.

.. autosummary::
   :toctree: ../_autosummary/

   OrthographicExtrinsic.from_frame
   OrthographicExtrinsic.from_rt
   OrthographicExtrinsic.from_tmatrix

Accessing the parameters of OrthographicExtrinsic objects
----------------------------------------------------------

The ``parameters`` and ``constants`` properties can be accessing using :class:`pycvcam.core.Transform` methods.
Some additional convenience methods are provided to access commonly used parameters of the OrthographicExtrinsic model:

.. seealso::

    - :meth:`pycvcam.core.Transform.parameters`
    - :meth:`pycvcam.core.Transform.constants`

.. autosummary::
   :toctree: ../_autosummary/

   OrthographicExtrinsic.frame
   OrthographicExtrinsic.rotation_vector
   OrthographicExtrinsic.transformation_matrix
   OrthographicExtrinsic.translation_vector


Performing projections with OrthographicExtrinsic objects
-------------------------------------------------------------

The ``transform`` and ``inverse_transform`` methods can be used to perform projections and unprojections using the OrthographicExtrinsic model (as described in the :class:`pycvcam.core.Transform` documentation).

.. seealso::

    - :meth:`pycvcam.core.Transform.transform`
    - :meth:`pycvcam.core.Transform.inverse_transform`

The implementation of theses transformations and more details on the options available can be found in the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   OrthographicExtrinsic._transform
   OrthographicExtrinsic._inverse_transform
   OrthographicExtrinsic._compute_rays


Examples
--------
Create an extrinsic object with a rotation vector and a translation vector:

.. code-block:: python

   import numpy
   from pycvcam import OrthographicExtrinsic

   rvec = numpy.array([0.1, 0.2, 0.3])
   tvec = numpy.array([0.5, 0.5, 0.5])

   extrinsic = OrthographicExtrinsic.from_rt(rvec, tvec)

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

   - :meth:`pycvcam.OrthographicExtrinsic._transform` to transform the ``world_points`` to ``normalized_points``.
   - :meth:`pycvcam.OrthographicExtrinsic._inverse_transform` to transform the ``normalized_points`` back to ``world_points``.