.. currentmodule:: pycvcam

pycvcam.FisheyeDistortion
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

FisheyeDistortion Class
------------------------

.. autoclass:: FisheyeDistortion


Accessing the parameters of FisheyeDistortion objects
-------------------------------------------------------

The ``parameters`` and ``constants`` properties can be accessing using :class:`pycvcam.core.Transform` methods.
Some additional convenience methods are provided to access commonly used parameters of the FisheyeDistortion model:

.. seealso::

    - :meth:`pycvcam.core.Transform.parameters`
    - :meth:`pycvcam.core.Transform.constants`

.. autosummary::
   :toctree: ../_autosummary/

   FisheyeDistortion.get_di
   FisheyeDistortion.set_di


Performing distortion with FisheyeDistortion objects
----------------------------------------------------

The ``transform`` and ``inverse_transform`` methods can be used to perform distortion and undistortion using the FisheyeDistortion model (as described in the :class:`pycvcam.core.Transform` documentation).

.. seealso::

    - :meth:`pycvcam.core.Transform.transform`
    - :meth:`pycvcam.core.Transform.inverse_transform`

The implementation of theses transformations and more details on the options available can be found in the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   FisheyeDistortion._cartesian_to_polar
   FisheyeDistortion._polar_to_cartesian
   FisheyeDistortion._transform
   FisheyeDistortion._inverse_transform


Examples
--------
Create an distortion object with a specific number of parameters:

.. code-block:: python

    import numpy
    from pycvcam import FisheyeDistortion

    parameters = numpy.array([0.1, 0.01, 0.02])

    distortion = FisheyeDistortion(parameters=parameters)

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
    distorted_points_dp = result.jacobian_dp  # Shape (n_points, 2, n_params = 3)
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

    - :meth:`pycvcam.FisheyeDistortion._transform` to transform the ``normalized_points`` to ``distorted_points``.
    - :meth:`pycvcam.FisheyeDistortion._inverse_transform` to transform the ``distorted_points`` back to ``normalized_points``.