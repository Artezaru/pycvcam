API Reference
==============

To set the type of the arrays in the package use the :class:`pycvcam.core.Package` class.

Transformations
----------------

The package provides a set of transformations that can be applied to process the transformation from the ``world_points`` to the ``image_points``.
The structure of objects is given by the abstract classes stored in the ``pycvcam.core`` module.

.. toctree::
   :maxdepth: 1
   :caption: pycvcam.core:

   ./api_doc/package.rst
   ./api_doc/transform.rst
   ./api_doc/transform_result.rst
   ./api_doc/intrinsic.rst
   ./api_doc/distortion.rst
   ./api_doc/extrinsic.rst
   ./api_doc/rays.rst

Some default ``Extrinsic``, ``Intrinsic``, and ``Distortion`` objects are provided in the package.

.. toctree::
   :maxdepth: 1
   :caption: Extrinsic Models:

   ./api_doc/cv2_extrinsic.rst
   ./api_doc/no_extrinsic.rst

.. toctree::
   :maxdepth: 1
   :caption: Intrinsic Models:

   ./api_doc/cv2_intrinsic.rst
   ./api_doc/skew_intrinsic.rst
   ./api_doc/no_intrinsic.rst

.. toctree::
   :maxdepth: 1
   :caption: Distortion Models:

   ./api_doc/cv2_distortion.rst
   ./api_doc/zernike_distortion.rst
   ./api_doc/no_distortion.rst



Transformation processes
-------------------------

The package provides a set of transformation processes that can be used to apply the transformations to the points.

.. toctree::
   :maxdepth: 1
   :caption: Transformation Processes:

   ./api_doc/undistort_image.rst
   ./api_doc/undistort_points.rst
   ./api_doc/project_points.rst
   ./api_doc/compute_rays.rst
   ./api_doc/distort_image.rst
   ./api_doc/read_transform.rst
   ./api_doc/write_transform.rst


Optimisation processes
----------------------

The package provides a set of optimisation processes that can be used to estimate the parameters of the transformations.
The optimisations are located in the ``pycvcam.optimize`` module.

.. toctree::
   :maxdepth: 1
   :caption: Optimisation Processes:

   ./api_doc/optimize_parameters.rst
   ./api_doc/optimize_input_points.rst



