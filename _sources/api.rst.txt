API Reference
==============

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


This section contains a detailed description of the functions, modules, and objects included in ``pycvcam``. The reference describes how the methods work and which parameters can be used. It assumes that you have an understanding of the key concepts.
The API is organized into several sections, each corresponding to a specific aspect of the package.

Base Classes Transformations
------------------------------------------------

Abstract base classes and data classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package provides a set of transformations that can be applied to process the transformation from the ``world_points`` to the ``image_points``.
The structure of objects is given by the abstract classes stored in the ``pycvcam.core`` module.
The following base classes can be used to define new transformations by inheriting from them and implementing the required methods:

- :doc:`Transform <_docs/transform>`: The base class for transformations, which defines the interface for applying transformations to points.
- :doc:`TransformResult <_docs/transform_result>`: The data class for storing the results of transformations, which defines the structure for storing the transformed points and respected jacobians.
- :doc:`TransformComposition <_docs/transform_composition>`: The class for composing multiple transformations together, which defines the structure for storing and applying a sequence of transformations.
- :doc:`Intrinsic <_docs/intrinsic>`: The base class for intrinsic transformations, which defines the interface for applying intrinsic transformations to points.
- :doc:`Distortion <_docs/distortion>`: The base class for distortion transformations, which defines the interface for applying distortion transformations to points.
- :doc:`Extrinsic <_docs/extrinsic>`: The base class for extrinsic transformations, which defines the interface for applying extrinsic transformations to points.
- :doc:`Rays <_docs/rays>`: The data class for storing rays, which defines the structure for storing the origin and direction of rays in 3D space.

.. note::

   The each transformation model implemented in the package: 

   - ``parameters`` refers to the parameters of the transformation that are optimisable and the jacobians are computed with respect to these parameters.
   - ``constants`` refers to the parameters of the transformation that are not optimisable and the jacobians are not computed with respect to these parameters.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./_docs/transform.rst
   ./_docs/transform_result.rst
   ./_docs/transform_composition.rst
   ./_docs/intrinsic.rst
   ./_docs/distortion.rst
   ./_docs/extrinsic.rst
   ./_docs/rays.rst

Implemented transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. grid:: 3

    .. grid-item-card::
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Implemented Extrinsic
      ^^^

      This section provide the implemented ``Extrinsic`` transformation models in the package from
      the ``world_points`` to the ``normalized_points``.

      +++

      .. button-ref:: _docs/implemented_extrinsic
         :expand:
         :color: secondary
         :click-parent:

         To the extrinsic models reference guide

    .. grid-item-card::
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Implemented Distortion
      ^^^

      This section provide the implemented ``Distortion`` transformation models in the package from
      the ``normalized_points`` to the ``distorted_points``.

      +++

      .. button-ref:: _docs/implemented_distortion
         :expand:
         :color: secondary
         :click-parent:

         To the distortion models reference guide

    .. grid-item-card:: 
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Implemented Intrinsic
      ^^^

      This section provide the implemented ``Intrinsic`` transformation models in the package from
      the ``distorted_points`` to the ``image_points``.

      +++

      .. button-ref:: _docs/implemented_intrinsic
         :expand:
         :color: secondary
         :click-parent:

         To the intrinsic models reference guide

.. toctree::
   :maxdepth: 1
   :hidden:

   ./_docs/implemented_extrinsic.rst
   ./_docs/implemented_distortion.rst
   ./_docs/implemented_intrinsic.rst


Write and read transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To save and load transformations, the package provides the following functions:

- :doc:`Write (write_transform) <_docs/write_transform>`: A function to save a transformation object to a Json file.
- :doc:`Read (read_transform) <_docs/read_transform>`: A function to load a transformation object from a Json file.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./_docs/write_transform.rst
   ./_docs/read_transform.rst


Process basic transformations with ``pycvcam``
--------------------------------------------------------

The package ``pycvcam`` provides a set of transformation processes that can be used to apply the transformations to points or images.
The implemented processes are the following:

- :doc:`Distort and Undistort Images <_docs/distorting_images>`: A process to distort and undistort an image using a distortion model and an intrinsic model.
- :doc:`Distort and Undistort Points <_docs/distorting_points>`: A process to distort and undistort points using a distortion model and an intrinsic model.
- :doc:`Project Points <_docs/project_points>`: A process to project points from the world coordinate system to the image coordinate system using an extrinsic model, a distortion model, and an intrinsic model.
- :doc:`Compute Rays <_docs/compute_rays>`: A process to compute rays from the camera center to the world points using an extrinsic model and a distortion model.
- :doc:`Compute Optical Flow <_docs/optical_flow>`: A process to compute the optical flow between two images using the DIS method of OpenCV.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./_docs/distorting_images.rst
   ./_docs/distorting_points.rst
   ./_docs/project_points.rst
   ./_docs/compute_rays.rst
   ./_docs/optical_flow.rst


See the section :doc:`Usage <usage>` for more details and examples on how to use these processes.


Optimisation processes
----------------------

The package provides a set of optimisation processes that can be used to estimate the parameters of the transformations.
The optimisations are located in the ``pycvcam.optimize`` module.

- :doc:`Optimize Parameters Least Squares <_docs/optimize_parameters_least_squares>`: A process to optimize the parameters of a transformation or a camera model using a Scipy least squares optimization with bounds and scaling.
- :doc:`Optimize Input Points <_docs/optimize_input_points>`: A process to optimize the input points of a transformation using a least squares optimization.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./_docs/optimize_parameters_least_squares.rst
   ./_docs/optimize_input_points.rst
   



