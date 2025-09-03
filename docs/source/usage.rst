Usage
==============

.. include:: ../../pycvcam/resources/definition.rst

Examples
--------

The examples are provided in the `examples` directory of the package.

.. literalinclude:: ../../examples/project_points.py
   :language: python
   :caption: Example of Projecting Points

.. literalinclude:: ../../examples/undistort_points.py
   :language: python
   :caption: Example of Undistorting Points

.. literalinclude:: ../../examples/distort_image.py
   :language: python
   :caption: Example of Distorting an Image

.. literalinclude:: ../../examples/undistort_image.py
   :language: python
   :caption: Example of Undistorting an Image

.. literalinclude:: ../../examples/compute_rays.py
   :language: python
   :caption: Example of Computing Rays


Visualizers
------------

The package provides a set of visualizer UIs that can be used to interact with the transformations.
The visualizers are located in the ``pycvcam.visualizers`` module.

zernike_distortion
~~~~~~~~~~~~~~~~~~~~~

Run the following command to display the Zernike Distortion parameters.

.. code-block::

   pycvcam-gui -zernike

.. note::

   For now, the Zernike Distortion visualizer is the only one available. New work is being done to add more visualizers in the future and make them more user-friendly.

.. warning::

   PyQt5 file dialog crash when launched from integrated terminal and when json files are used.
   Run the command line in a standalone terminal instead to avoid this issue.
