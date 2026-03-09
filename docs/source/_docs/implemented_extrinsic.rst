Implemented Extrinsic Transformations
=====================================

The package ``pycvcam`` provides a set of implemented extrinsic transformation models 
that can be used to transform 3D ``world_points`` to 2D ``normalized_points``. 

- :doc:`NoExtrinsic <no_extrinsic>`: Identity transformation that does not apply any extrinsic transformation (Simply set third coordinate :math:`z` to 1 and keep the "x,y" coordinates of the ``world_points``).
- :doc:`Cv2Extrinsic <cv2_extrinsic>`: Like OpenCV's, apply a rigid transformation :math:`[R|t]` to the ``world_points`` and normalize by the third coordinate.
- :doc:`OrthographicExtrinsic <orthographic_extrinsic>`: Apply a rigid transformation :math:`[R|t]` to the ``world_points`` and ignore the third coordinate.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./no_extrinsic.rst
   ./cv2_extrinsic.rst
   ./orthographic_extrinsic.rst