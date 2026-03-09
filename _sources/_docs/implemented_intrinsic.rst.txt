Implemented Intrinsic Transformations
=====================================

The package ``pycvcam`` provides a set of implemented intrinsic transformation models 
that can be used to transform 2D ``distorted_points`` to 2D ``image_points``. 

- :doc:`NoIntrinsic <no_intrinsic>`: Identity transformation that does not apply any intrinsic transformation (Simply keep the "x,y" coordinates of the ``distorted_points``).
- :doc:`Cv2Intrinsic <cv2_intrinsic>`: Like OpenCV's, apply a linear transformation to the ``distorted_points`` using the intrinsic matrix :math:`K` and the distortion center :math:`(c_x, c_y)`.
- :doc:`SkewIntrinsic <skew_intrinsic>`: Apply a linear transformation to the ``distorted_points`` using the intrinsic matrix :math:`K` and the distortion center :math:`(c_x, c_y)`, with an additional skew parameter.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./no_intrinsic.rst
   ./cv2_intrinsic.rst
   ./skew_intrinsic.rst