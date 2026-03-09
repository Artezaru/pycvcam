Implemented Distortion Transformations
======================================

The package ``pycvcam`` provides a set of implemented distortion transformation models 
that can be used to transform 2D ``normalized_points`` to 2D ``distorted_points``. 

- :doc:`NoDistortion <no_distortion>`: Identity transformation that does not apply any distortion transformation (Simply keep the "x,y" coordinates of the ``normalized_points``).
- :doc:`Cv2Distortion <cv2_distortion>`: Like OpenCV's, use the radial, tangential, and thin prism distortion model.
- :doc:`ZernikeDistortion <zernike_distortion>`: Use a distortion model based on Zernike polynomials along the two axes.
- :doc:`FisheyeDistortion <fisheye_distortion>`: Like OpenCV's fisheye model, use the fisheye distortion model on the angular coordinates.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./no_distortion.rst
   ./cv2_distortion.rst
   ./zernike_distortion.rst
   ./fisheye_distortion.rst