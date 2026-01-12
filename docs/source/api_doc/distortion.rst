.. currentmodule:: pycvcam.core

pycvcam.core.Distortion
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

.. note::

    See the main :class:`Transform` documentation for more details on how to use transformations.

.. seealso::

    The implemented distortion models are:

    - :class:`NoDistortion` (does not apply any distortion transformation)
    - :class:`Cv2Distortion` (OpenCV distortion model)
    - :class:`FisheyeDistortion` (OpenCV fisheye distortion model)
    - :class:`ZernikeDistortion` (Zernike polynomial based distortion model)


Distortion Class
---------------------

.. autoclass:: Distortion

Adding Public Methods of Distortion subclasses
----------------------------------------------

.. autosummary::
   :toctree: ../generated/

   Distortion.distort
   Distortion.undistort