.. currentmodule:: pycvcam.core

pycvcam.core.Distortion
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

.. note::

    See the main :class:`pycvcam.core.Transform` documentation for more details on how to use transformations.

.. seealso::

    The implemented distortion models are:

    - :class:`pycvcam.NoDistortion` (does not apply any distortion transformation)
    - :class:`pycvcam.Cv2Distortion` (OpenCV distortion model)
    - :class:`pycvcam.FisheyeDistortion` (OpenCV fisheye distortion model)
    - :class:`pycvcam.ZernikeDistortion` (Zernike polynomial based distortion model)

Distortion Class
---------------------

.. autoclass:: Distortion

Adding Public Methods of Distortion subclasses
----------------------------------------------

.. autosummary::
   :toctree: ../_autosummary/

   Distortion.distort
   Distortion.undistort