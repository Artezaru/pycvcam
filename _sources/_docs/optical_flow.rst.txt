.. currentmodule:: pycvcam

Optical Flow Operations
==========================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

The package includes functions to compute the optical flow between two images based on ``OpenCV`` ``DIS`` (Dense Inverse Search) algorithm. The optical flow can be computed for a specific channel of the images and for a specific region of interest. 
The computed optical flow can be visualized using quiver plots or color-coded flow maps. 

Compute Optical Flow
---------------------

.. autofunction:: compute_optical_flow

Display Optical Flow
-----------------------

.. autofunction:: display_optical_flow

Examples
---------

See a complete example in the gallery: :ref:`sphx_glr__gallery_optical_flow.py`.

