.. currentmodule:: pycvcam.optimize

Optimize Parameters with Least Squares
==================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top


Optimize the parameters of an unique :class:`pycvcam.Transform` object
----------------------------------------------------------------------

.. autofunction:: optimize_parameters_least_squares

Optimize the parameters of a camera model
----------------------------------------------------------------------

.. autofunction:: optimize_camera_least_squares

Optimize the parameters of chains of transforms (multiple cameras with common parameters)
--------------------------------------------------------------------------------------------

.. autofunction:: optimize_chain_parameters_least_squares

Examples
--------

See a complete example in the gallery: :ref:`sphx_glr__gallery_optimize_parameters.py`.