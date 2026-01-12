.. currentmodule:: pycvcam.core

pycvcam.core.Intrinsic
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

.. note::

    See the main :class:`Transform` documentation for more details on how to use transformations.

.. seealso::

    The implemented intrinsic models are:

    - :class:`NoIntrinsic` (does not apply any intrinsic transformation)
    - :class:`Cv2Intrinsic` (OpenCV pinhole camera model)
    - :class:`SkewIntrinsic` (pinhole camera model with skew parameter)

Intrinsic Class
---------------------

.. autoclass:: Intrinsic


Adding Public Methods of Intrinsic subclasses
----------------------------------------------

.. autosummary::
   :toctree: ../generated/

   Intrinsic.scale
   Intrinsic.unscale

