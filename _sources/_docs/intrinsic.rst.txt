.. currentmodule:: pycvcam.core

pycvcam.core.Intrinsic
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

.. note::

    See the main :class:`pycvcam.core.Transform` documentation for more details on how to use transformations.

.. seealso::

    The implemented intrinsic models are:

    - :class:`pycvcam.NoIntrinsic` (does not apply any intrinsic transformation)
    - :class:`pycvcam.Cv2Intrinsic` (OpenCV pinhole camera model)
    - :class:`pycvcam.SkewIntrinsic` (pinhole camera model with skew parameter)

Intrinsic Class
---------------------

.. autoclass:: Intrinsic


Adding Public Methods of Intrinsic subclasses
----------------------------------------------

.. autosummary::
   :toctree: ../_autosummary/

   Intrinsic.scale
   Intrinsic.unscale

Implemented Intrinsic Models
----------------------------------------------

See the documentation :doc:`Intrinsic Models <implemented_intrinsic>` for the implemented intrinsic models in the package.

