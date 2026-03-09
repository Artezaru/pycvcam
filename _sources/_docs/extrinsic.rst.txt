.. currentmodule:: pycvcam.core

pycvcam.core.Extrinsic
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

.. note::

    See the main :class:`pycvcam.core.Transform` documentation for more details on how to use transformations.

.. seealso::

    The implemented extrinsic models are:

    - :class:`pycvcam.NoExtrinsic` (does not apply any extrinsic transformation)
    - :class:`pycvcam.Cv2Extrinsic` (rotation vector and translation vector representation)
    - :class:`pycvcam.OrthographicExtrinsic` (orthographic projection model)

Extrinsic Class
---------------------

.. autoclass:: Extrinsic


Adding Public Methods of Extrinsic subclasses
----------------------------------------------

.. autosummary::
   :toctree: ../_autosummary/

   Extrinsic.project
   Extrinsic.unproject
   Extrinsic.compute_rays

Developing Extrinsic Subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a new extrinsic transformation model, subclass the :class:`pycvcam.core.Extrinsic` class and implement the following methods (in addition to the methods required by the :class:`pycvcam.core.Transform` class):

.. autosummary::
   :toctree: ../_autosummary/

   Extrinsic._compute_rays