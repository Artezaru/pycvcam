.. currentmodule:: pycvcam.core

pycvcam.core.Extrinsic
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

.. note::

    See the main :class:`Transform` documentation for more details on how to use transformations.

.. seealso::

    The implemented extrinsic models are:

    - :class:`NoExtrinsic` (does not apply any extrinsic transformation)
    - :class:`Cv2Extrinsic` (rotation vector and translation vector representation)
    - :class:`OrthographicExtrinsic` (orthographic projection model)

Extrinsic Class
---------------------

.. autoclass:: Extrinsic


Adding Public Methods of Extrinsic subclasses
----------------------------------------------

.. autosummary::
   :toctree: ../generated/

   Extrinsic.project
   Extrinsic.unproject
   Extrinsic.compute_rays

Developing Extrinsic Subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a new extrinsic transformation model, subclass the :class:`Extrinsic` class and implement the following methods (in addition to the methods required by the :class:`Transform` class):

.. autosummary::
   :toctree: ../generated/

   Extrinsic._compute_rays