.. currentmodule:: pycvcam

Optimize parameters of transformations
=============================================

.. contents:: Table of Contents
   :local:
   :depth: 1

For each optimization process, the following functions are available:

- ``gn``: Gauss-Newton optimization solving directly :math:`\mathbf{J}^T \mathbf{J} \Delta = -\mathbf{J}^T \mathbf{r}` without damping, scaling or boundary constraints.
- ``trf``: Trust Region Reflective optimization using ``scipy.optimize.least_squares`` with scaling and boundary constraints.
- ``lm``: Levenberg-Marquardt optimization using ``scipy.optimize.least_squares`` without scaling and boundary constraints.

Optimize the parameters of a unique transformation
--------------------------------------------------

Lets consider a :class:`pycvcam.Transform` object and a set of input and output points. 
The following functions optimize the parameters of the transformation to minimize the reprojection error between the input and output points.

.. autosummary::
   :toctree: ../_autosummary/

   optimize_parameters_gn
   optimize_parameters_trf
   optimize_parameters_lm

Optimize the parameters of a camera transformation
--------------------------------------------------

Lets consider a :class:`pycvcam.Intrinsic` object, a :class:`pycvcam.Distortion` object, a :class:`pycvcam.Extrinsic` object and a set of input and output points. 
The following functions optimize the parameters of the camera transformation to minimize the reprojection error between the input and output points.

.. autosummary::
   :toctree: ../_autosummary/

   optimize_camera_gn
   optimize_camera_trf
   optimize_camera_lm


Optimize the parameters of chains of transformations
-----------------------------------------------------

Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a tuple of :math:`N_T` :class:`Transform` objects, and :math:`(C_0, C_1, ..., C_{N_C-1})` be a tuple of :math:`N_C` chains of transformations.

A chain :math:`C_i` is defined as a tuple of indices corresponding to thetransformations in the chain. For example:

.. code-block:: console

   C_0 = (1, 4, 8) -----> C_0(X) = T_8 ∘ T_4 ∘ T_1(X)

The objective is to optimize the parameters of the transformations to minimize the reprojection error between the input and output points of all the chains.

.. autosummary::
   :toctree: ../_autosummary/

   optimize_chains_gn
   optimize_chains_trf
   optimize_chains_lm

