.. currentmodule:: pycvcam

Optimize input points of transformations
=============================================

.. contents:: Table of Contents
   :local:
   :depth: 1

For each optimization process, the following functions are available:

- ``gn``: Gauss-Newton optimization solving directly :math:`\mathbf{J}^T \mathbf{J} \Delta = -\mathbf{J}^T \mathbf{r}` without damping, scaling or boundary constraints.

Optimize the input points of a unique transformation
-----------------------------------------------------

Lets consider a :class:`pycvcam.Transform` object and a set of input and output points. 
The following functions optimize the input points of the transformation to minimize the 
reprojection error between the input and output points.

.. autosummary::
   :toctree: ../_autosummary/

   optimize_input_points_gn
   optimize_input_points


Optimize the input points of chains of transformations
--------------------------------------------------------

Lets :math:`(T_0, T_1, ..., T_{N_T-1})` be a tuple of :math:`N_T` :class:`Transform` objects, and :math:`(C_0, C_1, ..., C_{N_C-1})` be a tuple of :math:`N_C` chains of transformations.

A chain :math:`C_i` is defined as a tuple of indices corresponding to thetransformations in the chain. For example:

.. code-block:: console

   C_0 = (1, 4, 8) -----> C_0(X) = T_8 ∘ T_4 ∘ T_1(X)

The objective is to optimize the input points based on the output points of all the chains.

.. autosummary::
   :toctree: ../_autosummary/

   optimize_chains_input_points_gn
