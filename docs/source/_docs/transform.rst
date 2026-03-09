.. currentmodule:: pycvcam.core

pycvcam.core.Transform
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


Transform Class
---------------------

.. autoclass:: Transform


Public Methods of Transform subclasses
----------------------------------------------

Transformation caracteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To have informations on the transformation caracteristics, use the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   Transform.input_dim
   Transform.output_dim
   Transform.parameters
   Transform.n_params
   Transform.parameter_names
   Transform.constants
   Transform.n_constants
   Transform.constant_names
    
Transforming Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To transform points using the transformation from :math:`\mathbb{R}^{\text{input_dim}}` to :math:`\mathbb{R}^{\text{output_dim}}`, use the following method:

.. autosummary::
   :toctree: ../_autosummary/

   Transform.transform
   Transform.inverse_transform


Save and Load Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To save and load transformations, use the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   Transform.to_dict
   Transform.from_dict
   Transform.to_json
   Transform.from_json


Developing Custom Transformations
----------------------------------------------

To create a custom transformation, subclass the :class:`Transform` class and implement the required methods. 
Refer to the documentation of each method for guidance on their implementation.

First edit the class attributes if necessary:

- `_input_dim`: Set this to the dimension of the input space.
- `_output_dim`: Set this to the dimension of the output space.
- `_result_class`: Set this to the class used for the result of the transformation.
- `_inverse_result_class`: Set this to the class used for the result of the inverse transformation

.. autosummary::
   :toctree: ../_autosummary/

   Transform._get_jacobian_shorthands
   Transform._get_transform_aliases
   Transform._get_inverse_transform_aliases
   Transform.is_set
   Transform._return_transform_result
   Transform._return_inverse_transform_result
   Transform._transform
   Transform._inverse_transform