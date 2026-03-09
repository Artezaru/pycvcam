Welcome to pycvcam's documentation!
=========================================================================================

Description of the package
--------------------------

Python Computer Vision Cameras transformations and models.

.. include:: ../../pycvcam/resources/definition.rst

These notations are used throughout the documentation.

.. note::

   The package is designed to work with double-precision floating-point numbers to ensure numerical stability in all calculations.
   Therefore, all float arrays are automatically converted to ``numpy.float64`` for computation and all integer arrays are converted to ``numpy.int64`` for computation.
   This means that when you pass arrays to the functions in the package, they will be converted to these data types if they are not already in that format.


Contents
--------

.. grid:: 3

    .. grid-item-card:: 
      :img-top: /_static/_icons/download.png
      :text-align: center

      Installation
      ^^^

      This section describes how to install the package into a Python environment. It includes instructions for installing the package using pip, as well as any necessary dependencies.

      +++

      .. button-ref:: installation
         :expand:
         :color: secondary
         :click-parent:

         To the installation guide

    .. grid-item-card::
      :img-top: /_static/_icons/api.png
      :text-align: center

      API Reference
      ^^^

      The reference guide contains a detailed description of the functions,
      modules, and objects included in ``pycvcam``. The reference describes how the
      methods work and which parameters can be used. It assumes that you have an
      understanding of the key concepts.

      +++ 

      .. button-ref:: api
         :expand:
         :color: secondary
         :click-parent:

         To the API reference

    .. grid-item-card::
      :img-top: /_static/_icons/examples.png
      :text-align: center

      Examples Gallery
      ^^^

      This section contains a collection of examples demonstrating how to use the package for various applications. Each example includes a description of the problem being solved, the code used to solve it, and the resulting output.

      +++

      .. button-ref:: usage
         :expand:
         :color: secondary
         :click-parent:

         To the examples gallery

.. toctree::
   :caption: Contents:
   :hidden:

   installation
   api
   usage


Author
------

The package ``pycvcam`` was created by the following authors:

- Artezaru <artezaru.github@proton.me>

You can access the package and the documentation with the following URL:

- **Git Plateform**: https://github.com/Artezaru/pycvcam.git
- **Online Documentation**: https://Artezaru.github.io/pycvcam

License
-------

Please refer to the [LICENSE] file for the license of the package.
