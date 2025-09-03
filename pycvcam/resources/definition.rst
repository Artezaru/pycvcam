A computer vision camera is modeled by three main components:

1. **Extrinsic**: The transformation from the world coordinate system to the normalized camera coordinate system (``world_points`` to ``normalized_points``)
2. **Distortion**: The transformation from the normalized camera coordinate system to the distorted camera coordinate system (``normalized_points`` to ``distorted_points``)
3. **Intrinsic**: The transformation from the distorted camera coordinate system to the image coordinate system (``distorted_points`` to ``image_points``)

As described in the figure below, the package ``pycvcam`` uses the following notation:

- ``world_points``: The 3-D points :math:`\vec{X}_w` with shape (...,3) expressed in the world coordinate system :math:`(\vec{E}_x, \vec{E}_y, \vec{E}_z)`.
- ``normalized_points``: The 2-D points :math:`\vec{x}_n` with shape (...,2) expressed in the normalized camera coordinate system :math:`(\vec{I}, \vec{J})` with a unit distance along the optical axis :math:`(\vec{K})`.
- ``distorted_points``: The distorted 2-D points :math:`\vec{x}_d` with shape (...,2) expressed in the normalized camera coordinate system :math:`(\vec{I}, \vec{J})` with a unit distance along the optical axis :math:`(\vec{K})`.
- ``image_points``: The 2-D points :math:`\vec{x}_i` with shape (...,2) expressed in the image coordinate system :math:`(\vec{e}_x, \vec{e}_y)` in the sensor plane.
- ``pixel_points``: The 2-D points :math:`\vec{x}_p` with shape (...,2) expressed in the pixel coordinate system :math:`(u, v)` in the matrix of pixels.

.. figure:: ../../pycvcam/resources/definition.png
   :align: center
   :width: 60%
   :alt: Definition of quantities in ``pycvcam``.

To convert the ``image_points`` to the ``pixel_points``, a simple switch of coordinate system can be performed:

.. code-block:: python

    import numpy
    import cv2

    image = cv2.imread('image.jpg')
    image_height, image_width = image.shape[:2]

    pixel_points = numpy.indices((image_height, image_width), dtype=numpy.float64) # shape (2, H, W)
    pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]

    image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

To model a camera without distortion (or intrinsic respectively), simply use an identity transformation.
