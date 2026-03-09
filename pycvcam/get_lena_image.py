from importlib import resources
import cv2
import numpy


def get_lena_image() -> numpy.ndarray:
    r"""
    Get the Lena image as a numpy array.
    This function loads the Lena image from the package resources and returns it as a
    numpy array that can be used as a texture in visualizations.

    .. figure:: /_static/textures/lena_image.png
        :alt: Lena image
        :align: center
        :width: 200px

    Returns
    -------
    :class:`numpy.ndarray`
        The Lena image as a numpy array with shape (474, 474) and dtype uint8.

    """
    path = resources.files("pycvcam.resources") / "lena_image.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = numpy.asarray(image, dtype=numpy.uint8)
    return image
