import json
import os
from typing import Type

from .core.transform import Transform

def read_transform(file_path, cls: Type[Transform]) -> Transform:
    """
    Reads a json files containing a transformation.

    .. code-block:: python

       from pycvcam import Cv2Distortion
       from pycvcam import read_transform

       transform = read_transform("transform.json", Cv2Distortion)

    .. seealso::

       :func:`pycvcam.write_transform` for the corresponding write function and more information on the JSON format.

    Parameters
    ----------
    file_path: str
        The path to the JSON file to read from.
    cls: Type[Transform]
        The class of the Transform object to create.
    """
    # Type Checking
    if not issubclass(cls, Transform):
        raise TypeError("Expected a Transform subclass.")

    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError("Transform file not found.")

    # Read the JSON file
    with open(file_path, 'r') as f:
        transform_data = json.load(f)

    if not transform_data["type"] == cls.__name__:
        raise ValueError(f"Transform type mismatch, expected {cls.__name__} but got {transform_data['type']}")

    # Create an instance of the Transform subclass
    transform = cls(parameters=transform_data.get('parameters', None),
                    constants=transform_data.get('constants', None))

    return transform
