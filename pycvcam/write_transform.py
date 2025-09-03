import json
import os
import datetime

from .__version__ import __version__
from .core.transform import Transform

def write_transform(file_path, transform: Transform) -> None:
    """
    Writes the Transform object to a JSON file.

    .. code-block:: python

       from pycvcam import Cv2Distortion
       from pycvcam import write_transform

       distortion = Cv2Distortion(...)
       write_transform("transform.json", distortion)

    The content of the JSON file will be similar to:

    .. code-block:: json

        {
            "type": "Cv2Distortion",
            "version": "1.3.0",
            "date": "2023-01-01T00:00:00",
            "parameters": [0.1, 0.2, 0.3, 0.01, 0.5],
            "constants": null
        }

    .. seealso::

       :func:`pycvcam.read_transform` for the corresponding read function.

    Parameters
    ----------
    file_path: str
        The path to the JSON file to write to.
    transform: Transform
        The Transform object to write.
    """
    # Type Checking
    if not isinstance(transform, Transform):
        raise TypeError("Expected a Transform object.")

    # Create the directory if it doesn't exist
    file_path = os.path.abspath(os.path.expanduser(file_path))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Ensure the file is writable
    if not os.access(os.path.dirname(file_path), os.W_OK):
        raise PermissionError("No write access to the directory.")

    # Create a dict containing the Transform object's data
    transform_data = {}
    transform_data['type'] = type(transform).__name__
    transform_data['version'] = __version__
    transform_data['date'] = datetime.datetime.now().isoformat()
    transform_data['parameters'] = list(transform.parameters) if transform.parameters is not None else None
    transform_data['constants'] = list(transform.constants) if transform.constants is not None else None

    # Write the Transform object to the JSON file
    with open(file_path, 'w') as f:
        json.dump(transform_data, f, indent=4)

    
