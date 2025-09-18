# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Type

from .core.transform import Transform

def read_transform(file_path, cls: Type[Transform]) -> Transform:
    """
    Reads a json files containing a transformation.
    The JSON file must contain the following keys: 'parameters', 'constants', and optional 'type'.

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

    if not "parameters" in transform_data:
        raise ValueError("Missing 'parameters' key in transform data.")

    if not "constants" in transform_data:
        raise ValueError("Missing 'constants' key in transform data.")

    if not "type" in transform_data:
        print("[pycvcam] Missing 'type' key in transform data. Loading without type verification.")

    if "type" in transform_data and not transform_data["type"] == cls.__name__:
        raise ValueError(f"Transform type mismatch, expected {cls.__name__} but got {transform_data['type']}")

    # Create an instance of the Transform subclass
    transform = cls(parameters=transform_data.get('parameters', None),
                    constants=transform_data.get('constants', None))

    return transform
