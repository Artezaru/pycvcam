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

import numpy

class Package(object):
    r"""
    Package class for managing data types used in the pycvcam package.
    This class allows setting and getting the default data types for floating-point,
    integer, and unsigned integer values. It also provides methods to reset these
    data types to their default values.

    .. code-block:: python

        import numpy
        from pycvcam.core import Package

        # Set custom data types
        Package.set_float_dtype(numpy.float32)
        Package.set_int_dtype(numpy.int64)
        Package.set_uint_dtype(numpy.uint64)

    Default data types are:

        - Floating-point: numpy.float64
        - Integer: numpy.int32
        - Unsigned Integer: numpy.uint8

    """

    __float_dtype = numpy.float64
    __int_dtype = numpy.int32
    __uint_dtype = numpy.uint8

    def __new__(cls, *args, **kwargs):
        raise TypeError("Package is a static class and cannot be instantiated.")

    @classmethod
    def set_float_dtype(cls, dtype):
        if not isinstance(dtype, numpy.dtype):
            try:
                dtype = numpy.dtype(dtype)
            except TypeError:
                raise TypeError("dtype must be convertible to a numpy.dtype")
        if dtype.kind != 'f':
            raise ValueError("dtype must be a floating-point type")
        cls.__float_dtype = dtype

    @classmethod
    def get_float_dtype(cls):
        return cls.__float_dtype
    
    @classmethod
    def set_int_dtype(cls, dtype):
        if not isinstance(dtype, numpy.dtype):
            try:
                dtype = numpy.dtype(dtype)
            except TypeError:
                raise TypeError("dtype must be convertible to a numpy.dtype")
        if dtype.kind != 'i':
            raise ValueError("dtype must be an integer type")
        cls.__int_dtype = dtype

    @classmethod
    def get_int_dtype(cls):
        return cls.__int_dtype  
    
    @classmethod
    def set_uint_dtype(cls, dtype):
        if not isinstance(dtype, numpy.dtype):
            try:
                dtype = numpy.dtype(dtype)
            except TypeError:
                raise TypeError("dtype must be convertible to a numpy.dtype")
        if dtype.kind != 'u':
            raise ValueError("dtype must be an unsigned integer type")
        cls.__uint_dtype = dtype

    @classmethod
    def get_uint_dtype(cls):
        return cls.__uint_dtype
    
    @classmethod
    def reset_dtypes(cls):
        cls.__float_dtype = numpy.float64
        cls.__int_dtype = numpy.int32
        cls.__uint_dtype = numpy.uint32
