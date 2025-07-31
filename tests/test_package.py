import pycvcam
from pycvcam.core import Package
from pycvcam import project_points
import numpy

def test_set_float_dtype():
    points = numpy.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=numpy.float64)

    result_points = project_points(points, intrinsic=None, distortion=None, extrinsic=None)
    assert result_points.image_points.dtype == numpy.float64

    Package.set_float_dtype(numpy.float32)

    result_points = project_points(points, intrinsic=None, distortion=None, extrinsic=None)
    assert result_points.image_points.dtype == numpy.float32

    Package.reset_dtypes()


    

