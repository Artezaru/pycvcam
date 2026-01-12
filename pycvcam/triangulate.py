# Copyright 2026 Artezaru
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

from typing import Optional, Tuple
import numpy
import cv2
import scipy

from .core.distortion import Distortion
from .core.intrinsic import Intrinsic
from .core.extrinsic import Extrinsic
from .core.rays import Rays

from .distortion_objects.no_distortion import NoDistortion
from .intrinsic_objects.no_intrinsic import NoIntrinsic
from .extrinsic_objects.no_extrinsic import NoExtrinsic
from .extrinsic_objects.cv2_extrinsic import Cv2Extrinsic

from .undistort_points import undistort_points

def triangulate(
    image_points: Tuple[numpy.ndarray],
    intrinsic: Tuple[Optional[Intrinsic]],
    distortion: Tuple[Optional[Distortion]],
    extrinsic: Tuple[Optional[Extrinsic]],
    ) -> numpy.ndarray:
    r"""
    
    Triangulate 3D points from multiple views based on the given image points, intrinsic parameters, distortion models, and extrinsic parameters.
    
    The process to triangulate the 3D points is as follows:
    
    1. For each view, convert the ``image_points`` (:math:`\vec{x}_i`) to ``normalized_points`` (:math:`\vec{x}_n`) by applying the inverse intrinsic matrix transformation and undistorting using the distortion model (See :func:`pycvcam.undistort_points`).
    2. For each view, build the projection matrix :math:`P` using extrinsic parameters to create the following system of equations:
    
    ... math::
    
        A \cdot X = b
        
    Solve the system of equations using least squares on all views to obtain the 3D points.
    
    .. warning::
    
        This function does not accept (..., 2) shaped image points directly. Instead, it requires a tuple of 2D numpy arrays, each representing the image points from a different view.
        
    .. warning::
    
        To do current implementation, only work for implemented extrinsic models.
    
    Parameters
    ----------
    image_points : Tuple[numpy.ndarray]
        A tuple of 2D image points from multiple views. Each element should be of shape (N, 2), where N is the number of points.
        
    intrinsic : Tuple[Optional[Intrinsic]]
        A tuple of intrinsic parameter objects for each view. If None, a default intrinsic object with no intrinsic parameters will be used.
        
    distortion : Tuple[Optional[Distortion]]
        A tuple of distortion model objects for each view. If None, a default distortion object with no distortion will be used.
        
    extrinsic : Tuple[Optional[Extrinsic]]
        A tuple of extrinsic parameter objects for each view. If None, a default extrinsic object with no extrinsic parameters will be used.
    
    Returns
    -------
    numpy.ndarray
        The triangulated 3D points in the world coordinate system. Shape (N, 3)
        
    """
    if not isinstance(image_points, (tuple, list)):
        raise TypeError("image_points must be a tuple or list of numpy.ndarray")
    n_views = len(image_points)
    
    if not (isinstance(intrinsic, (tuple, list)) and len(intrinsic) == n_views):
        raise TypeError("intrinsic must be a tuple of Optional[Intrinsic] with the same length as image_points")
    if not (isinstance(distortion, (tuple, list)) and len(distortion) == n_views):
        raise TypeError("distortion must be a tuple of Optional[Distortion] with the same length as image_points")
    if not (isinstance(extrinsic, (tuple, list)) and len(extrinsic) == n_views):
        raise TypeError("extrinsic must be a tuple of Optional[Extrinsic] with the same length as image_points")
    
    n_points = None
    for index, view_image_points in enumerate(image_points):
        view_image_points = numpy.asarray(view_image_points)
        if view_image_points.ndim != 2 or view_image_points.shape[1] != 2:
            raise ValueError(f"Each element in image_points must be of shape (N, 2), but got {view_image_points.shape} at index {index}")
        if n_points is None:
            n_points = view_image_points.shape[0]
        elif n_points != view_image_points.shape[0]:
            raise ValueError("All views must have the same number of points")
        
    for index, view_intrinsic in enumerate(intrinsic):
        if view_intrinsic is None:
            intrinsic[index] = NoIntrinsic()
        if not isinstance(intrinsic[index], Intrinsic):
            raise TypeError(f"Each element in intrinsic must be of type Intrinsic or None, but got {type(intrinsic[index])} at index {index}")
        if not intrinsic[index].is_set():
            raise ValueError(f"Intrinsic parameters must be set for view at index {index}")
        
    for index, view_distortion in enumerate(distortion):
        if view_distortion is None:
            distortion[index] = NoDistortion()
        if not isinstance(distortion[index], Distortion):
            raise TypeError(f"Each element in distortion must be of type Distortion or None, but got {type(distortion[index])} at index {index}")
        if not distortion[index].is_set():
            raise ValueError(f"Distortion parameters must be set for view at index {index}")
        
    for index, view_extrinsic in enumerate(extrinsic):
        if view_extrinsic is None:
            extrinsic[index] = NoExtrinsic()
        if not isinstance(extrinsic[index], Extrinsic):
            raise TypeError(f"Each element in extrinsic must be of type Extrinsic or None, but got {type(extrinsic[index])} at index {index}")
        if not extrinsic[index].is_set():
            raise ValueError(f"Extrinsic parameters must be set for view at index {index}")
        
    # Undistort all the points
    normalized_points = []
    for index in range(n_views):
        undistorted = undistort_points(
            image_points[index],
            intrinsic[index],
            distortion[index],
        )  # shape (N, 2)
        normalized_points.append(undistorted)
        
    # Build the system of equations
    A = numpy.zeros((2 * n_views, 3, n_points), dtype=numpy.float64)
    b = numpy.zeros((2 * n_views, n_points), dtype=numpy.float64)
    
    for index in range(n_views):
        # Extrinsic = NoExtrinsic
        if isinstance(extrinsic[index], NoExtrinsic):
            R = numpy.eye(3)
            t = numpy.zeros((3, 1))
            for j in range(3):
                A[2 * index + 0, j, :] = normalized_points[index][:, 0] * R[2, j] - R[0, j]
                A[2 * index + 1, j, :] = normalized_points[index][:, 1] * R[2, j] - R[1, j]
            b[2 * index + 0, :] = t[0]
            b[2 * index + 1, :] = t[1]
            
        # Extrinsic = Cv2Extrinsic
        elif isinstance(extrinsic[index], Cv2Extrinsic):
            R_mat = extrinsic[index].rotation_matrix
            t_vec = extrinsic[index].translation_vector.reshape(3, 1)
            for j in range(3):
                A[2 * index + 0, j, :] = normalized_points[index][:, 0] * R_mat[2, j] - R_mat[0, j]
                A[2 * index + 1, j, :] = normalized_points[index][:, 1] * R_mat[2, j] - R_mat[1, j]
            b[2 * index + 0, :] = t_vec[0]
            b[2 * index + 1, :] = t_vec[1]
            
        else:
            raise NotImplementedError(f"Extrinsic type {type(extrinsic[index])} not implemented in triangulate function")
        
    # Solve the system of equations using least squares
    points_3d = numpy.zeros((n_points, 3), dtype=numpy.float64)
    for i in range(n_points):
        Ai = A[:, :, i]  # shape (2*n_views, 3)
        bi = b[:, i]     # shape (2*n_views,)
        Xi, _, _, _ = numpy.linalg.lstsq(Ai, bi, rcond=None)
        points_3d[i, :] = Xi
        
    return points_3d