import numpy
import cv2
from pycvcam import compute_rays, Cv2Extrinsic, Cv2Intrinsic, Cv2Distortion
import os

# Read the image : 
image = cv2.imread(os.path.join(os.path.dirname(__file__), 'image.png'))
image_height, image_width = image.shape[:2]

# Construct the intrinsic transformation :
intrinsic = Cv2Intrinsic.from_matrix(numpy.array([[1000, 0, image_width / 2],
                                                    [0, 1000, image_height / 2],
                                                    [0, 0, 1]]))

# Construct the distortion transformation:
distortion = Cv2Distortion(parameters=numpy.array([0.1, -0.05, 0, 0, 0]))

# Construct the extrinsic transformation:
extrinsic = Cv2Extrinsic.from_rt(rvec=[0.1, 0.2, 0.3], tvec=[0, 0, 5])

# Define the image points (e.g., pixels in the image):
pixel_points = numpy.indices((image_height, image_width)) # shape (2, H, W)
pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

# Throw rays from the image points to the scene:
rays = compute_rays(image_points, intrinsic, distortion, extrinsic, transpose=False).rays

# Here `rays` will contain the origin and direction of the rays in the world coordinate system with shape (..., 6).
# rays[i, :] = [origin_x, origin_y, origin_z, direction_x, direction_y, direction_z]

print(rays)  # Output the rays for further processing or visualization