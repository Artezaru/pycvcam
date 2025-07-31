import numpy
from pycvcam import undistort_points, Cv2Distortion, Cv2Intrinsic

# Define the 2D image points in the camera coordinate system
image_points = numpy.array([[320.0, 240.0],
                            [420.0, 440.0],
                            [520.0, 540.0],
                            [620.0, 640.0],
                            [720.0, 740.0]]) # shape (5, 2)

# Define the intrinsic camera matrix
K = numpy.array([[1000.0, 0.0, 320.0],
                [0.0, 1000.0, 240.0],
                [0.0, 0.0, 1.0]])

# Create the intrinsic object
intrinsic = Cv2Intrinsic.from_matrix(K)

# Define the distortion model (optional)
distortion = Cv2Distortion([0.1, 0.2, 0.3, 0.4, 0.5])

# Undistort the 2D image points
normalized_points = undistort_points(image_points, intrinsic=intrinsic, distortion=distortion, _skip=False)

print("Normalized points:")
print(normalized_points)  # shape (5, 2)