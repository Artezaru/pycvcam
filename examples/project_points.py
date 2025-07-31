import numpy
from pycvcam import project_points, Cv2Distortion, Cv2Extrinsic, Cv2Intrinsic

# Define the 3D points in the world coordinate system
world_points = numpy.array([[0.0, 0.0, 5.0],
                            [0.1, -0.1, 5.0],
                            [-0.1, 0.2, 5.0],
                            [0.2, 0.1, 5.0],
                            [-0.2, -0.2, 5.0]]) # shape (5, 3)

# Define the rotation vector and translation vector
rvec = numpy.array([0.01, 0.02, 0.03])  # small rotation
tvec = numpy.array([0.1, -0.1, 0.2])    # small translation
extrinsic = Cv2Extrinsic.from_rt(rvec, tvec)

# Define the intrinsic camera matrix
K = numpy.array([[1000.0, 0.0, 320.0],
                [0.0, 1000.0, 240.0],
                [0.0, 0.0, 1.0]])

intrinsic = Cv2Intrinsic.from_matrix(K)

# Define the distortion model (optional)
distortion = Cv2Distortion(parameters = [0.1, 0.2, 0.3, 0.4, 0.5])

# Project the 3D points to 2D image points
result = project_points(world_points, intrinsic=intrinsic, distortion=distortion, extrinsic=extrinsic, transpose=False, dp=True, dx=True)
print("Projected image points:")
print(result.image_points) # shape (5, 2)