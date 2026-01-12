import numpy
from pycvcam import project_points, Cv2Distortion, Cv2Extrinsic, Cv2Intrinsic, triangulate

# Define the 3D points in the world coordinate system
world_points = numpy.array([[0.0, 0.0, 5.0],
                            [0.1, -0.1, 5.0],
                            [-0.1, 0.2, 5.0],
                            [0.2, 0.1, 5.0],
                            [-0.2, -0.2, 5.0]]) # shape (5, 3)

# Define a first camera
rvec = numpy.array([0.01, 0.02, 0.03])  # small rotation
tvec = numpy.array([0.1, -0.1, 0.2])    # small translation
extrinsic1 = Cv2Extrinsic.from_rt(rvec, tvec)

K = numpy.array([[1000.0, 0.0, 320.0],
                [0.0, 1000.0, 240.0],
                [0.0, 0.0, 1.0]])

intrinsic1 = Cv2Intrinsic.from_matrix(K)

distortion1 = Cv2Distortion(parameters = [0.1, 0.2, 0.3, 0.4, 0.5])

# Define a second camera
rvec = numpy.array([-0.02, 0.01, 0.04])  # small rotation
tvec = numpy.array([-0.1, 0.2, -0.1])    # small translation
extrinsic2 = Cv2Extrinsic.from_rt(rvec, tvec)

intrinsic2 = Cv2Intrinsic.from_matrix(K) 

distortion2 = Cv2Distortion(parameters = [-0.1, 0.05, -0.02, 0.03, -0.04])

# Project points using both cameras
projected_points_list = [
    project_points(world_points, intrinsic=intrinsic1, distortion=distortion1, extrinsic=extrinsic1, transpose=False).image_points,
    project_points(world_points, intrinsic=intrinsic2, distortion=distortion2, extrinsic=extrinsic2, transpose=False).image_points,
]

# triangulate the 3D points from the two views
reconstructed_points = triangulate(
    projected_points_list,
    intrinsic=[intrinsic1, intrinsic2],
    distortion=[distortion1, distortion2],
    extrinsic=[extrinsic1, extrinsic2],
)
print("Reconstructed 3D points:")
print(reconstructed_points)