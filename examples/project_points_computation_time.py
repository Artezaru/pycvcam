import numpy
from pycvcam import project_points, ZernikeDistortion, Cv2Extrinsic, Cv2Intrinsic
import os
import time
import matplotlib.pyplot as plt

# Define the rotation vector and translation vector
rvec = numpy.array([0.01, 0.02, 0.03])  # small rotation
tvec = numpy.array([0.1, -0.1, 0.2])    # small translation
extrinsic = Cv2Extrinsic.from_rt(rvec, tvec)

# Define the intrinsic camera matrix
K = numpy.eye(3)  # Identity matrix for simplicity
intrinsic = Cv2Intrinsic.from_matrix(K)

# Define the distortion model (optional)
file = os.path.join(os.path.dirname(__file__), 'zernike_transform.json')
distortion = ZernikeDistortion.from_json(file)
print(distortion.Nzer)

times = []
times_jacobian = []
times_only_dx = []
times_only_ddistortion = []
for N_points in [10, 100, 1000, 10000, 100000, 1000000]:
    # Define the 3D points in the world coordinate system
    world_points = numpy.random.uniform(-1, 1, size=(N_points, 3)) + numpy.array([0.0, 0.0, 5.0])  # shape (N_points, 3)

    start_time = time.time()
    result = project_points(world_points, intrinsic=intrinsic, distortion=distortion, extrinsic=extrinsic, transpose=False, dp=False, dx=False)
    end_time = time.time()
    times.append(end_time - start_time)
    
    start_time = time.time()
    result = project_points(world_points, intrinsic=intrinsic, distortion=distortion, extrinsic=extrinsic, transpose=False, dp=True, dx=True)
    end_time = time.time()
    times_jacobian.append(end_time - start_time)
    
    start_time = time.time()
    result = project_points(world_points, intrinsic=intrinsic, distortion=distortion, extrinsic=extrinsic, transpose=False, dp=False, dx=True)
    end_time = time.time()
    times_only_dx.append(end_time - start_time)
    
    start_time = time.time()
    result = project_points(world_points, intrinsic=intrinsic, distortion=distortion, extrinsic=extrinsic, transpose=False, ddistortion=True, dx=False)
    end_time = time.time()
    times_only_ddistortion.append(end_time - start_time)
    

# Plot the results
plt.figure()
plt.loglog([10, 100, 1000, 10000, 100000, 1000000], times, label='No Jacobian')
plt.loglog([10, 100, 1000, 10000, 100000, 1000000], times_jacobian, label='With Jacobian (dx, dp)')
plt.loglog([10, 100, 1000, 10000, 100000, 1000000], times_only_dx, label='Only dx Jacobian')
plt.loglog([10, 100, 1000, 10000, 100000, 1000000], times_only_ddistortion, label='Only ddistortion Jacobian')
plt.xlabel('Number of 3D points')
plt.ylabel('Computation time (s)')
plt.title('3D to 2D Point Projection Computation Time')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(os.path.dirname(__file__), 'project_points_computation_time.png'))
plt.show()