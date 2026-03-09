import numpy
from pycvcam import (
    project_points,
    ZernikeDistortion,
    Cv2Extrinsic,
    Cv2Intrinsic,
    Cv2Distortion,
)
import os
import time
import matplotlib.pyplot as plt
import cv2

# Define the rotation vector and translation vector
rvec = numpy.array([0.01, 0.02, 0.03])  # small rotation
tvec = numpy.array([0.1, -0.1, 0.2])  # small translation
extrinsic = Cv2Extrinsic.from_rt(rvec, tvec)
print(extrinsic.frame)

# Define the intrinsic camera matrix
K = numpy.eye(3)
K[0, 0] = 1.1
K[1, 1] = 1.2
K[0, 2] = 0.5
K[1, 2] = 0.4
intrinsic = Cv2Intrinsic.from_matrix(K)

# Define the distortion model (Zernike with 72 parameters)
file = os.path.join(os.path.dirname(__file__), "zernike_transform.json")
zernike_distortion = ZernikeDistortion.from_json(file)
zernike_distortion.center = (0.1, -0.1)
print(
    "Zernike orders:",
    zernike_distortion.n_zer,
    "with N parameters:",
    zernike_distortion.n_params,
)

# Define a Cv2 like distortion model with 14 parameters (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, t1, t2)
cv2_parameters = numpy.array(
    [
        0.1,
        -0.05,
        0.01,
        0.02,
        0.005,
        0.002,
        -0.001,
        0.0005,
        0.003,
        -0.001,
        0.0005,
        -0.0002,
        0.1,
        -0.1,
    ]
)
cv2_distortion = Cv2Distortion(parameters=cv2_parameters)
print("Cv2 Distortion with 14 parameters.")


pycvcam_Zernike_times = []
pycvcam_Cv2_times = []
pycvcam_Zernike_times_jacobian = []
pycvcam_Cv2_times_jacobian = []
pycvcam_Zernike_times_only_dx = []
pycvcam_Cv2_times_only_dx = []
pycvcam_Zernike_times_only_ddistortion = []
pycvcam_Cv2_times_only_ddistortion = []
opencv_times = []

Npts = [10, 100, 1000, 10000, 100000, 1000000]
for N_points in Npts:
    # Define the 3D points in the world coordinate system
    world_points = numpy.random.uniform(-1, 1, size=(N_points, 3)) + numpy.array(
        [0.0, 0.0, 5.0]
    )  # shape (N_points, 3)

    start_time = time.time()
    result = project_points(
        world_points,
        intrinsic=intrinsic,
        distortion=zernike_distortion,
        extrinsic=extrinsic,
        transpose=False,
        dp=False,
        dx=False,
    )
    end_time = time.time()
    pycvcam_Zernike_times.append(end_time - start_time)
    assert numpy.all(
        numpy.isfinite(result.image_points)
    ), "Projection resulted in non-finite values" + str(result.image_points)

    start_time = time.time()
    result = project_points(
        world_points,
        intrinsic=intrinsic,
        distortion=cv2_distortion,
        extrinsic=extrinsic,
        transpose=False,
        dp=False,
        dx=False,
    )
    end_time = time.time()
    pycvcam_Cv2_times.append(end_time - start_time)
    assert numpy.all(
        numpy.isfinite(result.image_points)
    ), "Projection resulted in non-finite values."

    start_time = time.time()
    result = project_points(
        world_points,
        intrinsic=intrinsic,
        distortion=zernike_distortion,
        extrinsic=extrinsic,
        transpose=False,
        dp=True,
        dx=True,
    )
    end_time = time.time()
    pycvcam_Zernike_times_jacobian.append(end_time - start_time)

    start_time = time.time()
    result = project_points(
        world_points,
        intrinsic=intrinsic,
        distortion=cv2_distortion,
        extrinsic=extrinsic,
        transpose=False,
        dp=True,
        dx=True,
    )
    end_time = time.time()
    pycvcam_Cv2_times_jacobian.append(end_time - start_time)

    start_time = time.time()
    result = project_points(
        world_points,
        intrinsic=intrinsic,
        distortion=zernike_distortion,
        extrinsic=extrinsic,
        transpose=False,
        dp=False,
        dx=True,
    )
    end_time = time.time()
    pycvcam_Zernike_times_only_dx.append(end_time - start_time)

    start_time = time.time()
    result = project_points(
        world_points,
        intrinsic=intrinsic,
        distortion=cv2_distortion,
        extrinsic=extrinsic,
        transpose=False,
        dp=False,
        dx=True,
    )
    end_time = time.time()
    pycvcam_Cv2_times_only_dx.append(end_time - start_time)

    start_time = time.time()
    result = project_points(
        world_points,
        intrinsic=intrinsic,
        distortion=zernike_distortion,
        extrinsic=extrinsic,
        transpose=False,
        ddistortion=True,
        dx=False,
    )
    end_time = time.time()
    pycvcam_Zernike_times_only_ddistortion.append(end_time - start_time)

    start_time = time.time()
    result = project_points(
        world_points,
        intrinsic=intrinsic,
        distortion=cv2_distortion,
        extrinsic=extrinsic,
        transpose=False,
        ddistortion=True,
        dx=False,
    )
    end_time = time.time()
    pycvcam_Cv2_times_only_ddistortion.append(end_time - start_time)

    # OpenCV projection for comparison (only for Cv2Distortion)
    start_time = time.time()
    image_points_opencv, _ = cv2.projectPoints(
        world_points,
        rvec,
        tvec,
        K,
        cv2_parameters,
    )
    end_time = time.time()
    opencv_times.append(end_time - start_time)
    assert numpy.all(
        numpy.isfinite(image_points_opencv)
    ), "OpenCV projection resulted in non-finite values."
    assert image_points_opencv.shape == (
        N_points,
        1,
        2,
    ), "Unexpected shape from OpenCV projection: " + str(image_points_opencv.shape)
    image_points_opencv = image_points_opencv.reshape(-1, 2)  # shape (N_points, 2)
    assert numpy.allclose(image_points_opencv, result.image_points, atol=1e-5), (
        "OpenCV and pycvcam projections do not match within tolerance for Cv2Distortion."
        + str(image_points_opencv - result.image_points)
    )


# Plot the results
plt.figure(figsize=(12, 7))
plt.loglog(
    Npts,
    pycvcam_Zernike_times,
    label=f"Zernike Distortion - {pycvcam_Zernike_times[-1]:.1f}s",
)
plt.loglog(
    Npts, pycvcam_Cv2_times, label=f"Cv2 Distortion - {pycvcam_Cv2_times[-1]:.1f}s"
)
plt.loglog(
    Npts,
    pycvcam_Zernike_times_jacobian,
    label=f"Zernike with Jacobian (dx, dp) - {pycvcam_Zernike_times_jacobian[-1]:.1f}s",
)
plt.loglog(
    Npts,
    pycvcam_Cv2_times_jacobian,
    label=f"Cv2 with Jacobian (dx, dp) - {pycvcam_Cv2_times_jacobian[-1]:.1f}s",
)
plt.loglog(
    Npts,
    pycvcam_Zernike_times_only_dx,
    label=f"Zernike Only dx Jacobian - {pycvcam_Zernike_times_only_dx[-1]:.1f}s",
)
plt.loglog(
    Npts,
    pycvcam_Cv2_times_only_dx,
    label=f"Cv2 Only dx Jacobian - {pycvcam_Cv2_times_only_dx[-1]:.1f}s",
)
plt.loglog(
    Npts,
    pycvcam_Zernike_times_only_ddistortion,
    label=f"Zernike Only ddistortion Jacobian - {pycvcam_Zernike_times_only_ddistortion[-1]:.1f}s",
)
plt.loglog(
    Npts,
    pycvcam_Cv2_times_only_ddistortion,
    label=f"Cv2 Only ddistortion Jacobian - {pycvcam_Cv2_times_only_ddistortion[-1]:.1f}s",
)
plt.loglog(
    Npts,
    opencv_times,
    label=f"OpenCV projectPoints (Full Model) - {opencv_times[-1]:.1f}s",
)
plt.xlabel("Number of 3D points")
plt.ylabel("Computation time (s)")
plt.title(
    "3D to 2D Point Projection Computation Time \n(Zernike 72 parameters vs Cv2 14 parameters)"
)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(
    os.path.join(os.path.dirname(__file__), "project_points_computation_time.png")
)
plt.show()
