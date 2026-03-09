"""

.. _sphx_glr__gallery_optimize_parameters.py:

Optimizing distortion parameters with least squares
======================================================================================================

This example illustrate how to use the ``optimize_parameters_least_squares`` function to optimize the distortion parameters of a camera model, which includes the intrinsic and distortion transformations.

.. seealso::

    - :func:`pycvcam.optimize_parameters_least_squares` for the function to optimize distortion parameters.
    - :func:`pycvcam.optimize_camera_least_squares` for the function to optimize camera parameters (intrinsic, distortion, and extrinsic).
    - :func:`pycvcam.optimize_chain_parameters_least_squares` for the function to optimize parameters of chains of transformations.

"""

# %%
# Optimizing Parameters to fit a Distortion model (Image Alignment)
# -------------------------------------------------------------------
# In this example, we consider a random image, an intrinsic matrix and a distortion model.
# We distort the image by the distortion model and then add noise.
#
# The objective is to optimize the distortion parameters of the distortion model to
# fit the distorted image with noise.
# To do this, we will compute the flow between the both images and then optimize the
# distortion parameters to minimize the flow in the normalized image space.
#

import numpy

numpy.random.seed(36)  # For reproducibility
from pycvcam import Cv2Distortion, Cv2Extrinsic, Cv2Intrinsic
import pycvcam
import pycvcam.optimize as pycvopt
import cv2
import matplotlib.pyplot as plt

image = pycvcam.get_lena_image()
image_height, image_width = image.shape[:2]
print("Image shape:", image.shape)
intrinsic = Cv2Intrinsic.from_matrix(
    numpy.array(
        [[100, 0, image_width / 2], [0, 100, image_height / 2], [0, 0, 1]],
        dtype=numpy.float64,
    )
)
distortion = Cv2Distortion(
    parameters=numpy.array([0.1, 0.04, 0.01, 0.01, 0.01], dtype=numpy.float64)
)

distorted_image = pycvcam.distort_image(
    image, intrinsic, distortion, interpolation="spline3"
)
noise_image = numpy.random.normal(0, 0, distorted_image.shape).astype(numpy.float64)
distorted_image = distorted_image.astype(numpy.float64) + noise_image
distorted_image = numpy.clip(distorted_image, 0, 255).astype(numpy.uint8)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(image, cmap="gray")
ax1.set_title("Original Image")
ax1.axis("off")
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(distorted_image, cmap="gray")
ax2.set_title("Distorted Image with Noise")
ax2.axis("off")


# %%
# Then compute the optical flow between both images and convert it to the normalized image
# space.

flow_x, flow_y = pycvcam.compute_optical_flow(
    image,
    distorted_image,
    disflow_params={
        "PatchSize": 15,
        "PatchStride": 1,
    },
)

images_mask = numpy.ones_like(image, dtype=bool)
border_size = 50
images_mask[0:border_size, :] = False
images_mask[-border_size:, :] = False
images_mask[:, 0:border_size] = False
images_mask[:, -border_size:] = False

pixel_points = numpy.indices(
    (image_height, image_width), dtype=numpy.float64
)  # shape (2, H, W)
pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

flow = numpy.stack((flow_x, flow_y), axis=-1).reshape(-1, 2)  # shape (H*W, 2)
distorted_points = image_points + flow  # shape (H*W, 2)

normalized_points = intrinsic.inverse_transform(image_points).transformed_points
distorted_points = intrinsic.inverse_transform(distorted_points).transformed_points
normalized_points = normalized_points[images_mask.flatten()]
distorted_points = distorted_points[images_mask.flatten()]

# pycvcam.display_optical_flow(image, flow_x, flow_y)

# %%
# To finish, we optimize the distortion parameters to minimize the
# flow in the normalized image space.

initial_distortion = Cv2Distortion(parameters=numpy.zeros(5, dtype=numpy.float64))
parameters, result = pycvopt.optimize_parameters_least_squares(
    initial_distortion,
    normalized_points,
    distorted_points,
    auto=True,  # Set ftol, xtol and gtol to 1e-8
    return_result=True,
    verbose_level=3,
)

error = numpy.linalg.norm(parameters - distortion.parameters)
rel_error = error / numpy.linalg.norm(distortion.parameters)

print("Optimized parameters:", parameters)
print("Optimization success:", result.success)
print("Optimization message:", result.message)
print("Optimization cost:", result.cost)
print("Error:", error, f"({rel_error:.2%})")

# %%
# Optimizing Parameters of a complete Camera Model (PnP problem)
# -----------------------------------------------------------------
# In this example, we optimize the parameters of a complete camera model,
# which includes the intrinsic, distortion and extrinsic transformations.
# To do this, consider a PnP problem with a set of 3D points and their
# corresponding 2D projections in the image.
#
# For example, create a point cloud in a rectangle with bounds :math:`[-1, 1]` meters in the :math:`x` and :math:`y` directions and with a :math:`z` component in the range :math:`[4.5, 5.5]` meters.
# Then place a camera near the origin with a small rotation and translation and project the 3D points to 2D image points using a pinhole camera model with Brown-Conrady distortion with a focal length of 1000 pixels and a principal point at (320, 240) pixels.

print("\n\n\n")

# Define the 3D points in the world coordinate system
x = numpy.random.uniform(-1.0, 1.0, (100, 1))  # shape (100, 1)
y = numpy.random.uniform(-1.0, 1.0, (100, 1))  # shape (100, 1)
z = numpy.random.uniform(4.5, 5.5, (100, 1))  # shape (100, 1)
world_points = numpy.hstack((x, y, z))  # shape (100, 3)

# Define the Extrinsic transformation, example rotation vector and translation vector
rvec = numpy.array([0.01, 0.02, 0.03])  # small rotation
tvec = numpy.array([0.01, -0.05, 0.04])  # small translation
extrinsic = Cv2Extrinsic.from_rt(rvec, tvec)

# Define the Intrinsic transformation, example intrinsic camera matrix
K = numpy.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
intrinsic = Cv2Intrinsic.from_matrix(K)

# Define the Distortion transformation, example Brown-Conrady 5 parameters
distortion = Cv2Distortion(parameters=[0.1, 0.05, 0.01, 0.01, 0.01])

# Project the 3D points to 2D image points (without Jacobians)
result = pycvcam.project_points(
    world_points,
    intrinsic=intrinsic,
    distortion=distortion,
    extrinsic=extrinsic,
)  # pycvcam.TransformResult data class

image_points = result.image_points  # shape (100, 2)

# Add noise to the image points
noise = numpy.random.normal(0, 0.1, image_points.shape)  # shape (100, 2)
noisy_image_points = image_points  # + noise

# Optimize the camera parameters to fit the noisy image points
initial_distortion = Cv2Distortion(parameters=numpy.zeros(5, dtype=numpy.float64))
initial_extrinsic = Cv2Extrinsic.from_rt(
    rvec=numpy.zeros(3, dtype=numpy.float64), tvec=numpy.zeros(3, dtype=numpy.float64)
)

distortion_bounds = (
    numpy.array([-0.1, -0.1, -0.1, -0.1, -0.1]),  # k1, k2, p1, p2, k3 lower bounds
    numpy.array([0.1, 0.1, 0.1, 0.1, 0.1]),  # k1, k2, p1, p2, k3 upper bounds
)
extrinsic_bounds = (  # rvec and tvec bounds
    numpy.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]),  # rvec and tvec lower bounds
    numpy.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # rvec and tvec upper bounds
)

optimized_intrinsic, optimized_distortion, optimized_extrinsic, result = (
    pycvopt.optimize_camera_least_squares(
        intrinsic,
        initial_distortion,
        initial_extrinsic,
        world_points,
        noisy_image_points,
        mask_intrinsic=[False for _ in range(4)],  # Do not optimize intrinsic
        bounds_distortion=distortion_bounds,
        bounds_extrinsic=extrinsic_bounds,
        auto=True,  # Set ftol, xtol and gtol to 1e-8
        return_result=True,
        verbose_level=2,
    )
)

optimized_intrinsic, optimized_distortion, optimized_extrinsic, result = (
    pycvopt.optimize_camera_least_squares(
        intrinsic,
        initial_distortion,
        initial_extrinsic,
        world_points,
        noisy_image_points,
        mask_intrinsic=[False for _ in range(4)],  # Do not optimize intrinsic
        bounds_distortion=distortion_bounds,
        bounds_extrinsic=extrinsic_bounds,
        auto=True,  # Set ftol, xtol and gtol to 1e-8
        return_result=True,
        verbose_level=3,
    )
)

error_distortion = numpy.linalg.norm(optimized_distortion - distortion.parameters)
error_extrinsic = numpy.linalg.norm(optimized_extrinsic - extrinsic.parameters)
rel_error_distortion = error_distortion / numpy.linalg.norm(distortion.parameters)
rel_error_extrinsic = error_extrinsic / numpy.linalg.norm(extrinsic.parameters)

print("Optimized Distortion Parameters:\n", optimized_distortion)
print("Optimized Extrinsic Parameters:\n", optimized_extrinsic)
print("Optimization success:", result.success)
print("Optimization message:", result.message)
print("Optimization cost:", result.cost)
print("Error Distortion:", error_distortion, f"({rel_error_distortion:.2%})")
print("Error Extrinsic:", error_extrinsic, f"({rel_error_extrinsic:.2%})")


# %%
# Optimize a complete camera setup with chains of transformations
# -----------------------------------------------------------------
#
# Lets assume, we have camera with same intrinsic and distortion
# transformations as before but with two different extrinsic transformations
# (for example, two different poses of the camera).
#
# To optimize the parameters of this camera setup, we can use the
# ``optimize_chain_parameters_least_squares`` function, which allows to optimize the
# parameters of a chain of transformations.
#
# Define the first chain of transformations as :
#
# - Extrinsic transformation 1 + Distortion transformation + Intrinsic transformation
#
# Define the second chain of transformations as :
#
# - Extrinsic transformation 2 + Distortion transformation + Intrinsic transformation
#
# .. warning::
#
#   The order of the transformations in the chain is important and should be consistent
#   with the order of the parameters in the optimization function.

print("\n\n\n")

x = numpy.random.uniform(-1.0, 1.0, (100, 1))  # shape (100, 1)
y = numpy.random.uniform(-1.0, 1.0, (100, 1))  # shape (100, 1)
z = numpy.random.uniform(4.5, 5.5, (100, 1))  # shape (100, 1)
world_points = numpy.hstack((x, y, z))  # shape (100, 3)

K = numpy.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
intrinsic = Cv2Intrinsic.from_matrix(K)

distortion = Cv2Distortion(parameters=[0.1, 0.05, 0.01, 0.01, 0.01])

rvec1 = numpy.array([0.01, 0.02, 0.03])  # small rotation
tvec1 = numpy.array([0.01, -0.05, 0.04])  # small translation
extrinsic1 = Cv2Extrinsic.from_rt(rvec1, tvec1)

rvec2 = numpy.array([-0.02, 0.01, -0.01])  # small rotation
tvec2 = numpy.array([-0.03, 0.02, -0.01])  # small translation
extrinsic2 = Cv2Extrinsic.from_rt(rvec2, tvec2)

transforms = [extrinsic1, extrinsic2, distortion, intrinsic]
chains = [
    [0, 2, 3],  # Chain 1: Extrinsic 1 + Distortion + Intrinsic
    [1, 2, 3],  # Chain 2: Extrinsic 2 + Distortion + Intrinsic
]

image_points_chain1 = pycvcam.project_points(
    world_points,
    intrinsic=intrinsic,
    distortion=distortion,
    extrinsic=extrinsic1,
).image_points

image_points_chain2 = pycvcam.project_points(
    world_points,
    intrinsic=intrinsic,
    distortion=distortion,
    extrinsic=extrinsic2,
).image_points


initial_distortion = Cv2Distortion(parameters=numpy.zeros(5, dtype=numpy.float64))
initial_extrinsic1 = Cv2Extrinsic.from_rt(
    rvec=numpy.zeros(3, dtype=numpy.float64), tvec=numpy.zeros(3, dtype=numpy.float64)
)
initial_extrinsic2 = Cv2Extrinsic.from_rt(
    rvec=numpy.zeros(3, dtype=numpy.float64), tvec=numpy.zeros(3, dtype=numpy.float64)
)

optimized_transforms, result = pycvopt.optimize_chain_parameters_least_squares(
    transforms,
    chains,
    [world_points, world_points],
    [image_points_chain1, image_points_chain2],
    mask=[None, None, None, [False for _ in range(4)]],  # Do not optimize intrinsic
    bounds=[
        extrinsic_bounds,  # Extrinsic 1 bounds
        extrinsic_bounds,  # Extrinsic 2 bounds
        distortion_bounds,  # Distortion bounds
        None,  # No bounds for intrinsic
    ],
    auto=True,  # Set ftol, xtol and gtol to 1e-8
    return_result=True,
)

optimized_extrinsic1 = optimized_transforms[0]
optimized_extrinsic2 = optimized_transforms[1]
optimized_distortion = optimized_transforms[2]

error_distortion = numpy.linalg.norm(optimized_distortion - distortion.parameters)
error_extrinsic1 = numpy.linalg.norm(optimized_extrinsic1 - extrinsic1.parameters)
error_extrinsic2 = numpy.linalg.norm(optimized_extrinsic2 - extrinsic2.parameters)
rel_error_distortion = error_distortion / numpy.linalg.norm(distortion.parameters)
rel_error_extrinsic1 = error_extrinsic1 / numpy.linalg.norm(extrinsic1.parameters)
rel_error_extrinsic2 = error_extrinsic2 / numpy.linalg.norm(extrinsic2.parameters)

print("Optimized Distortion Parameters:\n", optimized_distortion)
print("Optimized Extrinsic 1 Parameters:\n", optimized_extrinsic1)
print("Optimized Extrinsic 2 Parameters:\n", optimized_extrinsic2)
print("Optimization success:", result.success)
print("Optimization message:", result.message)
print("Optimization cost:", result.cost)
print("Error Distortion:", error_distortion, f"({rel_error_distortion:.2%})")
print("Error Extrinsic 1:", error_extrinsic1, f"({rel_error_extrinsic1:.2%})")
print("Error Extrinsic 2:", error_extrinsic2, f"({rel_error_extrinsic2:.2%})")
