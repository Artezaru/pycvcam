"""

.. _sphx_glr__gallery_optimize_parameters.py:

Optimizing distortion parameters with least squares
======================================================================================================

This example illustrate how to use the ``optimize_parameters_trf`` function to optimize the distortion parameters of a camera model, which includes the intrinsic and distortion transformations.

.. seealso::

    - :func:`pycvcam.optimize_parameters_trf`: Optimize the parameters of a transformation using Trust Region Reflective optimization.
    - :func:`pycvcam.optimize_camera_trf`: Optimize the parameters of a complete camera model using Trust Region Reflective optimization.

"""

# %%
# Optimizing Parameters to fit a Distortion model (Image Alignment)
# -------------------------------------------------------------------
# In this example, we consider a random image, an intrinsic matrix and a distortion model.
# We distort the image by the distortion model and then add noise.
#
# The objective is to optimize the distortion parameters of the distortion model to
# fit the distorted image with noise.
#
# To do this, we will compute the flow between the both images and then optimize the
# distortion parameters to minimize the flow in the normalized image space.
#
# First create the images and compute the real flow between the original and
# distorted images.

import numpy
import pycvcam
import cv2
import matplotlib.pyplot as plt

image = pycvcam.get_lena_image()
image_height, image_width = image.shape[:2]
print("Image shape:", image.shape)

pixel_points = numpy.indices((image_height, image_width), dtype=numpy.float64)
pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

# Create an intrisic transformation
intrinsic = pycvcam.Cv2Intrinsic.from_matrix(
    [[1000.0, 0.0, image_width / 2], [0.0, 1000.0, image_height / 2], [0.0, 0.0, 1.0]]
)

# Create a distortion transformation in the image space and convert it to the normalized image space
distortion = pycvcam.ZernikeDistortion(
    parameters=[
        0.8541972545746392,
        -5.468596289790535,
        -5.974287819021697,
        14.292956075116104,
        2.1403205479372627,
        4.544169430137205,
        -0.10099732464199339,
        0.4363509204067417,
        -0.5106374355681896,
        -5.770087687650705,
        -0.39147505788710696,
        11.699411273002498,
    ]  # In pixels units, example values for a small distortion
)
distortion.center = ((image_width - 1) / 2, (image_height - 1) / 2)
distortion.radius = numpy.sqrt(
    (distortion.center[0]) ** 2 + (distortion.center[1]) ** 2
)
distortion.parameters_x = distortion.parameters_x / intrinsic.fx
distortion.parameters_y = distortion.parameters_y / intrinsic.fy
distortion.radius_x = distortion.radius_x / intrinsic.fx
distortion.radius_y = distortion.radius_y / intrinsic.fy
distortion.center_x = (distortion.center_x - intrinsic.cx) / intrinsic.fx
distortion.center_y = (distortion.center_y - intrinsic.cy) / intrinsic.fy

true_distorted_points = pycvcam.distort_points(
    image_points, intrinsic, distortion, P=intrinsic
)
true_flow = true_distorted_points - image_points  # shape (H*W, 2)
true_flow_magnitude = numpy.linalg.norm(true_flow, axis=1)

print("True flow magnitude statistics:")
print("  Min:", numpy.min(true_flow_magnitude))
print("  Max:", numpy.max(true_flow_magnitude))
print("  Mean:", numpy.mean(true_flow_magnitude))
print("  Median:", numpy.median(true_flow_magnitude))
print("  Std:", numpy.std(true_flow_magnitude))

# Create the distorted image with noise (5% of noise in the GL units)
distorted_image = pycvcam.distort_image(
    image, intrinsic, distortion, interpolation="spline3"
)
noise_GLpc = 5.0
noise = numpy.random.normal(0, noise_GLpc / 100, distorted_image.shape)
distorted_image = distorted_image * (1 + noise)  # Add multiplicative noise
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
# Then compute the optical flow between both images using the ``compute_optical_flow``
# function, which computes the dense optical flow between two images using the DIS algorithm.

flow_x, flow_y = pycvcam.compute_optical_flow(
    image,
    distorted_image,
    disflow_params={
        "PatchSize": 15,
        "PatchStride": 1,
    },
)
flow_x = flow_x.reshape(-1, 1)  # shape (H*W, 1)
flow_y = flow_y.reshape(-1, 1)  # shape (H*W, 1)

flow = numpy.hstack((flow_x, flow_y))  # shape (H*W, 2)

# Compare the computed flow with the true flow by visualizing the flow magnitude and
# components for both the computed flow and the true flow to ensure consistency.

fx = flow[:, 0].reshape(image_height, image_width)
fy = flow[:, 1].reshape(image_height, image_width)
F = numpy.sqrt(fx**2 + fy**2)
tfx = true_flow[:, 0].reshape(image_height, image_width)
tfy = true_flow[:, 1].reshape(image_height, image_width)
tF = numpy.sqrt(tfx**2 + tfy**2)
tF_max = numpy.max(tF)
tfx_min, tfx_max = numpy.min(tfx), numpy.max(tfx)
tfy_min, tfy_max = numpy.min(tfy), numpy.max(tfy)

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(F, cmap="inferno", vmin=0, vmax=tF_max)
ax1.set_title("Flow Magnitude")
ax1.axis("off")
ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(fx, cmap="inferno", vmin=tfx_min, vmax=tfx_max)
ax2.set_title("Flow X Component")
ax2.axis("off")
ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(fy, cmap="inferno", vmin=tfy_min, vmax=tfy_max)
ax3.set_title("Flow Y Component")
ax3.axis("off")
ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(tF, cmap="inferno", vmin=0, vmax=tF_max)
ax4.set_title("True Flow Magnitude")
ax4.axis("off")
ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(tfx, cmap="inferno", vmin=tfx_min, vmax=tfx_max)
ax5.set_title("True Flow X Component")
ax5.axis("off")
ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(tfy, cmap="inferno", vmin=tfy_min, vmax=tfy_max)
ax6.set_title("True Flow Y Component")
ax6.axis("off")
plt.show()


# %%
# To finish, we optimize the distortion parameters to minimize the
# flow in the normalized image space.
#
# To avoid the border effects of the distortion and undistortion, we will consider
# only the inner part of the image for the optimization by applying a mask to the points.

images_mask = numpy.ones_like(image, dtype=bool)
border_size = 50
images_mask[0:border_size, :] = False
images_mask[-border_size:, :] = False
images_mask[:, 0:border_size] = False
images_mask[:, -border_size:] = False

distorted_points = image_points + flow  # shape (H*W, 2)
normalized_points = intrinsic.inverse_transform(image_points).transformed_points
distorted_points = intrinsic.inverse_transform(distorted_points).transformed_points
normalized_points = normalized_points[images_mask.flatten()]
distorted_points = distorted_points[images_mask.flatten()]

initial_distortion = distortion.copy()
initial_distortion.parameters = numpy.zeros_like(distortion.parameters)

print("\n")
parameters, result = pycvcam.optimize_parameters_trf(
    initial_distortion,
    normalized_points,
    distorted_points,
    auto=True,  # Set ftol, xtol and gtol to 1e-8
    return_result=True,
    verbose_level=3,
)
print("\n")

optimize_distortion = initial_distortion.copy()
optimize_distortion.parameters = parameters

optimized_flow = (
    pycvcam.distort_points(image_points, intrinsic, optimize_distortion, P=intrinsic)
    - image_points
)

flow_error = numpy.linalg.norm(optimized_flow - true_flow, axis=1)  # shape (H*W,)
rmse_flow = numpy.sqrt(numpy.mean(flow_error[images_mask.flatten()] ** 2))
params_error = numpy.linalg.norm(parameters - distortion.parameters)
params_rel_error = params_error / numpy.linalg.norm(distortion.parameters)

print("Optimization success:", result.success)
print("Optimization message:", result.message)
print("Optimization cost:", result.cost)
print("Flow RMSE:", rmse_flow)
print("Parameters Error:", params_error, f"({params_rel_error:.2%})")


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
extrinsic = pycvcam.Cv2Extrinsic.from_rt(rvec, tvec)

# Define the Intrinsic transformation, example intrinsic camera matrix
K = numpy.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
intrinsic = pycvcam.Cv2Intrinsic.from_matrix(K)

# Define the Distortion transformation, example Brown-Conrady 5 parameters
distortion = pycvcam.Cv2Distortion(parameters=[0.1, 0.05, 0.01, 0.01, 0.01])

# Project the 3D points to 2D image points (without Jacobians)
result = pycvcam.project_points(
    world_points,
    intrinsic=intrinsic,
    distortion=distortion,
    extrinsic=extrinsic,
)  # pycvcam.TransformResult data class

image_points = result.image_points  # shape (100, 2)

# Optimize the camera parameters to fit the noisy image points
initial_distortion = distortion.copy()
initial_distortion.parameters = numpy.zeros_like(distortion.parameters)

initial_extrinsic = extrinsic.copy()
initial_extrinsic.parameters = numpy.zeros_like(extrinsic.parameters)

distortion_bounds = (
    numpy.array([-0.1, -0.1, -0.1, -0.1, -0.1]),  # k1, k2, p1, p2, k3 lower bounds
    numpy.array([0.1, 0.1, 0.1, 0.1, 0.1]),  # k1, k2, p1, p2, k3 upper bounds
)
extrinsic_bounds = (  # rvec and tvec bounds
    numpy.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]),  # rvec and tvec lower bounds
    numpy.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # rvec and tvec upper bounds
)

print("\n")
params, result = pycvcam.optimize_camera_trf(
    intrinsic,
    pycvcam.NoDistortion(),
    initial_extrinsic,
    world_points,
    image_points,
    mask_intrinsic=[False for _ in range(4)],  # Do not optimize intrinsic
    bounds_distortion=None,
    bounds_extrinsic=extrinsic_bounds,
    auto=True,  # Set ftol, xtol and gtol to 1e-8
    return_result=True,
    verbose_level=3,
)
print("\n")
optimized_intrinsic_params, optimized_distortion_params, optimized_extrinsic_params = (
    params
)
print("Optimized Intrinsic Parameters:", optimized_intrinsic_params)
print("Optimized Distortion Parameters:", optimized_distortion_params)
print("Optimized Extrinsic Parameters:", optimized_extrinsic_params)

optimized_distortion = initial_distortion.copy()
optimized_distortion.parameters = optimized_distortion_params
optimized_extrinsic = initial_extrinsic.copy()
optimized_extrinsic.parameters = optimized_extrinsic_params

optimized_image_points = pycvcam.project_points(
    world_points,
    intrinsic=intrinsic,
    distortion=optimized_distortion,
    extrinsic=optimized_extrinsic,
).image_points

error_points = numpy.linalg.norm(optimized_image_points - image_points, axis=1)
rmse_points = numpy.sqrt(numpy.mean(error_points**2))

error_distortion = numpy.linalg.norm(
    optimized_distortion.parameters - distortion.parameters
)
error_extrinsic = numpy.linalg.norm(
    optimized_extrinsic.parameters - extrinsic.parameters
)
rel_error_distortion = error_distortion / numpy.linalg.norm(distortion.parameters)
rel_error_extrinsic = error_extrinsic / numpy.linalg.norm(extrinsic.parameters)

print("Optimization success:", result.success)
print("Optimization message:", result.message)
print("Optimization cost:", result.cost)
print("RMSE Real Points:", rmse_points)
print("Error Distortion:", error_distortion, f"({rel_error_distortion:.2%})")
print("Error Extrinsic:", error_extrinsic, f"({rel_error_extrinsic:.2%})")
