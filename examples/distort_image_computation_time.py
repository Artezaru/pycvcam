import numpy
from pycvcam import distort_image, ZernikeDistortion
from pycvcam import read_transform
import cv2
import csv
import os
import time
import matplotlib.pyplot as plt

# Define the distortion parameters
distortion = read_transform(
    os.path.join(os.path.dirname(__file__), "zernike_transform.json"), ZernikeDistortion
)

image_sizes = [
    50,
    100,
    500,
    1000,
    2000,
    3000,
]
methods = [
    ("undistort", "linear"),
    ("undistort", "cubic"),
    ("undistort", "lanczos4"),
    ("undistort", "spline3"),
    ("distort", "linear"),
    ("distort", "clough"),
]

results = []
for size in image_sizes:
    print(f"Processing image size: {size}x{size} ...")
    # Create a dummy image of the specified size
    src = numpy.random.randint(0, 256, (size, size), dtype=numpy.uint8)
    sub_results = []

    for method, interpolation in methods:
        print(f"  Processing method: {method} with interpolation: {interpolation} ...")
        start_time = time.time()
        distorted_image = distort_image(
            src,
            intrinsic=None,
            distortion=distortion,
            method=method,
            interpolation=interpolation,
            distortion_kwargs=(
                {
                    "max_iter": 10,
                    "eps": 5e-2,
                }
                if method == "undistort"
                else None
            ),  # 0.05 pixel tolerance for convergence
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        sub_results.append(elapsed_time)
    results.extend(sub_results)

# Display the results
plt.figure(figsize=(12, 7))
for method, interpolation in methods:
    method_results = results[methods.index((method, interpolation)) :: len(methods)]
    plt.loglog(
        numpy.array(image_sizes) ** 2,
        method_results,
        marker="o",
        label=f"{method} - {interpolation}",
    )
plt.xticks([s**2 for s in image_sizes], [f"{s}x{s}" for s in image_sizes], rotation=45)
plt.xlabel("Image Size (width x height)")
plt.ylabel("Computation Time (seconds)")
plt.title(
    "Distort/Undistort Image Computation Time for Different Methods and Interpolations"
)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(
    os.path.join(os.path.dirname(__file__), "distort_image_computation_time.png")
)
plt.show()
