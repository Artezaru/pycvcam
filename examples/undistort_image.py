import numpy
from pycvcam import undistort_image, ZernikeDistortion
from pycvcam import read_transform
import cv2
import csv
import os

# Load the image to be distorted
src = cv2.imread(os.path.join(os.path.dirname(__file__), 'image.png'))

H, W = src.shape[:2]

# Define the distortion parameters
distortion = read_transform(os.path.join(os.path.dirname(__file__), 'zernike_transform.json'), ZernikeDistortion)

# Distort the image
undistorted_image = undistort_image(src, intrinsic=None, distortion=distortion, interpolation="spline3")
undistorted_image = numpy.clip(undistorted_image, 0, 255).astype(numpy.uint8)  # Ensure pixel values are valid

# Save the undistorted image
cv2.imwrite(os.path.join(os.path.dirname(__file__), 'undistorted_image.png'), undistorted_image)
