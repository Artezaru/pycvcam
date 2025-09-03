import numpy
from pycvcam import distort_image, ZernikeDistortion
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
distorted_image = distort_image(src, intrinsic=None, distortion=distortion, interpolation="spline3")
distorted_image = numpy.clip(distorted_image, 0, 255).astype(numpy.uint8)  # Ensure pixel values are valid

# Save the distorted image
cv2.imwrite(os.path.join(os.path.dirname(__file__), 'distorted_image.png'), distorted_image)