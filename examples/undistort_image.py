import numpy
from pycvcam import undistort_image, ZernikeDistortion
import cv2
import csv
import os

def read_array1D(file_path):
    """
    Reads a 1D array from a text file.
    """
    values = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            try:
                float_values = [float(x) for x in row if x.strip()]
                values.extend(float_values)
            except ValueError:
                continue  # Skip malformed or empty lines
    return numpy.array(values, dtype=numpy.float64)

# Load the image to be distorted
src = cv2.imread(os.path.join(os.path.dirname(__file__), 'image.png'))

H, W = src.shape[:2]

# Define the distortion parameters
distortion = ZernikeDistortion(parameters=read_array1D(os.path.join(os.path.dirname(__file__), 'zernike_parameters.txt')))
distortion.radius = numpy.sqrt(((H-1)/2)**2 + ((W-1)/2)**2)  # Set the radius based on the image size
distortion.center = numpy.array([(W-1)/2, (H-1)/2], dtype=numpy.float64)  # Set the center of distortion

# Distort the image
undistorted_image = undistort_image(src, intrinsic=None, distortion=distortion, interpolation="spline3")
undistorted_image = numpy.clip(undistorted_image, 0, 255).astype(numpy.uint8)  # Ensure pixel values are valid

# Save the undistorted image
cv2.imwrite(os.path.join(os.path.dirname(__file__), 'undistorted_image.png'), undistorted_image)