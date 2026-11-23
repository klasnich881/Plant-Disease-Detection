import cv2
import numpy as np

# Reads an image from disk, resizes it to 256x256 pixels, and applies a 5x5
# Gaussian blur filter using OpenCV to reduce high-frequency image noise.
def preprocess_image(image_path: str, target_size: tuple = (256, 256), kernel_size: tuple = (5, 5)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    smoothed = cv2.GaussianBlur(resized, kernel_size, sigmaX=0)
    return smoothed