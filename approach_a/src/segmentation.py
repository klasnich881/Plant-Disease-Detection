import cv2
import numpy as np

from src.config import (
    HEALTHY_GREEN_HUE_LOWER,
    HEALTHY_GREEN_HUE_UPPER,
    LEAF_HSV_LOWER,
    LEAF_HSV_UPPER,
    MORPH_KERNEL_SIZE,
)


def segment_leaf(hsv_image):
    lower = np.array(LEAF_HSV_LOWER, dtype=np.uint8)
    upper = np.array(LEAF_HSV_UPPER, dtype=np.uint8)

    mask = cv2.inRange(hsv_image, lower, upper)

    # clean up background noise and fill small holes inside the leaf
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # keep only the largest contour (the main leaf)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros_like(mask)

    if len(contours) > 0:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(leaf_mask, [biggest], -1, 255, thickness=cv2.FILLED)
    else:
        leaf_mask[:] = 255

    return leaf_mask


def detect_diseased_regions(hsv_image, leaf_mask):
    h = hsv_image[:, :, 0]

    # flag yellowing/browning pixels outside normal green hue
    non_green = ((h < HEALTHY_GREEN_HUE_LOWER) | (h > HEALTHY_GREEN_HUE_UPPER)).astype(np.uint8) * 255
    disease_mask = cv2.bitwise_and(non_green, leaf_mask)
    return disease_mask