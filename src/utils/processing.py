"""
This module contains functions to process breast images, such as
cropping the breast from the image and checking if the breast is
on the left side of the image
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger()


def left_side_breast(image: np.ndarray) -> bool:
    """
    Check if the breast is on the left side of the image by counting the
    number of non-zero pixels on the left and right side of the image

    :param image: Image to be processed
    :return bool: True if the breast is on the left side, False otherwise
    """
    left_blank_space = image[:, : image.shape[1] // 2]
    right_blank_space = image[:, image.shape[1] // 2 :]
    left_empty = cv2.countNonZero(left_blank_space)
    right_empty = cv2.countNonZero(right_blank_space)
    return left_empty > right_empty


def recort_breast(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Recort the breast from the image, excluding the excess empty space
    and the artifacts that may be present in the image

    :param image: Image to be processed
    :return image: Processed image and the bounding box of the breast
    """
    # -- thresholding
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # -- find the biggest contour
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    breast_contour = max(contours, key=cv2.contourArea)

    # -- crop the image
    x, y, w, h = cv2.boundingRect(breast_contour)
    image = image[y : y + h, x : x + w]

    return image, (x, y, w, h)
