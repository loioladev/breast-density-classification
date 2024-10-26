"""
This module contains functions to process breast images, such as
cropping the breast from the image and checking if the breast is
on the left side of the image
"""

import logging

import cv2
import numpy as np
from skimage.morphology import binary_closing, binary_opening, dilation, disk, erosion

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


def recort_breast_morp(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Recort the breast from the image, excluding the excess empty space
    and the artifacts that may be present in the image. This function
    uses morphological operations to process the image, inspired by
    the following work: DOI - 10.1109/ICBAPS.2015.7292214

    :param image: Image to be processed
    :return image: Processed image and the bounding box of the breast
    """
    # -- thresholding
    _, binary_image = cv2.threshold(image, 18, 255, cv2.THRESH_BINARY)

    # -- find the largest contour
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)

    # -- create a binary image with the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    # -- apply morphological operations
    mask = binary_opening(mask, disk(1))
    mask = binary_closing(mask, disk(1))
    eroded_mask = erosion(mask, disk(5))
    dilatated_mask = dilation(eroded_mask, disk(5))

    # -- convert mask from bool to int
    dilatated_mask = dilatated_mask.astype(np.uint8)

    # -- apply the mask to the image
    image = cv2.bitwise_and(image, image, mask=dilatated_mask)

    # -- crop the image
    x, y, w, h = cv2.boundingRect(largest_contour)
    image = image[y : y + h, x : x + w]
    return image, (x, y, w, h)
