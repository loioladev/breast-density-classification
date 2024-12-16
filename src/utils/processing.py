"""
This module contains functions to process breast images, such as
cropping the breast from the image and checking if the breast is
on the left side of the image
"""

import logging

import cv2
import numpy as np
import pydicom
import torch
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
    the following work -> DOI - 10.1109/ICBAPS.2015.7292214

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


def apply_windowing(
    image: np.ndarray | torch.Tensor, ds: pydicom.Dataset
) -> np.ndarray:
    """
    Apply the windowing to the image using the DICOM metadata. The windowing
    is a technique to adjust the contrast and brightness of the image.

    :param image: Image to be processed
    :param ds: DICOM metadata
    :return: Processed image
    """

    # -- obtain the windowing parameters
    voi_func = ds.get("VOILUTFunction", "LINEAR").upper()
    window_width = ds.get("WindowWidth", [])
    window_center = ds.get("WindowCenter", [])
    try:
        if not isinstance(window_center, list):
            window_center = [window_center]
        window_center = [float(x) for x in window_center]
        if not isinstance(window_width, list):
            window_width = [window_width]
        window_width = [float(x) for x in window_width]
    except (TypeError, ValueError):
        window_center = []
        window_width = []

    if not window_width:
        logger.debug("No windowing parameters found")
        return image

    window_width = window_width[0]
    window_center = window_center[0]

    # -- convert the image to a tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # -- apply the windowing
    y_max = 255
    y_min = 0
    y_range = y_max - y_min
    image = image.float()
    if voi_func == "LINEAR" or voi_func == "LINEAR_EXACT":
        # -- dicom PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == "LINEAR":
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation"
                )
            window_center -= 0.5
            window_width -= 1
        s = y_range / window_width
        b = (-window_center / window_width + 0.5) * y_range + y_min
        image = image * s + b
        image = torch.clamp(image, y_min, y_max)
    elif voi_func == "SIGMOID":
        s = -4 / window_width
        image = y_range / (1 + torch.exp((image - window_center) * s)) + y_min
    else:
        raise ValueError(f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return image.numpy()
