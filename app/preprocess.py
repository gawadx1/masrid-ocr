from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.utils import order_points, rotate_image


def load_image(image_source: str | bytes | np.ndarray) -> np.ndarray:
    """Load an image from path, bytes, or an existing numpy array."""
    if isinstance(image_source, np.ndarray):
        image = image_source.copy()
    elif isinstance(image_source, bytes):
        buffer = np.frombuffer(image_source, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    elif isinstance(image_source, str):
        image = cv2.imread(image_source, cv2.IMREAD_COLOR)
    else:
        raise TypeError("image_source must be a path, bytes, or numpy.ndarray.")

    if image is None or image.size == 0:
        raise ValueError("Unable to load image from the provided source.")
    return image


def resize_large_image(image: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    """Resize oversized images while preserving aspect ratio."""
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_dim:
        return image

    scale = max_dim / float(largest_side)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def detect_card(image: np.ndarray) -> np.ndarray:
    """Detect the largest rectangular contour and return its four corner points."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = image.shape[0] * image.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.15:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    if contours:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        return np.asarray(box, dtype=np.float32)

    height, width = image.shape[:2]
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def perspective_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Warp the detected card region into a top-down view."""
    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_right = np.linalg.norm(top_right - bottom_right)
    height_left = np.linalg.norm(top_left - bottom_left)

    max_width = max(int(width_top), int(width_bottom))
    max_height = max(int(height_left), int(height_right))
    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _estimate_skew_angle(image: np.ndarray) -> float:
    """Estimate text skew angle from foreground pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 50:
        return 0.0

    angle = float(cv2.minAreaRect(coords)[-1])
    if angle < -45.0:
        angle = 90.0 + angle
    elif angle > 45.0:
        angle = angle - 90.0
    return angle


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """Rotate the card so its text becomes horizontally aligned."""
    corrected = image.copy()

    if corrected.shape[0] > corrected.shape[1]:
        corrected = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)

    angle = _estimate_skew_angle(corrected)
    if abs(angle) > 0.25:
        corrected = rotate_image(corrected, angle)
    return corrected


def preprocess_image(image_source: str | bytes | np.ndarray, max_dim: int = 1800) -> dict[str, Any]:
    """Run the full image preprocessing pipeline for OCR."""
    original = load_image(image_source)
    resized = resize_large_image(original, max_dim=max_dim)
    card_points = detect_card(resized)
    warped = perspective_transform(resized, card_points)
    oriented = correct_orientation(warped)

    gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    return {
        "original": original,
        "resized": resized,
        "card_points": card_points,
        "card": oriented,
        "gray": gray,
        "blurred": blurred,
        "thresholded": thresholded,
    }

