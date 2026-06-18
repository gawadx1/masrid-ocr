from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


EASTERN_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


@dataclass(slots=True)
class OCRToken:
    """Single OCR token with geometry and confidence."""

    text: str
    confidence: float
    box: np.ndarray

    @property
    def min_x(self) -> float:
        return float(np.min(self.box[:, 0]))

    @property
    def max_x(self) -> float:
        return float(np.max(self.box[:, 0]))

    @property
    def min_y(self) -> float:
        return float(np.min(self.box[:, 1]))

    @property
    def max_y(self) -> float:
        return float(np.max(self.box[:, 1]))

    @property
    def center_x(self) -> float:
        return float(np.mean(self.box[:, 0]))

    @property
    def center_y(self) -> float:
        return float(np.mean(self.box[:, 1]))

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def width(self) -> float:
        return self.max_x - self.min_x


@dataclass(slots=True)
class OCRRow:
    """OCR tokens merged into a text row."""

    tokens: list[OCRToken]

    @property
    def text(self) -> str:
        return " ".join(token.text.strip() for token in sorted(self.tokens, key=lambda item: item.center_x, reverse=True)).strip()

    @property
    def text_ltr(self) -> str:
        return " ".join(token.text.strip() for token in sorted(self.tokens, key=lambda item: item.center_x)).strip()

    @property
    def confidence(self) -> float:
        return aggregate_confidence(token.confidence for token in self.tokens)

    @property
    def center_y(self) -> float:
        return float(np.mean([token.center_y for token in self.tokens]))

    @property
    def min_y(self) -> float:
        return float(min(token.min_y for token in self.tokens))

    @property
    def max_y(self) -> float:
        return float(max(token.max_y for token in self.tokens))


def normalize_digits(value: str) -> str:
    """Convert Eastern Arabic and Persian digits into Western digits."""
    return value.translate(EASTERN_ARABIC_DIGITS).translate(PERSIAN_DIGITS)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("order_points expects a (4, 2) array.")

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image around its center while preserving full content."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    bound_w = int((height * sin) + (width * cos))
    bound_h = int((height * cos) + (width * sin))

    matrix[0, 2] += (bound_w / 2.0) - center[0]
    matrix[1, 2] += (bound_h / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def aggregate_confidence(scores: Iterable[float]) -> float:
    """Aggregate confidence values as a bounded mean."""
    values = [max(0.0, min(1.0, float(score))) for score in scores]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def group_tokens_by_rows(tokens: Sequence[OCRToken], y_tolerance: float | None = None) -> list[OCRRow]:
    """Merge OCR tokens into reading rows based on vertical proximity."""
    if not tokens:
        return []

    ordered = sorted(tokens, key=lambda token: token.center_y)
    median_height = float(np.median([max(token.height, 1.0) for token in ordered]))
    tolerance = y_tolerance if y_tolerance is not None else max(12.0, median_height * 0.75)

    groups: list[list[OCRToken]] = []
    current: list[OCRToken] = [ordered[0]]
    current_center = ordered[0].center_y

    for token in ordered[1:]:
        if abs(token.center_y - current_center) <= tolerance:
            current.append(token)
            current_center = float(np.mean([item.center_y for item in current]))
        else:
            groups.append(sorted(current, key=lambda item: item.center_x))
            current = [token]
            current_center = token.center_y

    groups.append(sorted(current, key=lambda item: item.center_x))
    rows = [OCRRow(tokens=group) for group in groups]
    return sorted(rows, key=lambda row: row.center_y)


def draw_bounding_boxes(
    image: np.ndarray,
    tokens: Sequence[OCRToken],
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Draw OCR polygons and token indices onto a copy of the image."""
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    for index, token in enumerate(tokens, start=1):
        polygon = np.asarray(token.box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        label_point = tuple(np.asarray(token.box[0], dtype=np.int32))
        cv2.putText(
            annotated,
            str(index),
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if output_path is not None:
        output = Path(output_path)
        ensure_directory(output.parent)
        cv2.imwrite(str(output), annotated)

    return annotated


def export_results_to_json(result: dict, output_path: str | Path) -> Path:
    """Write extraction results to disk as UTF-8 JSON."""
    output = Path(output_path)
    ensure_directory(output.parent)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def polygon_to_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert PaddleOCR polygon output into a float32 array."""
    return np.asarray(points, dtype=np.float32).reshape(4, 2)


def clipped_ratio(numerator: float, denominator: float) -> float:
    """Return a bounded ratio and avoid division by zero."""
    if denominator == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def euclidean_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    return math.dist(point_a, point_b)

