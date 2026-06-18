from __future__ import annotations

from typing import Sequence

import numpy as np

from app.utils import OCRToken, polygon_to_array

try:
    from paddleocr import PaddleOCR
except ImportError as exc:  # pragma: no cover - import guarded for runtime environments
    PaddleOCR = None
    PADDLE_IMPORT_ERROR = exc
else:
    PADDLE_IMPORT_ERROR = None


class OCRService:
    """Wrapper around PaddleOCR for Arabic document recognition."""

    def __init__(self, lang: str = "ar", use_angle_cls: bool = True, show_log: bool = False) -> None:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed.") from PADDLE_IMPORT_ERROR
        self.engine = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, show_log=show_log)

    def extract_tokens(self, image: np.ndarray) -> list[OCRToken]:
        """Run OCR and return structured token results."""
        raw_result = self.engine.ocr(image, cls=True)
        tokens: list[OCRToken] = []

        for page in raw_result or []:
            if page is None:
                continue
            for item in page:
                if len(item) != 2:
                    continue
                box, payload = item
                if not payload:
                    continue
                text, confidence = payload
                tokens.append(
                    OCRToken(
                        text=str(text).strip(),
                        confidence=float(confidence),
                        box=polygon_to_array(box),
                    )
                )
        return tokens

    def extract_text(self, image: np.ndarray) -> list[str]:
        """Return recognized text strings."""
        return [token.text for token in self.extract_tokens(image)]

    def extract_boxes(self, image: np.ndarray) -> list[list[list[float]]]:
        """Return OCR polygon boxes."""
        return [token.box.astype(float).tolist() for token in self.extract_tokens(image)]

    def extract_all_text(self, image: np.ndarray, separator: str = "\n") -> str:
        """Return all OCR text concatenated as a single string."""
        return separator.join(self.extract_text(image))


_OCR_SERVICE: OCRService | None = None


def get_ocr_service() -> OCRService:
    """Return a cached OCR service instance."""
    global _OCR_SERVICE
    if _OCR_SERVICE is None:
        _OCR_SERVICE = OCRService()
    return _OCR_SERVICE


def extract_text(image: np.ndarray) -> list[str]:
    """Module-level shortcut for text extraction."""
    return get_ocr_service().extract_text(image)


def extract_boxes(image: np.ndarray) -> list[list[list[float]]]:
    """Module-level shortcut for box extraction."""
    return get_ocr_service().extract_boxes(image)


def extract_all_text(image: np.ndarray, separator: str = "\n") -> str:
    """Module-level shortcut for extracting all OCR text."""
    return get_ocr_service().extract_all_text(image, separator=separator)

