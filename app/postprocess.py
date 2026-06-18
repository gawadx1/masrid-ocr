from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas import ExtractionResponse
from app.utils import OCRRow, OCRToken, aggregate_confidence, group_tokens_by_rows, normalize_digits


ARABIC_TEXT_PATTERN = re.compile(r"^[\u0621-\u064A ]+$")
ID_PATTERN = re.compile(r"\b\d{14}\b")
NAME_LABELS = ("الاسم", "اسم", "الاسم بالكامل", "الاسم رباعي")
ADDRESS_LABELS = ("العنوان", "عنوان", "محل الإقامة", "الاقامة", "الإقامة", "السكن")


def clean_text(text: str, allow_digits: bool = False) -> str:
    """Normalize text and remove OCR artifacts while keeping Arabic content."""
    normalized = normalize_digits(text)
    normalized = normalized.replace("ـ", " ")
    allowed = r"\u0621-\u064A0-9\s" if allow_digits else r"\u0621-\u064A\s"
    cleaned = re.sub(fr"[^{allowed}]", " ", normalized)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def validate_name(text: str) -> bool:
    """Validate that a name is Arabic-only with at least two words."""
    candidate = clean_text(text, allow_digits=False)
    if not candidate:
        return False
    if not ARABIC_TEXT_PATTERN.fullmatch(candidate):
        return False
    return len(candidate.split()) >= 2


def validate_id(value: str) -> bool:
    """Validate an Egyptian national ID number."""
    normalized = normalize_digits(value)
    return bool(re.fullmatch(r"^\d{14}$", normalized))


def _ensure_rows(items: Sequence[OCRRow | OCRToken]) -> list[OCRRow]:
    if not items:
        return []
    first_item = items[0]
    if isinstance(first_item, OCRRow):
        return list(items)  # type: ignore[return-value]
    return group_tokens_by_rows(items)  # type: ignore[arg-type]


def _relative_position(rows: Sequence[OCRRow], row: OCRRow) -> float:
    if not rows:
        return 0.0
    min_y = min(item.min_y for item in rows)
    max_y = max(item.max_y for item in rows)
    denominator = max(max_y - min_y, 1.0)
    return (row.center_y - min_y) / denominator


def _row_contains_keywords(row: OCRRow, keywords: Sequence[str]) -> bool:
    row_text = clean_text(row.text, allow_digits=False)
    row_text_ltr = clean_text(row.text_ltr, allow_digits=False)
    return any(keyword in row_text or keyword in row_text_ltr for keyword in keywords)


def _strip_keywords(text: str, keywords: Sequence[str], allow_digits: bool = False) -> str:
    normalized = normalize_digits(text)
    for keyword in keywords:
        normalized = normalized.replace(keyword, " ")
    return clean_text(normalized, allow_digits=allow_digits)


def extract_national_id(items: Sequence[OCRRow | OCRToken]) -> tuple[str, float]:
    """Extract the 14-digit national ID candidate."""
    rows = _ensure_rows(items)
    best_value = ""
    best_score = -1.0
    best_confidence = 0.0

    for row in rows:
        variants = [normalize_digits(row.text), normalize_digits(row.text_ltr)]
        joined_digits = "".join(re.findall(r"\d", " ".join(token.text for token in row.tokens)))
        variants.append(joined_digits)

        for variant in variants:
            matches = ID_PATTERN.findall(variant)
            if not matches and len(variant) == 14 and variant.isdigit():
                matches = [variant]
            for match in matches:
                relative_y = _relative_position(rows, row)
                score = row.confidence + (relative_y * 0.35)
                if score > best_score:
                    best_score = score
                    best_value = match
                    best_confidence = row.confidence

    return best_value if validate_id(best_value) else "", best_confidence if validate_id(best_value) else 0.0


def extract_name(items: Sequence[OCRRow | OCRToken]) -> tuple[str, float]:
    """Extract the Arabic full name using label and position heuristics."""
    rows = _ensure_rows(items)

    for index, row in enumerate(rows):
        if _row_contains_keywords(row, NAME_LABELS):
            inline = _strip_keywords(row.text, NAME_LABELS)
            if validate_name(inline):
                return inline, row.confidence

            for next_row in rows[index + 1 : index + 3]:
                candidate = clean_text(next_row.text, allow_digits=False)
                if validate_name(candidate):
                    return candidate, aggregate_confidence([row.confidence, next_row.confidence])

    candidates: list[tuple[float, str, float]] = []
    for row in rows:
        candidate = clean_text(row.text, allow_digits=False)
        if not validate_name(candidate):
            continue
        if _row_contains_keywords(row, ADDRESS_LABELS):
            continue

        words = len(candidate.split())
        relative_y = _relative_position(rows, row)
        score = row.confidence
        score += 0.20 if 0.05 <= relative_y <= 0.55 else 0.0
        score += 0.10 if 2 <= words <= 5 else 0.0
        candidates.append((score, candidate, row.confidence))

    if not candidates:
        return "", 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_name, confidence = candidates[0]
    return best_name, confidence


def extract_address(items: Sequence[OCRRow | OCRToken]) -> tuple[str, float]:
    """Extract the Arabic address using label and vertical-position heuristics."""
    rows = _ensure_rows(items)

    for index, row in enumerate(rows):
        if _row_contains_keywords(row, ADDRESS_LABELS):
            inline = _strip_keywords(row.text, ADDRESS_LABELS)
            if len(inline.split()) >= 2:
                return inline, row.confidence

            collected: list[str] = []
            confidences = [row.confidence]
            for next_row in rows[index + 1 : index + 3]:
                candidate = clean_text(next_row.text, allow_digits=False)
                if candidate:
                    collected.append(candidate)
                    confidences.append(next_row.confidence)
            merged = clean_text(" ".join(collected), allow_digits=False)
            if len(merged.split()) >= 2:
                return merged, aggregate_confidence(confidences)

    national_id, _ = extract_national_id(rows)
    name, _ = extract_name(rows)

    candidates: list[tuple[float, str, float]] = []
    for row in rows:
        if _row_contains_keywords(row, NAME_LABELS):
            continue

        if _row_contains_keywords(row, ADDRESS_LABELS):
            candidate = _strip_keywords(row.text, ADDRESS_LABELS)
        else:
            candidate = clean_text(row.text, allow_digits=False)

        if len(candidate.split()) < 2:
            continue
        if candidate == name:
            continue
        if national_id and national_id in normalize_digits(row.text):
            continue

        relative_y = _relative_position(rows, row)
        score = row.confidence
        score += 0.15 if 0.25 <= relative_y <= 0.85 else 0.0
        score += 0.10 if len(candidate.split()) >= 3 else 0.0
        candidates.append((score, candidate, row.confidence))

    if not candidates:
        return "", 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_address, confidence = candidates[0]
    return best_address, confidence


def build_extraction_response(items: Sequence[OCRToken | OCRRow]) -> ExtractionResponse:
    """Build the final API response from OCR output."""
    rows = _ensure_rows(items)
    name, name_confidence = extract_name(rows)
    address, address_confidence = extract_address(rows)
    national_id, id_confidence = extract_national_id(rows)

    confidence = aggregate_confidence(
        score for score in [name_confidence, address_confidence, id_confidence] if score > 0
    )

    return ExtractionResponse(
        name=name if validate_name(name) else "",
        address=clean_text(address, allow_digits=False),
        national_id=national_id if validate_id(national_id) else "",
        confidence=confidence,
    )
