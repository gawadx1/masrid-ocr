from app.postprocess import clean_text, validate_id
from app.utils import normalize_digits


def test_normalize_digits_converts_eastern_arabic_digits() -> None:
    assert normalize_digits("١٢٣٤٥٦٧٨٩٠") == "1234567890"


def test_validate_id_accepts_exactly_14_digits() -> None:
    assert validate_id("٢٩٨٠١٠١١٢٣٤٥٦٧")
    assert not validate_id("2980101123456")
    assert not validate_id("2980101123456A")


def test_clean_text_removes_noise_and_duplicate_spaces() -> None:
    dirty = "  الاسم***   محمد---  احمد 123 "
    assert clean_text(dirty, allow_digits=False) == "الاسم محمد احمد"

