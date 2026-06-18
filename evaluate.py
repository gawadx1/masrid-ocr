from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer

try:
    import evaluate
except ImportError:  # pragma: no cover - only used when evaluate is unavailable
    evaluate = None


def compute_cer(references: list[str], predictions: list[str]) -> float:
    """Compute character error rate with evaluate when available, else jiwer."""
    if evaluate is not None:
        cer_metric = evaluate.load("cer")
        return float(cer_metric.compute(references=references, predictions=predictions))
    return float(jiwer.cer(references, predictions))


def compute_wer(references: list[str], predictions: list[str]) -> float:
    """Compute word error rate with evaluate when available, else jiwer."""
    if evaluate is not None:
        wer_metric = evaluate.load("wer")
        return float(wer_metric.compute(references=references, predictions=predictions))
    return float(jiwer.wer(references, predictions))


def load_pairs(path: str | Path) -> tuple[list[str], list[str]]:
    """Load reference and prediction pairs from a JSON file."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of objects.")

    references: list[str] = []
    predictions: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each item must be an object with reference and prediction keys.")
        references.append(str(record.get("reference", "")))
        predictions.append(str(record.get("prediction", "")))
    return references, predictions


def evaluate_pairs(references: list[str], predictions: list[str]) -> dict[str, float | int]:
    """Compute CER and WER for aligned reference and prediction lists."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length.")

    return {
        "cer": round(compute_cer(references, predictions), 6),
        "wer": round(compute_wer(references, predictions), 6),
        "samples": len(references),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR output with CER and WER.")
    parser.add_argument("--input", required=True, help="Path to a JSON file with reference/prediction pairs.")
    args = parser.parse_args()

    references, predictions = load_pairs(args.input)
    metrics = evaluate_pairs(references, predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
