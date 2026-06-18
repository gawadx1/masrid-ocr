from __future__ import annotations

from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.ocr import get_ocr_service
from app.postprocess import build_extraction_response
from app.preprocess import preprocess_image
from app.schemas import ExtractionResponse, HealthResponse
from app.utils import draw_bounding_boxes, ensure_directory, export_results_to_json


app = FastAPI(
    title="Egyptian National ID OCR",
    description="OCR API for extracting Arabic name, Arabic address, and 14-digit national ID numbers.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health endpoint."""
    return HealthResponse()


@app.post("/extract", response_model=ExtractionResponse)
async def extract(
    image: UploadFile = File(..., description="Egyptian national ID image."),
    annotate: bool = Query(default=False, description="Save an annotated OCR image to outputs/."),
    export_json: bool = Query(default=False, description="Save the OCR result JSON to outputs/."),
) -> ExtractionResponse:
    """Extract target fields from an uploaded Egyptian ID card image."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        started_at = perf_counter()
        artifacts = preprocess_image(payload)
        tokens = get_ocr_service().extract_tokens(artifacts["card"])
        result = build_extraction_response(tokens)

        if annotate or export_json:
            output_dir = ensure_directory("outputs")
            stem = Path(image.filename or "image").stem
            if annotate:
                draw_bounding_boxes(artifacts["card"], tokens, output_dir / f"{stem}_annotated.jpg")
            if export_json:
                export_results_to_json(result.model_dump(), output_dir / f"{stem}_result.json")

        _ = perf_counter() - started_at
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {exc}") from exc

