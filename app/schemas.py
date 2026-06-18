from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtractionResponse(BaseModel):
    """Normalized OCR response returned by the API."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(default="", description="Extracted Arabic full name.")
    address: str = Field(default="", description="Extracted Arabic address.")
    national_id: str = Field(
        default="",
        description="Egyptian national ID number.",
        pattern=r"^(\d{14})?$",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Simple health-check payload."""

    status: str = "ok"

