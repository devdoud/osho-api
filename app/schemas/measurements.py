from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Roughly 8 MB of binary, which is far above any sane phone JPEG.
_MAX_PHOTO_CHARS = 11_000_000
_MIN_PHOTO_CHARS = 512


class Gender(str, Enum):
    male = "male"
    female = "female"


class MeasurementRequest(BaseModel):
    gender: Gender
    height: float = Field(..., ge=100, le=250, description="Height in cm")
    weight: float = Field(..., ge=30, le=300, description="Weight in kg")
    age: Optional[int] = Field(None, ge=10, le=120, description="Age in years")
    front_photo: Optional[str] = Field(
        None,
        description=(
            "Front-facing photo, base64 JPEG or PNG, with or without a data: "
            "prefix. The subject must stand straight and square to the camera, "
            "arms slightly away from the body so each arm is separate from the "
            "torso, head and feet both inside the frame, plain background."
        ),
    )
    side_photo: Optional[str] = Field(
        None,
        description="Right-side photo. Accepted but not yet used.",
    )

    @field_validator("front_photo", "side_photo", mode="before")
    @classmethod
    def normalise_photo(cls, v):
        """Accept raw base64 as well as data:image/...;base64,... URIs."""
        if not v:
            return None
        if not isinstance(v, str):
            raise ValueError("photo must be a base64 string")
        if v.startswith("data:") and "," in v:
            v = v.split(",", 1)[1]
        v = v.strip()
        if not v:
            return None
        if len(v) > _MAX_PHOTO_CHARS:
            raise ValueError("photo too large (max ~8 MB)")
        if len(v) < _MIN_PHOTO_CHARS:
            raise ValueError("photo too small to contain an image")
        return v


class MeasurementResult(BaseModel):
    # ── Circumferences (cm) ─────────────────────────────────────────────────
    chest: float = Field(..., description="Chest / bust girth")
    waist: float = Field(..., description="Natural waist girth")
    hips: float = Field(..., description="Hip girth, fullest point")
    neck: float = Field(..., description="Neck base girth")
    thigh: float = Field(..., description="Upper thigh girth")
    calf: float = Field(..., description="Calf girth, fullest point")
    wrist: float = Field(..., description="Wrist girth")
    upper_arm: float = Field(..., description="Biceps girth")

    # ── Width (cm) ──────────────────────────────────────────────────────────
    shoulder: float = Field(..., description="Shoulder point to shoulder point")

    # ── Lengths (cm) ────────────────────────────────────────────────────────
    inseam: float = Field(..., description="Crotch to ankle")
    back_length: float = Field(..., description="Nape to natural waist")
    sleeve: float = Field(..., description="Shoulder point to wrist")
    bust_height: float = Field(..., description="Shoulder to bust point")
    crotch_depth: float = Field(..., description="Waist to crotch, body rise")
    trouser_length: float = Field(..., description="Waist to ankle, outside leg")
    dress_length: Optional[float] = Field(
        None, description="Shoulder to hem, female only"
    )

    # ── Metadata ────────────────────────────────────────────────────────────
    mode: str = Field(..., description="'photo' | 'stats' | 'stats_fallback'")
    confidence: float = Field(..., ge=0, le=1, description="Estimate confidence")
    quality: float = Field(0, ge=0, le=1, description="Photo quality, 0 if unused")
    photos_used: int = Field(0, description="Photos successfully analysed")
    warnings: list[str] = Field(
        default_factory=list,
        description="Why a photo was downgraded or ignored, for user feedback",
    )
