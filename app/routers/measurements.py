"""Body measurement route — fully standalone, no external API."""
from __future__ import annotations

import logging

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.dependencies import TokenPayload, get_current_user
from app.schemas.measurements import MeasurementRequest, MeasurementResult
from app.services import measurement_engine
from app.services.mediapipe_service import mediapipe_available

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/measurements", tags=["Measurements"])


@router.post(
    "",
    response_model=MeasurementResult,
    status_code=status.HTTP_200_OK,
    summary="Estimate body measurements",
    description=(
        "Returns body measurements in cm from height, weight, gender, age and "
        "optionally a front photo.\n\n"
        "**Mode `stats`** (no photo): anthropometric regression, instant.\n\n"
        "**Mode `photo`**: pose landmarks give body levels and lengths, the "
        "person silhouette gives real body widths, and the ellipse model turns "
        "those widths into girths. Every width is cross-checked against the "
        "statistical estimate, so an unusable photo degrades to `stats_fallback` "
        "instead of returning wrong numbers. `warnings` says what went wrong and "
        "`quality` how good the photo was."
    ),
)
async def create_measurement(
    body: MeasurementRequest,
    current_user: TokenPayload = Depends(get_current_user),
    x_user_id: Annotated[Optional[str], Header()] = None,
) -> MeasurementResult:
    # The caller is the Edge Function holding a service token, so the end user
    # it is acting for arrives in a header. Untrusted, used only for logging.
    caller = x_user_id or current_user.sub
    logger.info(
        "Measurement request user=%s gender=%s height=%.0f weight=%.0f photo=%s",
        caller,
        body.gender.value,
        body.height,
        body.weight,
        bool(body.front_photo),
    )

    try:
        result = measurement_engine.run(
            gender=body.gender.value,
            height=body.height,
            weight=body.weight,
            age=body.age or 28,
            front_photo=body.front_photo,
            side_photo=body.side_photo,
        )
    except Exception:
        logger.exception("Measurement engine error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Measurement engine failed. Please retry.",
        )

    logger.info(
        "Measurement done user=%s mode=%s confidence=%.2f warnings=%s",
        caller,
        result["mode"],
        result["confidence"],
        result["warnings"],
    )

    return MeasurementResult.model_validate(
        {
            **result["measurements"],
            "mode": result["mode"],
            "confidence": result["confidence"],
            "quality": result["quality"],
            "photos_used": result["photos_used"],
            "warnings": result["warnings"],
        }
    )


@router.get(
    "/health/mediapipe",
    summary="Check photo-analysis availability",
)
async def mediapipe_status() -> dict:
    ready = mediapipe_available()
    return {
        "available": ready,
        "note": (
            "Photo mode ready."
            if ready
            else "Photo mode disabled; requests fall back to stats mode."
        ),
    }
