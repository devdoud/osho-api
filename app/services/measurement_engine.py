"""
Measurement engine: combines the statistical estimate with photo analysis.

The statistical estimate is always computed, because it is the prior that makes
the photo safe to use: every width the photo measures is checked against it,
and anything the photo cannot see falls back to it.

When a usable photo is present, each field is a weighted average of the two.
The weight on the photo depends on how the field was obtained and on the
photo's quality score, so a half-usable photo degrades smoothly towards the
statistical estimate instead of either being trusted blindly or thrown away.
"""

from __future__ import annotations

import logging

from .mediapipe_service import analyze_front, girth_zones, mediapipe_available
from .stats_service import estimate as stats_estimate

logger = logging.getLogger(__name__)

# Below this quality the photo tells us nothing reliable.
_MIN_QUALITY = 0.45

# A frame scale measured at the shoulders only partly carries to the waist and
# legs, so it is damped before being applied to girths the photo never saw.
_FRAME_SCALE_DAMPING = 0.5

# Anatomically possible range for each field, in cm.
_CLAMPS: dict[str, tuple[float, float]] = {
    "chest":          (60, 180),
    "waist":          (48, 170),
    "hips":           (65, 190),
    "neck":           (26, 60),
    "thigh":          (35, 100),
    "calf":           (24, 70),
    "wrist":          (12, 28),
    "upper_arm":      (18, 60),
    "shoulder":       (28, 65),
    "inseam":         (50, 110),
    "back_length":    (28, 60),
    "sleeve":         (40, 90),
    "bust_height":    (16, 45),
    "crotch_depth":   (16, 45),
    "dress_length":   (60, 150),
    "trouser_length": (70, 145),
}

# Minimum rise between the inseam and the outside leg length.
_MIN_RISE = 18.0


def _photo_weight(field: str, source: str, quality: float) -> float:
    if source == "mask":
        # A silhouette width is a direct observation of the body.
        return 0.55 + 0.35 * quality
    if field == "shoulder":
        # The biacromial span is exactly what pose landmarks measure well.
        return 0.55 + 0.30 * quality
    return 0.40 + 0.30 * quality


def run(
    gender: str,
    height: float,
    weight: float,
    age: int = 28,
    front_photo: str | None = None,
    side_photo: str | None = None,
) -> dict:
    """
    Returns:
      measurements : dict[str, float]  - all values in cm
      mode         : 'photo' | 'stats' | 'stats_fallback'
      confidence   : float 0-1
      quality      : float 0-1, photo quality (0 when no photo was used)
      photos_used  : int
      warnings     : list[str]
    """
    stats = stats_estimate(gender, height, weight, age)
    warnings: list[str] = []
    analysis = None

    if front_photo:
        if mediapipe_available():
            analysis = analyze_front(front_photo, height, gender, stats)
            if analysis is None:
                warnings.append("photo_analysis_unavailable")
            elif analysis.quality < _MIN_QUALITY:
                # analyze_front names the specific reason it could not be used.
                warnings.extend(analysis.warnings or ["photo_quality_too_low"])
                analysis = None
        else:
            warnings.append("photo_analysis_unavailable")

    if side_photo:
        # Depth from a side photo is not implemented yet; the ellipse model
        # uses tabulated depth ratios instead.
        warnings.append("side_photo_ignored")

    if analysis is None:
        measurements = dict(stats)
        mode = "stats_fallback" if front_photo else "stats"
        confidence = 0.70
        quality = 0.0
        photos_used = 0
    else:
        warnings.extend(analysis.warnings)
        quality = analysis.quality
        photos_used = 1
        mode = "photo"

        measured = analysis.measurements
        sources = analysis.sources
        # Damped frame scale for girths the photo could not measure.
        scale = 1.0 + (analysis.frame_scale - 1.0) * _FRAME_SCALE_DAMPING

        measurements = {}
        for field, stat_value in stats.items():
            photo_value = measured.get(field)
            if photo_value is None:
                if field in girth_zones():
                    measurements[field] = stat_value * scale
                else:
                    measurements[field] = stat_value
                continue
            w = _photo_weight(field, sources.get(field, "landmark"), quality)
            measurements[field] = w * photo_value + (1.0 - w) * stat_value

        n_mask = sum(1 for s in sources.values() if s == "mask")
        confidence = 0.70 + 0.25 * quality
        if n_mask == 0:
            confidence = min(confidence, 0.78)

    for field, (low, high) in _CLAMPS.items():
        if field in measurements:
            measurements[field] = min(max(measurements[field], low), high)

    # The outside leg length must clear the inseam by at least a rise, or the
    # pair of numbers cannot describe a real pair of trousers.
    inseam = measurements.get("inseam")
    trouser = measurements.get("trouser_length")
    if inseam is not None and trouser is not None and trouser - inseam < _MIN_RISE:
        measurements["trouser_length"] = inseam + _MIN_RISE

    measurements = {k: round(v, 1) for k, v in measurements.items()}

    return {
        "measurements": measurements,
        "mode": mode,
        "confidence": round(confidence, 2),
        "quality": round(quality, 2),
        "photos_used": photos_used,
        "warnings": sorted(set(warnings)),
    }
