"""
Ellipse-based circumference calculations.

A body cross-section is approximated as an ellipse:
  semi-axis a = half the frontal width  (measured from the photo silhouette)
  semi-axis b = half the depth          (a x depth ratio, or from a side photo)

Perimeter uses Ramanujan's second approximation (error < 1e-5 relative).

DEPTH_RATIOS are depth/width ratios of the *full* axes, which equals the
ratio of the semi-axes, so b = a * ratio. They come from ANSUR II and NHANES
cross-sectional data. Female hips carry a higher ratio because the pelvis
protrudes further posteriorly; male chest is rounder than female chest.
"""

from __future__ import annotations

import math

DEPTH_RATIOS: dict[str, dict[str, float]] = {
    "female": {
        "chest": 0.74,
        "waist": 0.70,
        "hips":  0.90,
        "neck":  0.90,
        "thigh": 0.92,
        "calf":  0.88,
        "wrist": 0.68,
        "upper_arm": 0.95,
    },
    "male": {
        "chest": 0.80,
        "waist": 0.78,
        "hips":  0.76,
        "neck":  0.90,
        "thigh": 0.92,
        "calf":  0.88,
        "wrist": 0.68,
        "upper_arm": 0.95,
    },
}


def ellipse_circumference(a: float, b: float) -> float:
    """Ramanujan's approximation of an ellipse perimeter. a, b are semi-axes."""
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def width_to_girth(
    width_cm: float,
    zone: str,
    gender: str,
    depth_cm: float | None = None,
) -> float:
    """
    Convert a frontal width (full left-to-right span, in cm) into a girth in cm.

    depth_cm, when given, is the full front-to-back depth from a side photo and
    overrides the tabulated ratio.
    """
    if width_cm <= 0:
        return 0.0
    a = width_cm / 2.0
    if depth_cm is not None and depth_cm > 0:
        b = depth_cm / 2.0
    else:
        ratio = DEPTH_RATIOS.get(gender, DEPTH_RATIOS["female"]).get(zone, 0.80)
        b = a * ratio
    return ellipse_circumference(a, b)


def girth_to_width(girth_cm: float, zone: str, gender: str) -> float:
    """
    Inverse of width_to_girth with the tabulated ratio. Used to turn a
    statistical girth into the frontal width the photo should have shown, so
    photo and stats can be compared on the same scale.
    """
    if girth_cm <= 0:
        return 0.0
    ratio = DEPTH_RATIOS.get(gender, DEPTH_RATIOS["female"]).get(zone, 0.80)
    # For a fixed ratio the perimeter is linear in a, so one probe gives the slope.
    unit = ellipse_circumference(1.0, ratio)
    return 2.0 * girth_cm / unit
