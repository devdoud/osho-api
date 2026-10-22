"""
Anthropometric regression: body measurements from height, weight, gender, age.

Reference model built from:
  - ANSUR II (US Army, 4,082 males / 1,986 females)
  - NHANES body measurement data
  - West African morphology adjustments (SizeAfrica 2022): female hips +3 cm,
    female chest +1 cm and male shoulder +1 cm at baseline, reflecting the
    higher hip-to-waist ratio typical of West African populations.

Each field is one tuple so a baseline can never drift out of step with its
sensitivities, and fields absent for a gender simply are not listed.

All values in centimetres.
"""

from __future__ import annotations

# Mid-section measurements that keep expanding gradually past 35.
_AGE_SENSITIVE = ("waist", "hips", "chest", "thigh")
_AGE_THRESHOLD = 35
_AGE_CM_PER_YEAR = 0.06

# field: (baseline cm at the reference body, cm per cm of height, cm per kg of weight)
_BASELINE: dict[str, dict] = {
    "female": {
        "ref_height": 165.0,
        "ref_weight": 62.0,
        "fields": {
            "chest":          (91.0, 0.20, 0.45),
            "waist":          (74.0, 0.15, 0.55),
            "hips":          (101.0, 0.25, 0.52),
            "neck":           (33.5, 0.08, 0.12),
            "thigh":          (57.0, 0.12, 0.32),
            "calf":           (36.0, 0.08, 0.18),
            "wrist":          (15.5, 0.04, 0.08),
            "upper_arm":      (28.0, 0.06, 0.30),
            "shoulder":       (38.5, 0.25, 0.08),
            "inseam":         (76.0, 0.55, 0.03),
            "back_length":    (40.0, 0.28, 0.00),
            "sleeve":         (59.0, 0.38, 0.03),
            "bust_height":    (26.5, 0.14, 0.04),
            "crotch_depth":   (28.0, 0.12, 0.10),
            "dress_length":   (98.0, 0.65, 0.00),
            "trouser_length": (98.0, 0.65, 0.00),
        },
    },
    "male": {
        "ref_height": 175.0,
        "ref_weight": 75.0,
        "fields": {
            "chest":         (100.0, 0.22, 0.42),
            "waist":          (84.0, 0.18, 0.58),
            "hips":           (98.0, 0.20, 0.38),
            "neck":           (37.5, 0.10, 0.14),
            "thigh":          (58.0, 0.10, 0.28),
            "calf":           (38.0, 0.08, 0.15),
            "wrist":          (17.0, 0.04, 0.07),
            "upper_arm":      (32.0, 0.07, 0.28),
            "shoulder":       (45.0, 0.30, 0.10),
            "inseam":         (81.0, 0.58, 0.02),
            "back_length":    (43.0, 0.30, 0.00),
            "sleeve":         (65.0, 0.42, 0.02),
            "bust_height":    (24.0, 0.14, 0.03),
            "crotch_depth":   (30.0, 0.13, 0.09),
            "trouser_length": (104.0, 0.68, 0.00),
        },
    },
}


def estimate(
    gender: str,
    height: float,
    weight: float,
    age: int = 28,
) -> dict[str, float]:
    """Body measurements estimated from anthropometric data alone."""
    profile = _BASELINE.get(gender) or _BASELINE["female"]
    dh = height - profile["ref_height"]
    dw = weight - profile["ref_weight"]
    extra = max(0, (age or 28) - _AGE_THRESHOLD) * _AGE_CM_PER_YEAR

    result: dict[str, float] = {}
    for field, (base, h_sens, w_sens) in profile["fields"].items():
        value = base + h_sens * dh + w_sens * dw
        if field in _AGE_SENSITIVE:
            value += extra
        result[field] = round(max(value, 10.0), 1)
    return result


def fields_for(gender: str) -> tuple[str, ...]:
    profile = _BASELINE.get(gender) or _BASELINE["female"]
    return tuple(profile["fields"])
