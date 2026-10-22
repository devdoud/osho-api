"""
Photo analysis: MediaPipe pose landmarks + person silhouette -> body measurements.

Division of labour, which is what makes the numbers trustworthy:

  * Landmarks give *positions* - joint centres. They are reliable for the
    biacromial shoulder span, for body levels (where the chest, waist, hips,
    thigh and calf sit) and for lengths (torso, leg, arm).
  * The silhouette mask gives *widths* - the real left-to-right extent of the
    body at a level, and of each leg separately. A girth can only come from
    here; a span between two joint centres is not a body width.
  * Girths follow from width via the ellipse model in geometry.py.

Every width is cross-checked against the width the statistical model predicts
for this subject (scaled by their measured frame), so a merged arm, a failed
mask or a cropped frame is dropped instead of being reported with confidence.

Calibration uses eye level to ankle, which is 0.897 of stature in the Drillis &
Contini segment proportions. The crown of the head is deliberately not used:
it is not a landmark, and estimating it from the nose-to-ear offset collapses
when the subject faces the camera square-on.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from . import segmentation_service as seg
from .geometry import girth_to_width, width_to_girth
from .model_store import ensure_model

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_MODEL_FILE = "pose_landmarker_full.task"

try:
    import mediapipe as _mp
except ImportError:  # pragma: no cover - mediapipe is a hard dependency in prod
    _mp = None

# ── Landmark indices (33-point BlazePose topology) ────────────────────────────
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

_HEAD_POINTS = (L_EYE, R_EYE)
_FOOT_POINTS = (L_ANKLE, R_ANKLE)
_FRAME_POINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE)

# Eye level minus ankle level as a fraction of stature.
_EYE_TO_ANKLE = 0.936 - 0.039

# Biacromial breadth as a fraction of stature (ANSUR II means), for judging
# frame size and how square-on the subject stands.
_EXPECTED_SHOULDER_RATIO = {"female": 0.223, "male": 0.226}

# Biacromial breadth to garment shoulder width, shoulder point to shoulder
# point. Larger for men, whose deltoids sit further outboard of the acromion.
_SHOULDER_GARMENT_FACTOR = {"female": 1.06, "male": 1.13}

# Measurement levels, as a fraction of the shoulder-to-hip span going down.
_FRAC_NECK = -0.13  # above the shoulder line
_FRAC_CHEST = 0.34
_FRAC_WAIST = 0.69
_FRAC_HIPS = 1.03

# Levels as a fraction of the hip-to-ankle span.
_FRAC_THIGH = 0.13
_FRAC_CALF = 0.67

# A mask width this far from the statistically predicted width is not a
# measurement of the body: an arm merged into the torso, or a broken mask.
_WIDTH_TOLERANCE_LOW = 0.65
_WIDTH_TOLERANCE_HIGH = 1.40

_MAX_IMAGE_SIDE = 1280

_GIRTH_ZONES = ("chest", "waist", "hips", "neck", "thigh", "calf", "upper_arm")


@dataclass
class PhotoAnalysis:
    measurements: dict[str, float]
    sources: dict[str, str]  # field -> 'mask' | 'landmark'
    quality: float
    frame_scale: float = 1.0
    warnings: list[str] = field(default_factory=list)


def _rejected(reason: str) -> PhotoAnalysis:
    """An unusable photo, carrying why — the app shows this to the user."""
    logger.info("Photo rejected: %s", reason)
    return PhotoAnalysis(measurements={}, sources={}, quality=0.0, warnings=[reason])


# ── Image loading ─────────────────────────────────────────────────────────────

def _decode_b64(payload: str) -> bytes | None:
    s = "".join(payload.split()).replace("-", "+").replace("_", "/")
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s)
    except (binascii.Error, ValueError):
        return None


def _load_image(payload: str) -> np.ndarray | None:
    data = _decode_b64(payload)
    if not data:
        return None
    try:
        pil = Image.open(io.BytesIO(data))
        # Phones store portrait shots rotated with an EXIF orientation tag.
        # Without this the body is analysed sideways and every number is wrong.
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as exc:
        logger.info("Cannot decode image: %s", exc)
        return None

    w, h = pil.size
    scale = min(1.0, _MAX_IMAGE_SIDE / max(w, h))
    if scale < 1.0:
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


# ── Detector ──────────────────────────────────────────────────────────────────

_detector = None
_detector_ready = False


def _get_detector():
    global _detector, _detector_ready
    if _detector_ready:
        return _detector
    _detector_ready = True

    if _mp is None:
        return None
    model_path = ensure_model(_MODEL_URL, _MODEL_FILE)
    if model_path is None:
        return None
    try:
        vision = _mp.tasks.vision
        _detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=_mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                output_segmentation_masks=True,
            )
        )
    except Exception as exc:
        logger.warning("PoseLandmarker unavailable: %s", exc)
        _detector = None
    return _detector


def _detect(arr: np.ndarray):
    det = _get_detector()
    if det is None:
        return None, None
    try:
        mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=arr)
        result = det.detect(mp_img)
    except Exception as exc:
        logger.warning("Pose detection failed: %s", exc)
        return None, None

    if not result.pose_landmarks:
        return None, None

    h, w = arr.shape[:2]
    points = result.pose_landmarks[0]
    lm = {
        i: (p.x * w, p.y * h, float(getattr(p, "visibility", 1.0) or 0.0))
        for i, p in enumerate(points)
    }
    if len(lm) < 29:
        return None, None

    mask = None
    masks = getattr(result, "segmentation_masks", None)
    if masks:
        try:
            mask = seg.to_bool_mask(masks[0].numpy_view())
        except Exception as exc:
            logger.info("Segmentation mask unusable: %s", exc)
    return lm, mask


# ── Small helpers ─────────────────────────────────────────────────────────────

def _dist(lm: dict, a: int, b: int) -> float:
    return math.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1])


def _mid_y(lm: dict, a: int, b: int) -> float:
    return (lm[a][1] + lm[b][1]) / 2.0


def _mid_x(lm: dict, a: int, b: int) -> float:
    return (lm[a][0] + lm[b][0]) / 2.0


def _visibility(lm: dict, idx) -> float:
    return sum(lm[i][2] for i in idx) / len(idx)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_front(
    payload: str,
    height_cm: float,
    gender: str,
    stats: dict[str, float],
) -> PhotoAnalysis | None:
    """
    Measure a front-facing photo.

    Returns None only when photo analysis is not installed. Otherwise a
    PhotoAnalysis whose `quality` says how much to trust it, and whose
    `warnings` name the reason when quality is zero.

    `stats` is the statistical estimate for this subject, used only to sanity
    check measured widths and to scale the fields the photo cannot see.
    """
    if _mp is None:
        return None

    arr = _load_image(payload)
    if arr is None:
        return _rejected("photo_unreadable")

    lm, mask = _detect(arr)
    if lm is None:
        return _rejected("no_body_detected")

    warnings: list[str] = []

    # MediaPipe extrapolates landmarks that fall outside the frame, so a photo
    # cropped at the knees still "succeeds". Calibration spans eyes to ankles,
    # so both ends must be visible — averaging would let a clear head hide
    # missing feet.
    calib_vis = min(_visibility(lm, _HEAD_POINTS), _visibility(lm, _FOOT_POINTS))
    if calib_vis < 0.5:
        return _rejected("body_not_fully_visible")

    px_span = _mid_y(lm, L_ANKLE, R_ANKLE) - _mid_y(lm, L_EYE, R_EYE)
    if px_span <= 0:
        return _rejected("subject_not_upright")
    px_per_cm = px_span / (height_cm * _EYE_TO_ANKLE)

    def to_cm(px: float) -> float:
        return px / px_per_cm

    shoulder_y = _mid_y(lm, L_SHOULDER, R_SHOULDER)
    hip_y = _mid_y(lm, L_HIP, R_HIP)
    ankle_y = _mid_y(lm, L_ANKLE, R_ANKLE)
    knee_y = _mid_y(lm, L_KNEE, R_KNEE)
    torso_px = hip_y - shoulder_y
    leg_px = ankle_y - hip_y
    if torso_px <= 0 or leg_px <= 0:
        return _rejected("subject_not_upright")

    # ── Frame size and how square-on the subject stands ──────────────────────
    shoulder_span_cm = to_cm(_dist(lm, L_SHOULDER, R_SHOULDER))
    expected_ratio = _EXPECTED_SHOULDER_RATIO.get(gender, 0.223)
    frame_scale = shoulder_span_cm / (height_cm * expected_ratio)

    if frame_scale < 0.78:
        # A biacromial span this short means the shoulders are foreshortened:
        # the subject is turned away from the camera, so no width is measurable.
        return _rejected("turned_away_from_camera")
    frame_scale = min(max(frame_scale, 0.82), 1.22)

    tilt = abs(lm[L_SHOULDER][1] - lm[R_SHOULDER][1]) / torso_px
    if tilt > 0.12:
        warnings.append("shoulders_not_level")

    measurements: dict[str, float] = {}
    sources: dict[str, str] = {}

    # ── Widths from the silhouette ───────────────────────────────────────────
    def centre_x_at(y: float) -> float:
        t = (y - shoulder_y) / torso_px
        return _lerp(_mid_x(lm, L_SHOULDER, R_SHOULDER), _mid_x(lm, L_HIP, R_HIP), t)

    def accept(zone: str, width_cm: float | None) -> float | None:
        """Keep a measured width only if it is near the predicted one."""
        if width_cm is None or width_cm <= 0:
            return None
        predicted = girth_to_width(stats.get(zone, 0.0), zone, gender) * frame_scale
        if predicted <= 0:
            return None
        ratio = width_cm / predicted
        if not _WIDTH_TOLERANCE_LOW <= ratio <= _WIDTH_TOLERANCE_HIGH:
            return None
        return width_cm

    if mask is not None:
        levels = {
            "neck": shoulder_y + _FRAC_NECK * torso_px,
            "chest": shoulder_y + _FRAC_CHEST * torso_px,
            "waist": shoulder_y + _FRAC_WAIST * torso_px,
            "hips": shoulder_y + _FRAC_HIPS * torso_px,
        }
        for zone, y in levels.items():
            raw = seg.torso_width_px(mask, int(round(y)), centre_x_at(y))
            width = accept(zone, to_cm(raw) if raw else None)
            if width is not None:
                measurements[zone] = width_to_girth(width, zone, gender)
                sources[zone] = "mask"

        # Legs: each limb measured on its own centre line.
        for zone, frac in (("thigh", _FRAC_THIGH), ("calf", _FRAC_CALF)):
            y = hip_y + frac * leg_px
            lx = _lerp(lm[L_HIP][0], lm[L_ANKLE][0], frac)
            rx = _lerp(lm[R_HIP][0], lm[R_ANKLE][0], frac)
            widths = [
                w
                for w in (
                    seg.limb_width_px(mask, int(round(y)), lx, rx),
                    seg.limb_width_px(mask, int(round(y)), rx, lx),
                )
                if w
            ]
            if widths:
                width = accept(zone, to_cm(float(np.median(widths))))
                if width is not None:
                    measurements[zone] = width_to_girth(width, zone, gender)
                    sources[zone] = "mask"

        # Upper arm at mid-biceps on the side that is better exposed.
        arm = L_SHOULDER if lm[L_ELBOW][2] >= lm[R_ELBOW][2] else R_SHOULDER
        elbow = L_ELBOW if arm == L_SHOULDER else R_ELBOW
        arm_y = _lerp(lm[arm][1], lm[elbow][1], 0.35)
        arm_x = _lerp(lm[arm][0], lm[elbow][0], 0.35)
        raw = seg.limb_width_px(mask, int(round(arm_y)), arm_x, centre_x_at(arm_y))
        width = accept("upper_arm", to_cm(raw) if raw else None)
        if width is not None:
            measurements["upper_arm"] = width_to_girth(width, "upper_arm", gender)
            sources["upper_arm"] = "mask"
    else:
        warnings.append("silhouette_unavailable")

    # ── Lengths and spans from landmarks ─────────────────────────────────────
    # Shoulder *width* for garments runs to the shoulder points, outboard of
    # the acromion landmarks the pose model reports.
    measurements["shoulder"] = shoulder_span_cm * _SHOULDER_GARMENT_FACTOR.get(
        gender, 1.06
    )
    sources["shoulder"] = "landmark"

    # Nape-to-waist over the curve of the back, from the vertical
    # shoulder-to-hip span.
    measurements["back_length"] = to_cm(torso_px) * 0.84
    # The hip landmark sits at the greater trochanter, above the crotch.
    measurements["inseam"] = to_cm(leg_px) * 0.91
    measurements["sleeve"] = (
        to_cm(_dist(lm, L_SHOULDER, L_ELBOW) + _dist(lm, L_ELBOW, L_WRIST)) * 1.03
    )
    waist_y = shoulder_y + _FRAC_WAIST * torso_px
    measurements["trouser_length"] = to_cm(ankle_y - waist_y) * 1.01
    if gender == "female":
        measurements["dress_length"] = to_cm(knee_y - shoulder_y) * 1.12

    for key in ("back_length", "inseam", "sleeve", "trouser_length", "dress_length"):
        if key in measurements:
            sources[key] = "landmark"

    # ── Quality ──────────────────────────────────────────────────────────────
    frame_vis = _visibility(lm, _FRAME_POINTS)
    quality = 0.45 * calib_vis + 0.25 * frame_vis
    if mask is not None:
        quality += 0.22
    if tilt <= 0.12:
        quality += 0.08
    quality = round(min(max(quality, 0.0), 1.0), 3)

    if not any(s == "mask" for s in sources.values()):
        warnings.append("no_girth_measured_from_photo")

    measurements = {k: round(v, 1) for k, v in measurements.items()}
    return PhotoAnalysis(
        measurements=measurements,
        sources=sources,
        quality=quality,
        frame_scale=round(frame_scale, 3),
        warnings=warnings,
    )


def mediapipe_available() -> bool:
    return _mp is not None and _get_detector() is not None


def girth_zones() -> tuple[str, ...]:
    return _GIRTH_ZONES
