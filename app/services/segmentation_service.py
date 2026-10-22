"""
Body-width measurement from a person silhouette mask.

Pose landmarks are joint centres, so they cannot give a body width: the span
between the two shoulder landmarks is the biacromial distance, not the outline
of the torso. PoseLandmarker can also emit a person segmentation mask, and this
module reads real widths off it by scanning single pixel rows.

Row scanning returns *runs*: maximal stretches of consecutive person pixels.
With arms held slightly away from the body, a chest-height row yields
[left arm, torso, right arm]; a thigh-height row yields [left leg, right leg].
"""

from __future__ import annotations

import numpy as np

# Runs thinner than this fraction of the image width are mask speckle, not a limb.
_MIN_RUN_FRAC = 0.012

# Person probability above which a pixel counts as body.
_MASK_THRESHOLD = 0.5


def to_bool_mask(raw: np.ndarray) -> np.ndarray | None:
    """
    Normalise a PoseLandmarker segmentation mask into a boolean array.

    A mask covering nearly none or nearly all of the frame is not a usable
    silhouette, so it is rejected rather than measured.
    """
    if raw is None or raw.ndim != 2:
        return None
    mask = raw > _MASK_THRESHOLD if raw.dtype.kind == "f" else raw.astype(bool)
    coverage = float(mask.mean())
    if not 0.02 <= coverage <= 0.92:
        return None
    return mask


def runs_in_row(mask: np.ndarray, y: int) -> list[tuple[int, int]]:
    """Maximal runs of person pixels in row y, as inclusive (x_start, x_end)."""
    h, w = mask.shape[:2]
    if not 0 <= y < h:
        return []

    # Pad with False so a body touching either edge still produces a boundary.
    edges = np.diff(np.concatenate(([False], mask[y], [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1

    min_len = max(2, int(w * _MIN_RUN_FRAC))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_len]


def _smoothed(fn, y: int, band: int) -> float | None:
    """Median of fn over three rows, to shrug off per-row mask noise."""
    values = [v for dy in (-band, 0, band) if (v := fn(y + dy)) is not None]
    return float(np.median(values)) if values else None


def torso_width_px(
    mask: np.ndarray, y: int, centre_x: float, band: int = 3
) -> float | None:
    """
    Width of the run containing the body midline at height y.

    At chest and waist height the arms appear as their own runs on either side;
    picking the run that spans centre_x discards them.
    """

    def once(yy: int) -> float | None:
        runs = runs_in_row(mask, yy)
        if not runs:
            return None
        inside = [r for r in runs if r[0] <= centre_x <= r[1]]
        chosen = inside[0] if inside else max(runs, key=lambda r: r[1] - r[0])
        return float(chosen[1] - chosen[0] + 1)

    return _smoothed(once, y, band)


def limb_width_px(
    mask: np.ndarray,
    y: int,
    want_x: float,
    other_x: float,
    band: int = 3,
) -> float | None:
    """
    Width of one limb at height y, matched to its expected centre line.

    When both limbs fall inside a single run they are touching, and the run is
    halved.
    """

    def once(yy: int) -> float | None:
        runs = runs_in_row(mask, yy)
        if not runs:
            return None
        if len(runs) == 1:
            s, e = runs[0]
            span = float(e - s + 1)
            covers_both = s <= want_x <= e and s <= other_x <= e
            return span / 2.0 if covers_both else span
        nearest = min(runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - want_x))
        return float(nearest[1] - nearest[0] + 1)

    return _smoothed(once, y, band)
