"""
End-to-end test of the photo pipeline against a synthetic body of known size.

A real photo cannot be asserted on, so pose detection is replaced by landmarks
and a silhouette built from published anthropometry for a 165 cm / 62 kg woman
at a known pixel scale. If calibration, level placement, silhouette scanning,
the ellipse model and blending are all correct, the engine must recover the
girths the silhouette was drawn from.
"""
import base64, io, os, sys
from pathlib import Path
os.environ["DEBUG"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from app.services import mediapipe_service as ms
from app.services import measurement_engine as engine
from app.services.geometry import girth_to_width
from app.services.stats_service import estimate

HEIGHT_CM, WEIGHT_KG, GENDER = 165.0, 62.0, "female"
PX_PER_CM = 4.0
IMG_W, IMG_H = 500, 700
CENTRE_X = 250.0

stature_px = HEIGHT_CM * PX_PER_CM
EYE_Y = 40.0
FLOOR_Y = EYE_Y + 0.936 * stature_px


def y_at(frac: float) -> float:
    """Image row for a height given as a fraction of stature above the floor."""
    return FLOOR_Y - frac * stature_px


def cm(px: float) -> float:
    return px / PX_PER_CM


def px(c: float) -> float:
    return c * PX_PER_CM


shoulder_y, hip_y = y_at(0.818), y_at(0.530)
elbow_y, wrist_y = y_at(0.630), y_at(0.485)
knee_y, ankle_y = y_at(0.285), y_at(0.039)
torso_px, leg_px = hip_y - shoulder_y, ankle_y - hip_y

shoulder_half = px(HEIGHT_CM * ms._EXPECTED_SHOULDER_RATIO[GENDER]) / 2
hip_half, knee_half, ankle_half = px(29) / 2, px(11) / 2, px(10) / 2
elbow_x_off, wrist_x_off = 125.0, 145.0

stats = estimate(GENDER, HEIGHT_CM, WEIGHT_KG, 28)

# Widths the silhouette is drawn from, taken from the statistical girths.
target_w = {z: px(girth_to_width(stats[z], z, GENDER))
            for z in ("neck", "chest", "waist", "hips", "thigh", "calf", "upper_arm")}

level_y = {
    "neck":  shoulder_y + ms._FRAC_NECK * torso_px,
    "chest": shoulder_y + ms._FRAC_CHEST * torso_px,
    "waist": shoulder_y + ms._FRAC_WAIST * torso_px,
    "hips":  shoulder_y + ms._FRAC_HIPS * torso_px,
}
thigh_y = hip_y + ms._FRAC_THIGH * leg_px
calf_y = hip_y + ms._FRAC_CALF * leg_px

# ── Landmarks ────────────────────────────────────────────────────────────────
lm = {i: (CENTRE_X, 0.0, 1.0) for i in range(33)}
lm[ms.NOSE] = (CENTRE_X, EYE_Y + 12, 1.0)
lm[ms.L_EYE] = (CENTRE_X - 10, EYE_Y, 1.0)
lm[ms.R_EYE] = (CENTRE_X + 10, EYE_Y, 1.0)
lm[ms.L_EAR] = (CENTRE_X - 30, EYE_Y + 4, 1.0)
lm[ms.R_EAR] = (CENTRE_X + 30, EYE_Y + 4, 1.0)
lm[ms.L_SHOULDER] = (CENTRE_X - shoulder_half, shoulder_y, 1.0)
lm[ms.R_SHOULDER] = (CENTRE_X + shoulder_half, shoulder_y, 1.0)
lm[ms.L_ELBOW] = (CENTRE_X - elbow_x_off, elbow_y, 1.0)
lm[ms.R_ELBOW] = (CENTRE_X + elbow_x_off, elbow_y, 1.0)
lm[ms.L_WRIST] = (CENTRE_X - wrist_x_off, wrist_y, 1.0)
lm[ms.R_WRIST] = (CENTRE_X + wrist_x_off, wrist_y, 1.0)
lm[ms.L_HIP] = (CENTRE_X - hip_half, hip_y, 1.0)
lm[ms.R_HIP] = (CENTRE_X + hip_half, hip_y, 1.0)
lm[ms.L_KNEE] = (CENTRE_X - knee_half, knee_y, 1.0)
lm[ms.R_KNEE] = (CENTRE_X + knee_half, knee_y, 1.0)
lm[ms.L_ANKLE] = (CENTRE_X - ankle_half, ankle_y, 1.0)
lm[ms.R_ANKLE] = (CENTRE_X + ankle_half, ankle_y, 1.0)

# ── Silhouette ───────────────────────────────────────────────────────────────
mask = np.zeros((IMG_H, IMG_W), dtype=bool)


def fill(y: int, centre: float, width: float) -> None:
    if not 0 <= y < IMG_H or width <= 0:
        return
    a = max(0, int(round(centre - width / 2)))
    b = min(IMG_W, int(round(centre + width / 2)))
    mask[y, a:b] = True


def interp(y: float, pts: list[tuple[float, float]]) -> float:
    ys = [p[0] for p in pts]
    return float(np.interp(y, ys, [p[1] for p in pts]))


head_top = y_at(1.0)
torso_pts = [
    (head_top, px(15)),                       # head breadth
    (level_y["neck"], target_w["neck"]),
    (level_y["chest"], target_w["chest"]),
    (level_y["waist"], target_w["waist"]),
    (level_y["hips"], target_w["hips"]),
]
for y in range(int(head_top), int(level_y["hips"]) + 12):
    fill(y, CENTRE_X, interp(y, torso_pts))

# The biceps is widest about a third of the way down the upper arm, which is
# also where the service samples it.
biceps_y = shoulder_y + 0.35 * (elbow_y - shoulder_y)
arm_pts = [(shoulder_y, target_w["upper_arm"] * 0.96),
           (biceps_y, target_w["upper_arm"]),
           (wrist_y, px(5.5))]
for y in range(int(shoulder_y), int(wrist_y) + 1):
    t = (y - shoulder_y) / (wrist_y - shoulder_y)
    w = interp(y, arm_pts)
    for sign in (-1, 1):
        start = CENTRE_X + sign * shoulder_half
        end = CENTRE_X + sign * wrist_x_off
        fill(y, start + t * (end - start), w)

leg_pts = [(hip_y, target_w["thigh"] * 1.06), (thigh_y, target_w["thigh"]),
           (calf_y, target_w["calf"]), (ankle_y, px(7))]
for y in range(int(level_y["hips"]) + 12, int(ankle_y) + 1):
    t = (y - hip_y) / leg_px
    w = interp(y, leg_pts)
    for sign in (-1, 1):
        start = CENTRE_X + sign * hip_half
        end = CENTRE_X + sign * ankle_half
        fill(y, start + t * (end - start), w)

print(f"silhouette coverage = {mask.mean():.3f}")

# ── Inject and run ───────────────────────────────────────────────────────────
ms._detect = lambda arr: (lm, mask)
buf = io.BytesIO()
Image.new("RGB", (IMG_W, IMG_H), "white").save(buf, format="JPEG")
photo = base64.b64encode(buf.getvalue()).decode()

analysis = ms.analyze_front(photo, HEIGHT_CM, GENDER, stats)
print(f"\nquality={analysis.quality} frame_scale={analysis.frame_scale} warnings={analysis.warnings}")

print("\n=== every girth must come from the silhouette, not a guess ===")
for zone in ("neck", "chest", "waist", "hips", "thigh", "calf", "upper_arm"):
    src = analysis.sources.get(zone, "MISSING")
    print(f"  {zone:10s} source={src}")
    assert src == "mask", f"{zone} was not measured from the silhouette ({src})"

print("\n=== recovered girth vs the girth the silhouette was drawn from ===")
worst = 0.0
for zone in ("neck", "chest", "waist", "hips", "thigh", "calf", "upper_arm"):
    got, want = analysis.measurements[zone], stats[zone]
    err = abs(got - want)
    worst = max(worst, err)
    print(f"  {zone:10s} got={got:6.1f} expected={want:6.1f} err={err:4.1f} cm")
    assert err <= 2.0, f"{zone} off by {err:.1f} cm"
print(f"  worst error {worst:.1f} cm")

print("\n=== the old code's failure mode must not reappear ===")
# Measuring the gap between the two leg axes instead of a limb gave ~78 cm.
assert analysis.measurements["thigh"] < 65, analysis.measurements["thigh"]
assert analysis.measurements["calf"] < 45, analysis.measurements["calf"]
print(f"  thigh={analysis.measurements['thigh']} (was ~78 before), "
      f"calf={analysis.measurements['calf']} (was ~57 before)")

print("\n=== lengths ===")
for key in ("shoulder", "back_length", "inseam", "sleeve", "trouser_length", "dress_length"):
    got, want = analysis.measurements[key], stats[key]
    print(f"  {key:15s} got={got:6.1f} stats={want:6.1f} diff={got - want:+5.1f} cm")
    assert abs(got - want) <= 6.0, f"{key} off by {got - want:+.1f} cm"

print("\n=== full engine result ===")
r = engine.run(GENDER, HEIGHT_CM, WEIGHT_KG, 28, front_photo=photo)
print(f"  mode={r['mode']} confidence={r['confidence']} quality={r['quality']} "
      f"photos_used={r['photos_used']} warnings={r['warnings']}")
assert r["mode"] == "photo", r["mode"]
assert r["photos_used"] == 1
assert r["confidence"] >= 0.85, r["confidence"]
m = r["measurements"]
print(f"  chest={m['chest']} waist={m['waist']} hips={m['hips']} thigh={m['thigh']} calf={m['calf']}")
assert m["trouser_length"] - m["inseam"] >= 18.0

print("\n=== a turned-away subject must be refused, not measured ===")
narrow = dict(lm)
narrow[ms.L_SHOULDER] = (CENTRE_X - 18, shoulder_y, 1.0)
narrow[ms.R_SHOULDER] = (CENTRE_X + 18, shoulder_y, 1.0)
ms._detect = lambda arr: (narrow, mask)
r2 = engine.run(GENDER, HEIGHT_CM, WEIGHT_KG, 28, front_photo=photo)
print(f"  mode={r2['mode']} warnings={r2['warnings']}")
assert r2["mode"] == "stats_fallback" and "turned_away_from_camera" in r2["warnings"]

print("\n=== a photo cropped at the knees must be refused ===")
cropped = {i: (x, y, 0.1 if i in (ms.L_ANKLE, ms.R_ANKLE) else v)
           for i, (x, y, v) in lm.items()}
ms._detect = lambda arr: (cropped, mask)
r3 = engine.run(GENDER, HEIGHT_CM, WEIGHT_KG, 28, front_photo=photo)
print(f"  mode={r3['mode']} warnings={r3['warnings']}")
assert r3["mode"] == "stats_fallback" and "body_not_fully_visible" in r3["warnings"]

print("\n=== no silhouette: lengths still used, girths fall back ===")
ms._detect = lambda arr: (lm, None)
r4 = engine.run(GENDER, HEIGHT_CM, WEIGHT_KG, 28, front_photo=photo)
print(f"  mode={r4['mode']} conf={r4['confidence']} warnings={r4['warnings']}")
assert r4["mode"] == "photo" and "silhouette_unavailable" in r4["warnings"]
assert r4["confidence"] <= 0.78, r4["confidence"]

print("\nALL OK")
