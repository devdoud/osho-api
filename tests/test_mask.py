import base64, io, os, sys
from pathlib import Path
os.environ["DEBUG"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from app.services import segmentation_service as seg
from app.services.mediapipe_service import _decode_b64, _load_image

W, H = 400, 800
m = np.zeros((H, W), dtype=bool)

# Chest row y=200: left arm 60-75, torso 150-250, right arm 325-340
m[195:206,  60:76]  = True
m[195:206, 150:251] = True
m[195:206, 325:341] = True
# Thigh row y=500: left leg 160-199, right leg 205-244 (separated)
m[495:506, 160:200] = True
m[495:506, 205:245] = True
# Calf row y=650: legs touching -> single run 170-230
m[645:656, 170:231] = True

print("=== torso width must ignore the arms ===")
w = seg.torso_width_px(m, 200, centre_x=200)
print(f"  measured={w}  expected=101 (torso only, not 281 arm-to-arm)")
assert w == 101, w

print("=== each leg measured separately ===")
left  = seg.limb_width_px(m, 500, want_x=180, other_x=225)
right = seg.limb_width_px(m, 500, want_x=225, other_x=180)
print(f"  left={left} right={right}  expected 40 / 40")
assert left == 40 and right == 40, (left, right)

print("=== touching legs: single run is halved ===")
lw = seg.limb_width_px(m, 650, want_x=185, other_x=215)
print(f"  measured={lw}  expected=30.5 (61 / 2)")
assert abs(lw - 30.5) < 0.01, lw

print("=== speckle below 1.2% of width is ignored ===")
spk = np.zeros((H, W), dtype=bool)
spk[300, 10:13] = True
print(f"  runs={seg.runs_in_row(spk, 300)}  expected []")
assert seg.runs_in_row(spk, 300) == []

print("=== body touching the frame edge still bounded ===")
edge = np.zeros((H, W), dtype=bool)
edge[400, 0:50] = True
print(f"  runs={seg.runs_in_row(edge, 400)}  expected [(0, 49)]")
assert seg.runs_in_row(edge, 400) == [(0, 49)]

print("=== mask coverage gate ===")
assert seg.to_bool_mask(np.ones((10, 10), dtype=np.float32)) is None, "all-person accepted"
assert seg.to_bool_mask(np.zeros((10, 10), dtype=np.float32)) is None, "empty accepted"
half = np.zeros((10, 10), dtype=np.float32); half[:5] = 0.9
assert seg.to_bool_mask(half) is not None, "valid mask rejected"
print("  all-person rejected, empty rejected, half accepted")

print("=== base64 padding: every length class must decode ===")
for n in range(1, 9):
    raw = bytes(range(n))
    std = base64.b64encode(raw).decode()
    for variant, label in [(std, "padded"), (std.rstrip("="), "unpadded"),
                           (std + "\n ", "whitespace")]:
        got = _decode_b64(variant)
        assert got == raw, f"{label} n={n}: {got!r} != {raw!r}"
print("  padded / unpadded / whitespace all decode for lengths 1-8")

print("=== EXIF orientation is applied ===")
img = Image.new("RGB", (60, 120), "white")
buf = io.BytesIO()
# Orientation 6 = rotate 90 CW on display, so a 60x120 file renders as 120x60.
exif = img.getexif(); exif[274] = 6
img.save(buf, format="JPEG", exif=exif)
arr = _load_image(base64.b64encode(buf.getvalue()).decode())
print(f"  stored 60x120 with orientation=6 -> loaded shape {arr.shape[:2]} (expect 60 rows, 120 cols)")
assert arr.shape[:2] == (60, 120), arr.shape

print("\nALL OK")
