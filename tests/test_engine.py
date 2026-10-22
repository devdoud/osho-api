import os, sys
from pathlib import Path
os.environ["DEBUG"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.measurement_engine import run
from app.services.geometry import width_to_girth, girth_to_width

print("=== geometry round-trip (must return the input) ===")
for zone, g in [("chest", 91), ("waist", 74), ("hips", 101), ("neck", 33.5),
                ("thigh", 57), ("calf", 36), ("upper_arm", 28)]:
    w = girth_to_width(g, zone, "female")
    back = width_to_girth(w, zone, "female")
    flag = "OK " if abs(back - g) < 0.05 else "FAIL"
    print(f"  {flag} {zone:10s} girth={g:6.1f} -> width={w:5.2f} -> girth={back:6.2f}")

cases = [
    ("female", 165, 62, 28),
    ("male",   175, 75, 30),
    ("female", 170, 85, 42),
    ("male",   180, 95, 45),
    ("female", 150, 45, 20),
]
print("\n=== stats mode ===")
for gender, h, w, age in cases:
    r = run(gender, h, w, age)
    m = r["measurements"]
    print(f"\n{gender} {h}cm {w}kg age{age}  mode={r['mode']} conf={r['confidence']}")
    print(f"  chest={m['chest']} waist={m['waist']} hips={m['hips']} shoulder={m['shoulder']} neck={m['neck']}")
    print(f"  thigh={m['thigh']} calf={m['calf']} wrist={m['wrist']} upper_arm={m['upper_arm']}")
    print(f"  inseam={m['inseam']} trouser={m['trouser_length']} back={m['back_length']} sleeve={m['sleeve']}")
    print(f"  bust_h={m['bust_height']} crotch_d={m['crotch_depth']} dress={m.get('dress_length')}")
    assert m["trouser_length"] - m["inseam"] >= 18.0, "rise constraint violated"
    assert r["warnings"] == [], r["warnings"]

print("\n=== garbage photo must degrade, never crash ===")
for label, payload in [
    ("not base64",   "!!!!not-base64!!!!" * 40),
    ("odd padding",  "QUJDRA" * 200),
    ("data uri txt", "data:image/jpeg;base64," + ("A" * 2000)),
]:
    r = run("female", 165, 62, 28, front_photo=payload)
    print(f"  {label:14s} mode={r['mode']} conf={r['confidence']} warnings={r['warnings']}")
    assert r["mode"] == "stats_fallback", r["mode"]
    assert r["measurements"]["chest"] > 0

print("\nALL OK")
