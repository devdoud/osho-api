import json, os, sys, urllib.error, urllib.request
from pathlib import Path
os.environ["JWT_SECRET"] = "test-secret-for-local-verification"
os.environ["DEBUG"] = "false"
os.environ["ALLOWED_ORIGINS"] = "https://app.example.com"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = os.environ.get("OSHO_API_BASE", "http://127.0.0.1:8000")


def call(path, payload=None, token=None, origin=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {}
    if data:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if origin:
        headers["Origin"] = origin
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            body = json.loads(body)
        except Exception:
            body = body.decode(errors="replace")[:200]
        return e.code, body, dict(e.headers)


from app.dependencies import create_access_token
token = create_access_token("user-abc-123")

print("=== GET /health ===")
print(" ", call("/health")[:2])

print("=== no token must be 403/401 ===")
code, body, _ = call("/api/v1/measurements", {"gender": "female", "height": 165, "weight": 62})
print(f"  status={code} body={body}")
assert code in (401, 403), code

print("=== wrong-secret token must be 401 ===")
from jose import jwt
bad = jwt.encode({"sub": "x", "exp": 9999999999}, "a-different-secret", algorithm="HS256")
code, body, _ = call("/api/v1/measurements", {"gender": "female", "height": 165, "weight": 62}, token=bad)
print(f"  status={code}")
assert code == 401, code

print("=== a real Supabase access token must be accepted ===")
import time
SECRET = os.environ["JWT_SECRET"]
supa = jwt.encode({
    "iss": "https://abcdefgh.supabase.co/auth/v1",
    "sub": "9f1c7e2a-0000-4a1b-8c3d-aaaabbbbcccc",
    "aud": "authenticated",          # present on every Supabase user token
    "role": "authenticated",
    "email": "user@example.com",
    "exp": int(time.time()) + 3600,
    "iat": int(time.time()),
}, SECRET, algorithm="HS256")
code, body, _ = call("/api/v1/measurements",
                     {"gender": "female", "height": 165, "weight": 62}, token=supa)
print(f"  status={code} chest={body.get('chest') if code == 200 else body}")
assert code == 200, body

print("=== the public anon key must NOT authenticate ===")
for role in ("anon", "service_role"):
    key = jwt.encode({"iss": "supabase", "ref": "abcdefgh", "role": role,
                      "iat": int(time.time()), "exp": int(time.time()) + 3600},
                     SECRET, algorithm="HS256")
    code, _, _ = call("/api/v1/measurements",
                      {"gender": "female", "height": 165, "weight": 62}, token=key)
    print(f"  role={role:13s} -> {code}")
    assert code == 401, (role, code)
# A role-bearing token that does carry a sub must still be refused.
svc = jwt.encode({"sub": "someone", "role": "service_role",
                  "exp": int(time.time()) + 3600}, SECRET, algorithm="HS256")
code, _, _ = call("/api/v1/measurements",
                  {"gender": "female", "height": 165, "weight": 62}, token=svc)
print(f"  service_role with sub -> {code}")
assert code == 401, code

print("=== expired token must be refused ===")
old = jwt.encode({"sub": "u", "aud": "authenticated",
                  "exp": int(time.time()) - 60}, SECRET, algorithm="HS256")
code, _, _ = call("/api/v1/measurements",
                  {"gender": "female", "height": 165, "weight": 62}, token=old)
print(f"  expired -> {code}")
assert code == 401, code

print("=== valid token, stats mode ===")
code, body, _ = call("/api/v1/measurements",
                     {"gender": "female", "height": 165, "weight": 62, "age": 28}, token=token)
print(f"  status={code} mode={body.get('mode')} conf={body.get('confidence')}")
assert code == 200, body
required = ["chest", "waist", "hips", "neck", "thigh", "calf", "wrist", "upper_arm",
            "shoulder", "inseam", "back_length", "sleeve", "bust_height",
            "crotch_depth", "trouser_length", "dress_length",
            "mode", "confidence", "quality", "photos_used", "warnings"]
missing = [k for k in required if k not in body]
assert not missing, f"missing from response: {missing}"
print(f"  all {len(required)} fields present; dress_length={body['dress_length']}")

print("=== male has no dress_length ===")
code, body, _ = call("/api/v1/measurements",
                     {"gender": "male", "height": 178, "weight": 80}, token=token)
assert code == 200 and body["dress_length"] is None, body.get("dress_length")
print(f"  ok, chest={body['chest']} waist={body['waist']}")

print("=== validation rejects out-of-range and bad enum ===")
for bad_payload, label in [
    ({"gender": "female", "height": 60, "weight": 62}, "height too small"),
    ({"gender": "female", "height": 165, "weight": 500}, "weight too big"),
    ({"gender": "alien", "height": 165, "weight": 62}, "bad gender"),
    ({"gender": "female", "height": 165, "weight": 62, "front_photo": "tiny"}, "photo too small"),
]:
    code, _, _ = call("/api/v1/measurements", bad_payload, token=token)
    print(f"  {label:18s} -> {code}")
    assert code == 422, (label, code)

print("=== CORS is restricted to the configured origin ===")
_, _, hdrs = call("/health", origin="https://app.example.com")
allow = hdrs.get("access-control-allow-origin")
print(f"  allowed origin header={allow}")
assert allow == "https://app.example.com", allow
_, _, hdrs = call("/health", origin="https://evil.example.com")
print(f"  evil origin header={hdrs.get('access-control-allow-origin')}")
assert hdrs.get("access-control-allow-origin") is None

print("=== photo mode degrades cleanly on a non-body image ===")
import base64, io
from PIL import Image
buf = io.BytesIO()
Image.new("RGB", (900, 1600), (210, 205, 200)).save(buf, format="JPEG")
code, body, _ = call("/api/v1/measurements",
                     {"gender": "female", "height": 165, "weight": 62,
                      "front_photo": base64.b64encode(buf.getvalue()).decode()},
                     token=token)
print(f"  status={code} mode={body.get('mode')} conf={body.get('confidence')} warnings={body.get('warnings')}")
assert code == 200 and body["mode"] == "stats_fallback", body.get("mode")
assert body["warnings"] == ["no_body_detected"], body["warnings"]
assert body["chest"] > 0

print("\nALL OK")
