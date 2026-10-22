# Tests

No test framework needed — each file is a script that asserts and exits non-zero
on failure. Run them with the project's virtualenv active.

```bash
python tests/test_engine.py       # geometry round-trip, stats across the size range
python tests/test_mask.py         # silhouette row scanning, base64, EXIF
python tests/test_photo_path.py   # full photo pipeline against a synthetic body
```

`test_api_live.py` needs a running server, because it checks auth and CORS
through real HTTP:

```bash
JWT_SECRET=test-secret DEBUG=false ALLOWED_ORIGINS=https://app.example.com \
  uvicorn app.main:app --port 8000
OSHO_API_BASE=http://127.0.0.1:8000 python tests/test_api_live.py
```

The test secret must match the one the server runs with, and `ALLOWED_ORIGINS`
must be `https://app.example.com` for the CORS assertions.

## What these lock down

Each suite guards a bug that shipped in the first version of this engine:

- **test_photo_path** — thigh and calf girth were derived from the *gap between
  the two legs* rather than a limb width, so a 57 cm thigh was reported as
  ~78 cm. It asserts every girth is sourced from the silhouette and lands within
  2 cm of the body the silhouette was drawn from, and that a turned-away or
  cropped subject is refused instead of measured.
- **test_mask** — EXIF orientation was ignored, so portrait phone photos were
  analysed sideways; and base64 was padded by appending `==`, which raises on
  payloads whose length is 3 mod 4.
- **test_api_live** — Supabase access tokens carry `aud="authenticated"`, which
  python-jose rejects unless audience verification is disabled, so every real
  user token returned 401. It also checks the public anon key cannot
  authenticate, since Supabase signs it with the same secret.
- **test_engine** — unreadable photos must degrade to the statistical estimate,
  never crash or return a girth of zero.
