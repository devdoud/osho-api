"""Download and cache MediaPipe .task / .tflite model files."""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(
    os.environ.get("MEDIAPIPE_MODEL_CACHE", Path.home() / ".osho_api_models")
)


def ensure_model(url: str, filename: str) -> Path | None:
    """Return the cached model path, downloading it once if needed."""
    path = CACHE_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path

    tmp: str | None = None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Download to a sibling temp file and rename, so an interrupted or
        # concurrent download never leaves a truncated model cached for good.
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".part")
        os.close(fd)
        logger.info("Downloading model %s", filename)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        tmp = None
        logger.info("Model cached at %s", path)
        return path
    except Exception as exc:
        logger.warning("Model %s unavailable: %s", filename, exc)
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
