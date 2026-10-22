# MediaPipe does not publish wheels for 3.13 yet.
FROM python:3.11-slim

WORKDIR /app

# mediapipe pulls in opencv-contrib-python, which needs libGL at runtime.
# Note: on Debian 12 (bookworm) the package is libgl1 — libgl1-mesa-glx was
# removed, and asking for it makes the build fail.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY run.py .

# Ship the pose model in the image. Downloading it on the first request would
# stall that request by ~30 MB and can trip the platform's request timeout.
ENV MEDIAPIPE_MODEL_CACHE=/opt/models
RUN python -c "\
from app.services.mediapipe_service import _MODEL_URL, _MODEL_FILE; \
from app.services.model_store import ensure_model; \
assert ensure_model(_MODEL_URL, _MODEL_FILE), 'pose model download failed'"

EXPOSE 8000

# Shell form so $PORT, which most hosts inject, is honoured.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
