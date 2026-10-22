"""Point d'entrée de l'API Osho — FastAPI."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_VERSION, settings
from app.routers import health, measurements

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="Osho API",
    description=(
        "Osho Tailor API — standalone body measurement engine.\n\n"
        "**Stats mode** (no photo): anthropometric regression, instant.\n\n"
        "**Photo mode** (front_photo): pose landmarks for body levels and "
        "lengths, person silhouette for real body widths, ellipse model for "
        "girths, cross-checked against the statistical estimate."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Auth is a Bearer token, never a cookie, so credentials are not needed — and
# allow_credentials with a wildcard origin is rejected by browsers anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health.router)
app.include_router(measurements.router, prefix="/api/v1")
