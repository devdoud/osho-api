from fastapi import APIRouter
from pydantic import BaseModel

from app.config import API_VERSION

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION)
