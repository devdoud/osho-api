"""Dépendances FastAPI réutilisables — notamment l'authentification JWT."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=True)

# Guard for the case where JWT_SECRET is set to a Supabase project's JWT
# secret: Supabase signs its anon API key with that same secret, and the anon
# key ships inside the mobile app, so it is effectively public.
_FORBIDDEN_ROLES = {"anon", "service_role"}

_INVALID = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Token invalide ou expiré.",
    headers={"WWW-Authenticate": "Bearer"},
)


class TokenPayload(BaseModel):
    sub: str  # user_id
    exp: Optional[int] = None


def create_access_token(user_id: str) -> str:
    """Crée un JWT signé valable `jwt_expire_minutes` minutes."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> TokenPayload:
    """
    Vérifie le Bearer token JWT émis par cette API — en pratique le jeton de
    service que présente l'Edge Function Supabase (voir
    scripts/make_service_token.py).
    """
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            # Un token Supabase porte aud="authenticated", un token émis ici
            # n'en porte aucun : vérifier l'audience rejetterait l'un ou
            # l'autre. La signature et l'expiration restent vérifiées.
            options={"verify_aud": False},
        )
    except JWTError as exc:
        logger.info("Token rejected: %s", exc)
        raise _INVALID

    sub = payload.get("sub") or ""
    if not sub:
        logger.info("Token rejected: no sub claim")
        raise _INVALID
    if payload.get("role") in _FORBIDDEN_ROLES:
        logger.info("Token rejected: role=%s is not a user token", payload.get("role"))
        raise _INVALID

    return TokenPayload(sub=sub, exp=payload.get("exp"))
