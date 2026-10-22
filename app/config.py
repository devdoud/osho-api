from pydantic_settings import BaseSettings, SettingsConfigDict

API_VERSION = "2.0.0"

_DEFAULT_SECRET = "change_me_in_production"


class Settings(BaseSettings):
    # extra="ignore" so unrelated variables already present in a deployment's
    # environment never break startup.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # This API's own signing secret, independent of Supabase.
    jwt_secret: str = _DEFAULT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080  # 7 days

    # Comma-separated list of allowed origins, or "*".
    allowed_origins: str = "*"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()

if not settings.debug and settings.jwt_secret == _DEFAULT_SECRET:
    raise RuntimeError(
        "JWT_SECRET is still the placeholder. Set it to a long random value "
        "before running in production — see .env.example."
    )
