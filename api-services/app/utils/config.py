"""Typed configuration loaded from api-services/config/.env via pydantic-settings.

Usage:
    from app.utils.config import settings
    print(settings.api_port)
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="config/.env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    camera_url: str = ""
    log_level: str = "INFO"


settings = Settings()
