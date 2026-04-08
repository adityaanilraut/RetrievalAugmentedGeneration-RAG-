from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-5-mini"
    openai_embed_model: str = "text-embedding-3-small"
    # text-embedding-3-small default dim; override if using dimensions= on API
    embedding_dimensions: int = 1536

    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
