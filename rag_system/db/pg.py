from __future__ import annotations

import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from rag_system.config import get_settings


def connect():
    settings = get_settings()
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_db() -> None:
    """Create extension and tables from bundled schema.sql."""
    settings = get_settings()
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    sql_schema = schema_path.read_text(encoding="utf-8")
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        for stmt in sql_schema.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.commit()


def check_db() -> str:
    try:
        with connect() as conn:
            conn.execute("SELECT 1;")
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"error: {e!s}"


def env_database_url() -> str:
    return os.environ.get("DATABASE_URL", get_settings().database_url)
