from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()


def _build_url() -> str:
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


@lru_cache(maxsize=1)
def get_engine(pool_size: int = 5, max_overflow: int = 10) -> Engine:
    url = _build_url()
    return create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,  # drop stale connections before use
        echo=False,
    )


def get_session() -> Session:
    """Return a new SQLAlchemy Session bound to the shared engine."""
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return factory()


@contextmanager
def session_scope():
    """Context manager that commits on exit and rolls back on exception."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Return True if the database is reachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        print(f"[db] connection failed: {exc}")
        return False
