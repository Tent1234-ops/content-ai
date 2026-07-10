import importlib.util
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings


def _resolve_mysql_driver() -> str | None:
    if importlib.util.find_spec("pymysql"):
        return "pymysql"
    return None


MYSQL_DRIVER = _resolve_mysql_driver()


def _mysql_server_url() -> str:
    return f"mysql+{MYSQL_DRIVER}://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}"


def _mysql_database_url() -> str:
    return f"{_mysql_server_url()}/{settings.db_name}"


def _build_database_url() -> str:
    explicit_url = os.environ.get("DATABASE_URL")
    if explicit_url:
        return explicit_url

    if settings.db_driver.lower() == "mysql" and MYSQL_DRIVER:
        return _mysql_database_url()

    return "sqlite:///./app.db"


def _ensure_mysql_database() -> None:
    bootstrap_engine = create_engine(_mysql_server_url(), pool_pre_ping=True)
    with bootstrap_engine.connect() as connection:
        connection.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{settings.db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
        connection.commit()
    bootstrap_engine.dispose()


def _ensure_mysql_schema_compat() -> None:
    bootstrap_engine = create_engine(_mysql_database_url(), pool_pre_ping=True)
    with bootstrap_engine.connect() as connection:
        column_exists = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = :schema_name
                  AND TABLE_NAME = 'dataset_contents'
                  AND COLUMN_NAME = 'duration_seconds'
                """
            ),
            {"schema_name": settings.db_name},
        ).scalar()
        if not column_exists:
            connection.execute(text("ALTER TABLE dataset_contents ADD COLUMN duration_seconds INT NULL"))
            connection.commit()
    bootstrap_engine.dispose()


DATABASE_URL = _build_database_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite")
USING_MYSQL = settings.db_driver.lower() == "mysql" and MYSQL_DRIVER is not None
DB_BOOTSTRAP_ERROR = None

if USING_MYSQL:
    try:
        _ensure_mysql_database()
        _ensure_mysql_schema_compat()
    except Exception as exc:
        DB_BOOTSTRAP_ERROR = f"{exc.__class__.__name__}: {exc}"

engine_kwargs = {"pool_pre_ping": True}
if IS_SQLITE:
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
