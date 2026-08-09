from __future__ import annotations

from typing import Dict

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def _available_archive_name(connection, base_name: str) -> str:
    tables = set(inspect(connection).get_table_names())
    if base_name not in tables:
        return base_name
    suffix = 2
    while f"{base_name}_{suffix}" in tables:
        suffix += 1
    return f"{base_name}_{suffix}"


def archive_phase10_notification_tables(engine: Engine) -> Dict[str, str]:
    """Archive incompatible Phase 10 tables before SQLAlchemy creates Phase 11 tables."""
    archived: Dict[str, str] = {}
    with engine.begin() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())

        if "notifications" in tables:
            columns = {column["name"] for column in inspector.get_columns("notifications")}
            is_phase11 = {"watch_session_id", "trend_key", "platform", "detected_at"}.issubset(columns)
            if not is_phase11:
                archive_name = _available_archive_name(connection, "notifications_legacy_phase10")
                if connection.dialect.name == "mysql":
                    connection.execute(text(f"RENAME TABLE notifications TO {archive_name}"))
                else:
                    connection.execute(text(f"ALTER TABLE notifications RENAME TO {archive_name}"))
                archived["notifications"] = archive_name

        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        if "user_trend_snapshots" in tables:
            archive_name = _available_archive_name(connection, "user_trend_snapshots_legacy_phase10")
            if connection.dialect.name == "mysql":
                connection.execute(text(f"RENAME TABLE user_trend_snapshots TO {archive_name}"))
            else:
                connection.execute(text(f"ALTER TABLE user_trend_snapshots RENAME TO {archive_name}"))
            archived["user_trend_snapshots"] = archive_name

    return archived
