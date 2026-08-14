from __future__ import annotations

from typing import Dict

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


PHASE13_DATASET_COLUMNS = {
    "dataset_source": "VARCHAR(100) NOT NULL DEFAULT 'legacy'",
    "dataset_version": "VARCHAR(100) NOT NULL DEFAULT 'legacy-v1'",
    "source_record_id": "VARCHAR(255) NULL",
    "source_creator": "VARCHAR(255) NULL",
    "taxonomy_version": "VARCHAR(50) NOT NULL DEFAULT 'legacy-v1'",
    "taxonomy_leaf_key": "VARCHAR(100) NULL",
    "category_level_1": "VARCHAR(150) NULL",
    "category_level_2": "VARCHAR(150) NULL",
    "category_level_3": "VARCHAR(150) NULL",
    "language": "VARCHAR(20) NOT NULL DEFAULT 'und'",
    "verification_status": "VARCHAR(30) NOT NULL DEFAULT 'unverified'",
    "label_source": "VARCHAR(100) NOT NULL DEFAULT 'unverified'",
    "license_name": "VARCHAR(100) NOT NULL DEFAULT 'unknown'",
    "license_url": "VARCHAR(1024) NULL",
    "data_split": "VARCHAR(20) NOT NULL DEFAULT 'unassigned'",
    "is_active": "BOOLEAN NOT NULL DEFAULT TRUE",
}

PHASE13_ANALYSIS_COLUMNS = {
    "classification_model_id": "INT NULL",
    "taxonomy_version": "VARCHAR(50) NULL",
    "taxonomy_leaf_key": "VARCHAR(100) NULL",
    "category_level_1": "VARCHAR(150) NULL",
    "category_level_2": "VARCHAR(150) NULL",
    "category_level_3": "VARCHAR(150) NULL",
    "classification_confidence": "FLOAT NULL",
    "classification_is_unknown": "BOOLEAN NOT NULL DEFAULT FALSE",
}

PHASE13_MODEL_METRIC_COLUMNS = {
    "taxonomy_leaf_key": "VARCHAR(100) NOT NULL DEFAULT '__overall__'",
}

YOUTUBE_CC_DATASET_COLUMNS = {
    "duration_seconds": "INT NULL",
    "collection_run_id": "INT NULL",
    "source_youtube_id": "VARCHAR(32) NULL",
    "source_channel_id": "VARCHAR(64) NULL",
    "source_category": "VARCHAR(100) NULL",
    "source_subcategory": "VARCHAR(100) NULL",
    "collection_query": "VARCHAR(255) NULL",
    "source_release_url": "VARCHAR(1024) NULL",
    "source_archive_sha256": "VARCHAR(64) NULL",
    "source_annotation_path": "VARCHAR(1024) NULL",
    "source_annotation_sha256": "VARCHAR(64) NULL",
    "import_batch_id": "VARCHAR(64) NULL",
    "split_strategy": "VARCHAR(100) NULL",
    "creator_group_key": "VARCHAR(64) NULL",
    "transcript_sha256": "VARCHAR(64) NULL",
    "transcript_segment_count": "INT NOT NULL DEFAULT 0",
    "transcript_start_seconds": "FLOAT NULL",
    "transcript_end_seconds": "FLOAT NULL",
    "transcript_window_seconds": "INT NULL",
    "transcript_source": "VARCHAR(50) NULL",
    "caption_type": "VARCHAR(50) NULL",
    "transcript_quality": "VARCHAR(30) NULL",
    "reviewed_by": "VARCHAR(255) NULL",
    "reviewed_at": "DATETIME NULL",
    "review_notes": "TEXT NULL",
    "statistics_captured_at": "DATETIME NULL",
    "license_verified_at": "DATETIME NULL",
    "raw_metadata_json": "TEXT NULL",
    "collection_strategy": "VARCHAR(50) NULL",
    "average_views_per_day": "FLOAT NOT NULL DEFAULT 0",
    "engagement_rate": "FLOAT NOT NULL DEFAULT 0",
    "is_training_eligible": "BOOLEAN NOT NULL DEFAULT FALSE",
    "is_keyword_recommendation_eligible": "BOOLEAN NOT NULL DEFAULT FALSE",
    "is_duration_recommendation_eligible": "BOOLEAN NOT NULL DEFAULT FALSE",
}

YOUTUBE_CC_COLLECTION_RUN_COLUMNS = {
    "review_artifact_path": "VARCHAR(1024) NULL",
    "review_artifact_sha256": "VARCHAR(64) NULL",
    "duplicates_skipped": "INT NOT NULL DEFAULT 0",
    "resume_count": "INT NOT NULL DEFAULT 0",
    "last_resumed_at": "DATETIME NULL",
}


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


def migrate_phase13_taxonomy_schema(engine: Engine) -> Dict[str, object]:
    """Add Phase 13 columns to existing databases without rewriting legacy rows."""
    added_columns: list[str] = []
    added_indexes: list[str] = []
    added_constraints: list[str] = []
    with engine.begin() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())

        for table_name, definitions in (
            ("dataset_contents", PHASE13_DATASET_COLUMNS),
            ("analysis_results", PHASE13_ANALYSIS_COLUMNS),
            ("model_evaluation_metrics", PHASE13_MODEL_METRIC_COLUMNS),
        ):
            if table_name not in tables:
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, sql_type in definitions.items():
                if column_name in existing:
                    continue
                connection.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type}")
                )
                added_columns.append(f"{table_name}.{column_name}")

        inspector = inspect(connection)

        def ensure_index(
            table_name: str,
            index_name: str,
            columns: tuple[str, ...],
            *,
            unique: bool = False,
        ) -> None:
            if table_name not in tables:
                return
            index_names = {
                str(index.get("name")) for index in inspector.get_indexes(table_name)
            }
            constraint_names = {
                str(item.get("name"))
                for item in inspector.get_unique_constraints(table_name)
            }
            if index_name in index_names or index_name in constraint_names:
                return
            unique_sql = "UNIQUE " if unique else ""
            connection.execute(
                text(
                    f"CREATE {unique_sql}INDEX {index_name} ON {table_name} "
                    f"({', '.join(columns)})"
                )
            )
            added_indexes.append(index_name)

        ensure_index(
            "dataset_contents",
            "uq_dataset_source_version_record",
            ("dataset_source", "dataset_version", "source_record_id"),
            unique=True,
        )
        ensure_index(
            "dataset_contents",
            "ix_dataset_taxonomy_leaf",
            ("taxonomy_version", "taxonomy_leaf_key"),
        )
        ensure_index(
            "dataset_contents",
            "ix_dataset_contents_taxonomy_leaf_key",
            ("taxonomy_leaf_key",),
        )
        ensure_index(
            "analysis_results",
            "ix_analysis_results_classification_model_id",
            ("classification_model_id",),
        )

        if connection.dialect.name == "mysql" and "model_evaluation_metrics" in tables:
            metric_unique = next(
                (
                    item
                    for item in inspector.get_unique_constraints("model_evaluation_metrics")
                    if item.get("name") == "uq_model_evaluation_metric"
                ),
                None,
            )
            expected_metric_columns = [
                "model_id",
                "dataset_split",
                "language",
                "taxonomy_level",
                "taxonomy_leaf_key",
                "metric_name",
            ]
            if metric_unique and metric_unique.get("column_names") != expected_metric_columns:
                connection.execute(
                    text(
                        "ALTER TABLE model_evaluation_metrics "
                        "DROP INDEX uq_model_evaluation_metric"
                    )
                )
                connection.execute(
                    text(
                        "ALTER TABLE model_evaluation_metrics "
                        "ADD CONSTRAINT uq_model_evaluation_metric UNIQUE "
                        "(model_id, dataset_split, language, taxonomy_level, "
                        "taxonomy_leaf_key, metric_name)"
                    )
                )
                added_constraints.append("uq_model_evaluation_metric:replaced")

        if (
            connection.dialect.name == "mysql"
            and "analysis_results" in tables
            and "classification_models" in tables
        ):
            foreign_keys = inspector.get_foreign_keys("analysis_results")
            has_model_foreign_key = any(
                item.get("constrained_columns") == ["classification_model_id"]
                for item in foreign_keys
            )
            if not has_model_foreign_key:
                constraint_name = "fk_analysis_results_classification_model"
                connection.execute(
                    text(
                        "ALTER TABLE analysis_results "
                        f"ADD CONSTRAINT {constraint_name} "
                        "FOREIGN KEY (classification_model_id) "
                        "REFERENCES classification_models(model_id) ON DELETE SET NULL"
                    )
                )
                added_constraints.append(constraint_name)

    return {
        "added_columns": added_columns,
        "added_indexes": added_indexes,
        "added_constraints": added_constraints,
    }


def migrate_youtube_cc_dataset_schema(engine: Engine) -> Dict[str, object]:
    """Add YouTube CC provenance fields and disable non-production evidence."""
    added_columns: list[str] = []
    added_indexes: list[str] = []
    added_constraints: list[str] = []
    legacy_rows_deactivated = 0

    with engine.begin() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())

        if "dataset_collection_runs" in tables:
            run_columns = {
                column["name"]
                for column in inspector.get_columns("dataset_collection_runs")
            }
            for column_name, sql_type in YOUTUBE_CC_COLLECTION_RUN_COLUMNS.items():
                if column_name in run_columns:
                    continue
                connection.execute(
                    text(
                        "ALTER TABLE dataset_collection_runs "
                        f"ADD COLUMN {column_name} {sql_type}"
                    )
                )
                added_columns.append(f"dataset_collection_runs.{column_name}")

        if "dataset_contents" not in tables:
            return {
                "added_columns": added_columns,
                "added_indexes": added_indexes,
                "added_constraints": added_constraints,
                "legacy_rows_deactivated": legacy_rows_deactivated,
            }

        existing_columns = {
            column["name"] for column in inspector.get_columns("dataset_contents")
        }
        for column_name, sql_type in YOUTUBE_CC_DATASET_COLUMNS.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                text(f"ALTER TABLE dataset_contents ADD COLUMN {column_name} {sql_type}")
            )
            added_columns.append(f"dataset_contents.{column_name}")

        inspector = inspect(connection)
        available_columns = {
            column["name"] for column in inspector.get_columns("dataset_contents")
        }
        index_names = {
            str(item.get("name")) for item in inspector.get_indexes("dataset_contents")
        }
        unique_names = {
            str(item.get("name"))
            for item in inspector.get_unique_constraints("dataset_contents")
        }

        def ensure_index(name: str, columns: tuple[str, ...], *, unique: bool = False) -> None:
            if not set(columns).issubset(available_columns):
                return
            if name in index_names or name in unique_names:
                return
            unique_sql = "UNIQUE " if unique else ""
            connection.execute(
                text(
                    f"CREATE {unique_sql}INDEX {name} ON dataset_contents "
                    f"({', '.join(columns)})"
                )
            )
            added_indexes.append(name)

        ensure_index("uq_dataset_source_youtube_id", ("source_youtube_id",), unique=True)
        ensure_index("uq_dataset_transcript_sha256", ("transcript_sha256",), unique=True)
        ensure_index("ix_dataset_contents_creator_group_key", ("creator_group_key",))
        ensure_index("ix_dataset_contents_source_channel_id", ("source_channel_id",))
        ensure_index("ix_dataset_contents_collection_run_id", ("collection_run_id",))
        ensure_index(
            "ix_dataset_training_eligibility",
            ("is_training_eligible", "data_split", "taxonomy_leaf_key"),
        )

        result = connection.execute(
            text(
                "UPDATE dataset_contents SET is_active = FALSE, "
                "is_training_eligible = FALSE "
                "WHERE (source_platform LIKE '%_seed' "
                "OR source_platform = 'seed' "
                "OR dataset_source IN ('demo_seed', 'mmsum')"
                ") AND (is_active = TRUE OR is_training_eligible = TRUE)"
            )
        )
        legacy_rows_deactivated = max(int(result.rowcount or 0), 0)

        connection.execute(
            text(
                "UPDATE dataset_contents SET "
                "is_keyword_recommendation_eligible = CASE "
                "WHEN dataset_source = 'youtube_cc' "
                "AND is_training_eligible = TRUE "
                "AND is_active = TRUE THEN TRUE ELSE FALSE END"
            )
        )
        connection.execute(
            text(
                "UPDATE dataset_contents SET "
                "is_duration_recommendation_eligible = CASE "
                "WHEN dataset_source = 'youtube_cc' "
                "AND is_training_eligible = TRUE "
                "AND is_active = TRUE "
                "AND duration_seconds > 0 "
                "AND duration_seconds <= 300 THEN TRUE ELSE FALSE END"
            )
        )

        if connection.dialect.name == "mysql" and "dataset_collection_runs" in tables:
            inspector = inspect(connection)
            foreign_keys = inspector.get_foreign_keys("dataset_contents")
            has_collection_run_fk = any(
                item.get("constrained_columns") == ["collection_run_id"]
                for item in foreign_keys
            )
            if not has_collection_run_fk:
                constraint_name = "fk_dataset_contents_collection_run"
                connection.execute(
                    text(
                        "ALTER TABLE dataset_contents "
                        f"ADD CONSTRAINT {constraint_name} "
                        "FOREIGN KEY (collection_run_id) "
                        "REFERENCES dataset_collection_runs(collection_run_id) "
                        "ON DELETE SET NULL"
                    )
                )
                added_constraints.append(constraint_name)

        if "classification_models" in tables:
            connection.execute(
                text(
                    "UPDATE classification_models SET is_active = FALSE, status = 'legacy' "
                    "WHERE training_dataset_source IN ('demo_seed', 'mmsum')"
                )
            )

    return {
        "added_columns": added_columns,
        "added_indexes": added_indexes,
        "added_constraints": added_constraints,
        "legacy_rows_deactivated": legacy_rows_deactivated,
    }
